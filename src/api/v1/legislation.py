from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.models.legislation import Bill, Committee, CommitteeHearing
from src.processing.timing_analysis import analyze_trade_context, get_trades_for_committee
from src.schemas.legislation import (
    BillResponse,
    CommitteeResponse,
    CommitteeTradeResponse,
    HearingResponse,
    LegislativeContextResponse,
)

router = APIRouter(tags=["legislation"])


# --- Bills ---


@router.get("/bills", response_model=list[BillResponse])
async def list_bills(
    congress: int | None = Query(None),
    bill_type: str | None = Query(None),
    policy_area: str | None = Query(None),
    sponsor_bioguide_id: str | None = Query(None),
    date_from: date | None = Query(None),
    date_to: date | None = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List bills with filters."""
    query = select(Bill)

    if congress:
        query = query.where(Bill.congress_number == congress)
    if bill_type:
        query = query.where(Bill.bill_type == bill_type.lower())
    if policy_area:
        query = query.where(Bill.policy_area.ilike(f"%{policy_area}%"))
    if sponsor_bioguide_id:
        query = query.where(Bill.sponsor_bioguide_id == sponsor_bioguide_id)
    if date_from:
        query = query.where(Bill.introduced_date >= date_from)
    if date_to:
        query = query.where(Bill.introduced_date <= date_to)

    query = query.order_by(Bill.introduced_date.desc()).limit(limit).offset(offset)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/bills/{bill_id}", response_model=BillResponse)
async def get_bill(
    bill_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific bill by its bill_id (e.g. hr1234-118)."""
    result = await db.execute(select(Bill).where(Bill.bill_id == bill_id))
    bill = result.scalar_one_or_none()
    if not bill:
        raise HTTPException(status_code=404, detail="Bill not found")
    return bill


# --- Committees ---


@router.get("/committees", response_model=list[CommitteeResponse])
async def list_committees(
    chamber: str | None = Query(None),
    current_only: bool = Query(True),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List committees with optional filters."""
    query = select(Committee)

    if chamber:
        query = query.where(Committee.chamber == chamber.lower())
    if current_only:
        query = query.where(Committee.is_current == True)  # noqa: E712

    query = query.order_by(Committee.name).limit(limit).offset(offset)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/committees/{system_code}", response_model=CommitteeResponse)
async def get_committee(
    system_code: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific committee by system code (e.g. HSBA00)."""
    result = await db.execute(
        select(Committee).where(Committee.system_code == system_code)
    )
    committee = result.scalar_one_or_none()
    if not committee:
        raise HTTPException(status_code=404, detail="Committee not found")
    return committee


@router.get("/committees/{system_code}/trades", response_model=list[CommitteeTradeResponse])
async def get_committee_trades(
    system_code: str,
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Get all trades by members who sit on this committee."""
    trades = await get_trades_for_committee(db, system_code, limit=limit, offset=offset)
    return trades


@router.get("/committees/{system_code}/hearings", response_model=list[HearingResponse])
async def get_committee_hearings(
    system_code: str,
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Get hearings for a specific committee."""
    result = await db.execute(
        select(CommitteeHearing)
        .where(CommitteeHearing.committee_code == system_code)
        .order_by(CommitteeHearing.hearing_date.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()


# --- Legislative Context for Trades ---


@router.get(
    "/trades/{trade_id}/legislative-context",
    response_model=LegislativeContextResponse,
)
async def get_trade_legislative_context(
    trade_id: int,
    hearing_window: int = Query(30, le=90, description="Days ± for hearing search"),
    bill_window: int = Query(90, le=180, description="Days ± for bill search"),
    db: AsyncSession = Depends(get_db),
):
    """Get legislative context for a specific trade.

    Returns nearby committee hearings, related bills, votes, sector alignment,
    and a heuristic timing suspicion score.
    """
    context = await analyze_trade_context(
        db, trade_id,
        hearing_window_days=hearing_window,
        bill_window_days=bill_window,
    )
    if not context:
        raise HTTPException(status_code=404, detail="Trade not found")

    return LegislativeContextResponse(
        trade_id=context.trade_id,
        member_bioguide_id=context.member_bioguide_id,
        transaction_date=context.transaction_date,
        disclosure_date=context.disclosure_date,
        disclosure_lag_days=context.disclosure_lag_days,
        nearby_hearings=context.nearby_hearings,
        nearby_bills=context.nearby_bills,
        nearby_votes=context.nearby_votes,
        committee_sector_alignment=context.committee_sector_alignment,
        aligned_committees=context.aligned_committees,
        min_hearing_distance_days=context.min_hearing_distance_days,
        min_bill_distance_days=context.min_bill_distance_days,
        timing_suspicion_score=context.timing_suspicion_score,
    )
