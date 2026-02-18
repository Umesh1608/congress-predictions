from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.deps import get_db
from src.models.member import CongressMember
from src.models.trade import TradeDisclosure
from src.schemas.member import MemberDetailResponse, MemberResponse
from src.schemas.trade import TradeResponse

router = APIRouter(prefix="/members", tags=["members"])


@router.get("", response_model=list[MemberResponse])
async def list_members(
    chamber: str | None = Query(None, description="Filter by house or senate"),
    party: str | None = Query(None),
    state: str | None = Query(None),
    in_office: bool | None = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List congress members with optional filters."""
    query = select(CongressMember)

    if chamber:
        query = query.where(CongressMember.chamber == chamber.lower())
    if party:
        query = query.where(CongressMember.party == party)
    if state:
        query = query.where(CongressMember.state == state.upper())
    if in_office is not None:
        query = query.where(CongressMember.in_office == in_office)

    query = query.order_by(CongressMember.full_name).limit(limit).offset(offset)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/{bioguide_id}", response_model=MemberDetailResponse)
async def get_member(
    bioguide_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get detailed info for a specific member."""
    result = await db.execute(
        select(CongressMember).where(CongressMember.bioguide_id == bioguide_id)
    )
    member = result.scalar_one_or_none()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    return member


@router.get("/{bioguide_id}/trades", response_model=list[TradeResponse])
async def get_member_trades(
    bioguide_id: str,
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Get all trades for a specific member."""
    result = await db.execute(
        select(TradeDisclosure)
        .where(TradeDisclosure.member_bioguide_id == bioguide_id)
        .order_by(TradeDisclosure.transaction_date.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()
