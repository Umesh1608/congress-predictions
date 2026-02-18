from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.models.trade import TradeDisclosure
from src.schemas.trade import TradeResponse, TradeStatsResponse

router = APIRouter(prefix="/trades", tags=["trades"])


@router.get("", response_model=list[TradeResponse])
async def list_trades(
    member_name: str | None = Query(None),
    ticker: str | None = Query(None),
    chamber: str | None = Query(None),
    transaction_type: str | None = Query(None),
    filer_type: str | None = Query(None),
    date_from: date | None = Query(None),
    date_to: date | None = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List trades with filters."""
    query = select(TradeDisclosure)

    if member_name:
        query = query.where(TradeDisclosure.member_name.ilike(f"%{member_name}%"))
    if ticker:
        query = query.where(TradeDisclosure.ticker == ticker.upper())
    if chamber:
        query = query.where(TradeDisclosure.chamber == chamber.lower())
    if transaction_type:
        query = query.where(TradeDisclosure.transaction_type == transaction_type.lower())
    if filer_type:
        query = query.where(TradeDisclosure.filer_type == filer_type.lower())
    if date_from:
        query = query.where(TradeDisclosure.transaction_date >= date_from)
    if date_to:
        query = query.where(TradeDisclosure.transaction_date <= date_to)

    query = query.order_by(TradeDisclosure.transaction_date.desc()).limit(limit).offset(offset)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/stats", response_model=TradeStatsResponse)
async def trade_stats(db: AsyncSession = Depends(get_db)):
    """Get aggregate trade statistics."""
    total_trades = await db.scalar(select(func.count(TradeDisclosure.id)))

    total_members = await db.scalar(
        select(func.count(func.distinct(TradeDisclosure.member_name)))
    )

    # Most traded tickers
    ticker_counts = await db.execute(
        select(TradeDisclosure.ticker, func.count().label("count"))
        .where(TradeDisclosure.ticker.is_not(None))
        .group_by(TradeDisclosure.ticker)
        .order_by(func.count().desc())
        .limit(10)
    )
    most_traded = [{"ticker": r[0], "count": r[1]} for r in ticker_counts.all()]

    # Most active members
    member_counts = await db.execute(
        select(TradeDisclosure.member_name, func.count().label("count"))
        .group_by(TradeDisclosure.member_name)
        .order_by(func.count().desc())
        .limit(10)
    )
    most_active = [{"member": r[0], "count": r[1]} for r in member_counts.all()]

    # Recent trades
    week_ago = date.today() - timedelta(days=7)
    recent_count = await db.scalar(
        select(func.count(TradeDisclosure.id))
        .where(TradeDisclosure.disclosure_date >= week_ago)
    )

    return TradeStatsResponse(
        total_trades=total_trades or 0,
        total_members=total_members or 0,
        most_traded_tickers=most_traded,
        most_active_members=most_active,
        recent_trades_count_7d=recent_count or 0,
    )
