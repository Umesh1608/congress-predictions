"""Signals and alerts API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.models.signal import AlertConfig, Signal
from src.schemas.signals import (
    AlertConfigCreate,
    AlertConfigResponse,
    SignalDetailResponse,
    SignalResponse,
    SignalStatsResponse,
)

router = APIRouter(tags=["signals"])


@router.get("/signals", response_model=list[SignalResponse])
async def list_signals(
    signal_type: str | None = None,
    ticker: str | None = None,
    member_bioguide_id: str | None = None,
    min_strength: float | None = None,
    active_only: bool = True,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    query = select(Signal).order_by(Signal.strength.desc(), Signal.created_at.desc())

    if active_only:
        query = query.where(Signal.is_active.is_(True))
    if signal_type:
        query = query.where(Signal.signal_type == signal_type)
    if ticker:
        query = query.where(Signal.ticker == ticker)
    if member_bioguide_id:
        query = query.where(Signal.member_bioguide_id == member_bioguide_id)
    if min_strength is not None:
        query = query.where(Signal.strength >= min_strength)

    query = query.offset(offset).limit(limit)
    result = await db.execute(query)

    return [
        {
            "id": s.id,
            "signal_type": s.signal_type,
            "member_bioguide_id": s.member_bioguide_id,
            "ticker": s.ticker,
            "direction": s.direction,
            "strength": float(s.strength),
            "confidence": float(s.confidence),
            "is_active": s.is_active,
            "expires_at": s.expires_at,
            "created_at": s.created_at,
        }
        for s in result.scalars().all()
    ]


@router.get("/signals/stats", response_model=SignalStatsResponse)
async def signal_stats(db: AsyncSession = Depends(get_db)) -> dict:
    # Active signals count
    total_result = await db.execute(
        select(func.count(Signal.id)).where(Signal.is_active.is_(True))
    )
    total_active = total_result.scalar() or 0

    # By type
    type_result = await db.execute(
        select(Signal.signal_type, func.count(Signal.id))
        .where(Signal.is_active.is_(True))
        .group_by(Signal.signal_type)
    )
    by_type = {row[0]: row[1] for row in type_result.all()}

    # Avg strength
    avg_result = await db.execute(
        select(func.avg(Signal.strength)).where(Signal.is_active.is_(True))
    )
    avg_strength = float(avg_result.scalar() or 0)

    # By direction
    dir_result = await db.execute(
        select(Signal.direction, func.count(Signal.id))
        .where(Signal.is_active.is_(True))
        .group_by(Signal.direction)
    )
    by_direction = {row[0]: row[1] for row in dir_result.all()}

    return {
        "total_active": total_active,
        "by_type": by_type,
        "avg_strength": avg_strength,
        "by_direction": by_direction,
    }


@router.get("/signals/{signal_id}", response_model=SignalDetailResponse)
async def get_signal(
    signal_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    result = await db.execute(select(Signal).where(Signal.id == signal_id))
    signal = result.scalar_one_or_none()
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")

    return {
        "id": signal.id,
        "signal_type": signal.signal_type,
        "member_bioguide_id": signal.member_bioguide_id,
        "ticker": signal.ticker,
        "direction": signal.direction,
        "strength": float(signal.strength),
        "confidence": float(signal.confidence),
        "evidence": signal.evidence or {},
        "is_active": signal.is_active,
        "expires_at": signal.expires_at,
        "created_at": signal.created_at,
    }


@router.post("/alerts/configs", response_model=AlertConfigResponse)
async def create_alert_config(
    config: AlertConfigCreate,
    db: AsyncSession = Depends(get_db),
) -> dict:
    alert_config = AlertConfig(
        name=config.name,
        signal_types=config.signal_types,
        min_strength=config.min_strength,
        tickers=config.tickers,
        members=config.members,
        webhook_url=config.webhook_url,
        is_active=True,
    )
    db.add(alert_config)
    await db.commit()
    await db.refresh(alert_config)

    return {
        "id": alert_config.id,
        "name": alert_config.name,
        "signal_types": alert_config.signal_types,
        "min_strength": float(alert_config.min_strength),
        "tickers": alert_config.tickers,
        "members": alert_config.members,
        "webhook_url": alert_config.webhook_url,
        "is_active": alert_config.is_active,
        "created_at": alert_config.created_at,
    }


@router.get("/alerts/configs", response_model=list[AlertConfigResponse])
async def list_alert_configs(
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    result = await db.execute(
        select(AlertConfig).order_by(AlertConfig.created_at.desc())
    )

    return [
        {
            "id": c.id,
            "name": c.name,
            "signal_types": c.signal_types,
            "min_strength": float(c.min_strength),
            "tickers": c.tickers,
            "members": c.members,
            "webhook_url": c.webhook_url,
            "is_active": c.is_active,
            "created_at": c.created_at,
        }
        for c in result.scalars().all()
    ]
