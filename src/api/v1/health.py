"""Health check endpoints with detailed service status."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.postgres import get_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


async def _check_postgres(session: AsyncSession) -> dict:
    try:
        result = await session.execute(text("SELECT 1"))
        result.scalar()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


async def _check_redis() -> dict:
    try:
        import redis

        from src.config import settings

        r = redis.from_url(settings.redis_url, socket_connect_timeout=3)
        r.ping()
        r.close()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


async def _check_neo4j() -> dict:
    try:
        from src.db.neo4j import verify_connectivity

        ok = await verify_connectivity()
        return {"status": "ok"} if ok else {"status": "error", "detail": "connectivity check failed"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


async def _check_data_freshness(session: AsyncSession) -> dict:
    """Check that we have recent data (trade within 7 days)."""
    try:
        from src.models.trade import TradeDisclosure

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        result = await session.execute(
            select(func.count()).where(TradeDisclosure.created_at >= cutoff)
        )
        recent_count = result.scalar() or 0
        return {
            "status": "ok" if recent_count > 0 else "warning",
            "recent_trades_7d": recent_count,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.get("/health/detailed")
async def detailed_health(
    db: AsyncSession = Depends(get_session),
) -> dict:
    """Detailed health check — verifies all backing services."""
    postgres = await _check_postgres(db)
    redis_status = await _check_redis()
    neo4j = await _check_neo4j()
    data = await _check_data_freshness(db)

    checks = {
        "postgres": postgres,
        "redis": redis_status,
        "neo4j": neo4j,
        "data_freshness": data,
    }

    all_ok = all(c.get("status") == "ok" for c in checks.values())
    return {
        "status": "healthy" if all_ok else "degraded",
        "checks": checks,
    }


@router.get("/health/legal")
async def legal_disclaimer() -> dict:
    return {
        "disclaimer": (
            "This system is for informational and research purposes only. "
            "It does NOT constitute financial advice, investment recommendations, "
            "or solicitation to buy or sell securities. Past congressional trading "
            "patterns do not guarantee future results. Always consult a qualified "
            "financial advisor before making investment decisions."
        ),
    }
