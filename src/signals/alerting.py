"""Alert dispatch for matching signals against user alert configurations."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.signal import AlertConfig, Signal

logger = logging.getLogger(__name__)

# Track dispatch counts per config for rate limiting
_dispatch_counts: dict[int, list[datetime]] = {}
MAX_ALERTS_PER_HOUR = 10


async def dispatch_alerts(session: AsyncSession) -> int:
    """Check new signals against alert configs and dispatch notifications.

    Returns count of alerts dispatched.
    """
    # Get active alert configs
    config_result = await session.execute(
        select(AlertConfig).where(AlertConfig.is_active.is_(True))
    )
    configs = config_result.scalars().all()

    if not configs:
        return 0

    # Get recent undispatched signals (created in last 30 min)
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
    signal_result = await session.execute(
        select(Signal).where(
            and_(
                Signal.is_active.is_(True),
                Signal.created_at >= cutoff,
            )
        )
    )
    signals = signal_result.scalars().all()

    if not signals:
        return 0

    dispatched = 0
    for config in configs:
        for signal in signals:
            if _matches_config(signal, config) and _within_rate_limit(config.id):
                await _send_alert(config, signal)
                dispatched += 1

    logger.info("Dispatched %d alerts for %d signals", dispatched, len(signals))
    return dispatched


def _matches_config(signal: Signal, config: AlertConfig) -> bool:
    """Check if a signal matches an alert configuration."""
    # Check signal type
    allowed_types = config.signal_types or []
    if allowed_types and signal.signal_type not in allowed_types:
        return False

    # Check minimum strength
    if signal.strength < float(config.min_strength or 0):
        return False

    # Check ticker filter
    ticker_filter = config.tickers or []
    if ticker_filter and signal.ticker not in ticker_filter:
        return False

    # Check member filter
    member_filter = config.members or []
    if member_filter and signal.member_bioguide_id not in member_filter:
        return False

    return True


def _within_rate_limit(config_id: int) -> bool:
    """Check if we're within the rate limit for this config."""
    now = datetime.now(timezone.utc)
    hour_ago = now - timedelta(hours=1)

    if config_id not in _dispatch_counts:
        _dispatch_counts[config_id] = []

    # Clean old entries
    _dispatch_counts[config_id] = [
        t for t in _dispatch_counts[config_id] if t > hour_ago
    ]

    if len(_dispatch_counts[config_id]) >= MAX_ALERTS_PER_HOUR:
        return False

    _dispatch_counts[config_id].append(now)
    return True


async def _send_alert(config: AlertConfig, signal: Signal) -> None:
    """Send alert via configured channels."""
    alert_payload = {
        "signal_type": signal.signal_type,
        "ticker": signal.ticker,
        "direction": signal.direction,
        "strength": float(signal.strength),
        "confidence": float(signal.confidence),
        "evidence": signal.evidence,
        "member_bioguide_id": signal.member_bioguide_id,
        "created_at": signal.created_at.isoformat() if signal.created_at else None,
    }

    # Always log
    logger.info(
        "ALERT [%s]: %s %s (strength=%.2f, confidence=%.2f)",
        config.name,
        signal.signal_type,
        signal.ticker or "N/A",
        float(signal.strength),
        float(signal.confidence),
    )

    # Webhook dispatch
    if config.webhook_url:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    config.webhook_url,
                    json=alert_payload,
                )
                resp.raise_for_status()
        except Exception:
            logger.exception(
                "Failed to send webhook alert to %s", config.webhook_url
            )
