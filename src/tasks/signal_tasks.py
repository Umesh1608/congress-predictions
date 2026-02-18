"""Celery tasks for signal generation and alert dispatch."""

from __future__ import annotations

import asyncio
import logging

from src.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="src.tasks.signal_tasks.generate_signals")
def generate_signals():
    """Run all signal generators."""
    asyncio.run(_generate())


async def _generate():
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from src.config import settings
    from src.signals.generator import SignalGenerator

    engine = create_async_engine(settings.database_url, pool_size=5)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as session:
            generator = SignalGenerator()
            signals = await generator.generate_all(session)
            logger.info("Signal generation complete: %d signals", len(signals))
    finally:
        await engine.dispose()


@celery_app.task(name="src.tasks.signal_tasks.expire_signals")
def expire_signals():
    """Deactivate expired signals."""
    asyncio.run(_expire())


async def _expire():
    from datetime import datetime, timezone

    from sqlalchemy import and_, update
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from src.config import settings
    from src.models.signal import Signal

    engine = create_async_engine(settings.database_url, pool_size=5)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as session:
            result = await session.execute(
                update(Signal)
                .where(
                    and_(
                        Signal.is_active.is_(True),
                        Signal.expires_at <= datetime.now(timezone.utc),
                    )
                )
                .values(is_active=False)
            )
            await session.commit()
            logger.info("Expired %d signals", result.rowcount)
    finally:
        await engine.dispose()


@celery_app.task(name="src.tasks.signal_tasks.dispatch_alerts")
def dispatch_alert_task():
    """Check new signals against alert configs and dispatch."""
    asyncio.run(_dispatch())


async def _dispatch():
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from src.config import settings
    from src.signals.alerting import dispatch_alerts

    engine = create_async_engine(settings.database_url, pool_size=5)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as session:
            count = await dispatch_alerts(session)
            logger.info("Dispatched %d alerts", count)
    finally:
        await engine.dispose()
