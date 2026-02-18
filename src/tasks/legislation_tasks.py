"""Celery tasks for legislative data ingestion (Phase 2)."""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.config import settings
from src.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def _get_async_session():
    engine = create_async_engine(settings.database_url, pool_size=5)
    return async_sessionmaker(engine, expire_on_commit=False), engine


@celery_app.task(name="src.tasks.legislation_tasks.collect_members")
def collect_members():
    """Collect current congress member data from Congress.gov API."""
    from src.ingestion.legislation.congress_gov import CongressMemberCollector
    from src.ingestion.loader import upsert_members

    async def _run():
        factory, engine = _get_async_session()
        collector = CongressMemberCollector(current_only=True)
        try:
            records = await collector.run()
            async with factory() as session:
                count = await upsert_members(session, records)
                logger.info("Upserted %d members from Congress.gov", count)
        finally:
            await collector.close()
            await engine.dispose()

    asyncio.run(_run())


@celery_app.task(name="src.tasks.legislation_tasks.collect_bills")
def collect_bills():
    """Collect bills for the current congress session."""
    from src.ingestion.legislation.congress_gov import CongressBillCollector, _current_congress
    from src.ingestion.loader import upsert_bills

    async def _run():
        factory, engine = _get_async_session()
        congress = _current_congress()

        try:
            # Collect both House and Senate bills
            for bill_type in ["hr", "s", "hjres", "sjres"]:
                collector = CongressBillCollector(congress=congress, bill_type=bill_type)
                try:
                    records = await collector.run()
                    async with factory() as session:
                        count = await upsert_bills(session, records)
                        logger.info(
                            "Upserted %d %s bills for congress %d",
                            count, bill_type, congress,
                        )
                finally:
                    await collector.close()
        finally:
            await engine.dispose()

    asyncio.run(_run())


@celery_app.task(name="src.tasks.legislation_tasks.collect_committees")
def collect_committees():
    """Collect committee data from Congress.gov API."""
    from src.ingestion.legislation.congress_gov import CongressCommitteeCollector
    from src.ingestion.loader import upsert_committees

    async def _run():
        factory, engine = _get_async_session()
        collector = CongressCommitteeCollector()
        try:
            records = await collector.run()
            async with factory() as session:
                count = await upsert_committees(session, records)
                logger.info("Upserted %d committees", count)
        finally:
            await collector.close()
            await engine.dispose()

    asyncio.run(_run())


@celery_app.task(name="src.tasks.legislation_tasks.collect_hearings")
def collect_hearings():
    """Collect hearing data from Congress.gov API."""
    from src.ingestion.legislation.congress_gov import CongressHearingCollector
    from src.ingestion.loader import upsert_hearings

    async def _run():
        factory, engine = _get_async_session()
        collector = CongressHearingCollector()
        try:
            records = await collector.run()
            async with factory() as session:
                count = await upsert_hearings(session, records)
                logger.info("Upserted %d hearings", count)
        finally:
            await collector.close()
            await engine.dispose()

    asyncio.run(_run())


@celery_app.task(name="src.tasks.legislation_tasks.collect_voteview_scores")
def collect_voteview_scores():
    """Collect DW-NOMINATE ideology scores from Voteview."""
    from src.ingestion.legislation.voteview import VoteviewCollector
    from src.ingestion.loader import update_member_ideology

    async def _run():
        factory, engine = _get_async_session()
        collector = VoteviewCollector()
        try:
            records = await collector.run()
            async with factory() as session:
                count = await update_member_ideology(session, records)
                logger.info("Updated ideology scores for %d members", count)
        finally:
            await collector.close()
            await engine.dispose()

    asyncio.run(_run())
