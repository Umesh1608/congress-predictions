"""Celery tasks for network data collection, entity resolution, and graph sync."""

from __future__ import annotations

import asyncio
import logging

from src.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Bridge async code into Celery's sync worker."""
    return asyncio.run(coro)


@celery_app.task(name="src.tasks.network_tasks.collect_lobbying_filings")
def collect_lobbying_filings(filing_year: int | None = None):
    """Collect lobbying disclosure filings from Senate LDA API."""

    async def _collect():
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

        from src.config import settings
        from src.ingestion.network.lobbying import LobbyingFilingCollector
        from src.ingestion.loader import upsert_lobbying_filings

        engine = create_async_engine(settings.database_url)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)

        collector = LobbyingFilingCollector(filing_year=filing_year)
        try:
            records = await collector.run()
            async with session_factory() as session:
                count = await upsert_lobbying_filings(session, records)
                logger.info("Loaded %d lobbying filings", count)
        finally:
            await collector.close()
            await engine.dispose()

    _run_async(_collect())


@celery_app.task(name="src.tasks.network_tasks.collect_campaign_committees")
def collect_campaign_committees():
    """Collect FEC campaign committee data."""

    async def _collect():
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

        from src.config import settings
        from src.ingestion.network.campaign_finance import FECCommitteeCollector
        from src.ingestion.loader import upsert_campaign_committees

        engine = create_async_engine(settings.database_url)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)

        api_key = settings.fec_api_key
        collector = FECCommitteeCollector(api_key=api_key)
        try:
            records = await collector.run()
            async with session_factory() as session:
                count = await upsert_campaign_committees(session, records)
                logger.info("Loaded %d campaign committees", count)
        finally:
            await collector.close()
            await engine.dispose()

    _run_async(_collect())


@celery_app.task(name="src.tasks.network_tasks.resolve_entities")
def resolve_entities():
    """Run entity resolution to match lobbying clients and employers to tickers."""

    async def _resolve():
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

        from src.config import settings
        from src.processing.normalizer import resolve_lobbying_clients, resolve_campaign_employers

        engine = create_async_engine(settings.database_url)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)

        async with session_factory() as session:
            lobby_resolved = await resolve_lobbying_clients(session)
            campaign_resolved = await resolve_campaign_employers(session)
            logger.info(
                "Entity resolution: %d lobbying clients, %d campaign employers resolved",
                lobby_resolved, campaign_resolved,
            )

        await engine.dispose()

    _run_async(_resolve())


@celery_app.task(name="src.tasks.network_tasks.sync_graph")
def sync_graph():
    """Sync all PostgreSQL data to Neo4j graph."""

    async def _sync():
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

        from src.config import settings
        from src.db.neo4j import get_neo4j_session, verify_connectivity
        from src.graph.sync import run_full_sync

        if not await verify_connectivity():
            logger.warning("Neo4j not available, skipping graph sync")
            return

        engine = create_async_engine(settings.database_url)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)

        async with session_factory() as pg_session:
            async with get_neo4j_session() as neo4j_session:
                summary = await run_full_sync(pg_session, neo4j_session)
                logger.info("Graph sync complete: %s", summary)

        await engine.dispose()

    _run_async(_sync())
