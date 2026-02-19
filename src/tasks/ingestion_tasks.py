"""Celery tasks for data ingestion."""

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


async def _run_collector(collector_cls):
    """Run a collector and upsert results."""
    from src.ingestion.loader import upsert_trades

    factory, engine = _get_async_session()
    collector = collector_cls()
    try:
        records = await collector.run()
        async with factory() as session:
            count = await upsert_trades(session, records)
            logger.info("[%s] Inserted %d new records", collector.source_name, count)
    finally:
        await collector.close()
        await engine.dispose()


@celery_app.task(name="src.tasks.ingestion_tasks.collect_house_trades")
def collect_house_trades():
    """Collect House trades: try S3 watcher first, fall back to House Clerk scraper."""
    try:
        from src.ingestion.trades.house_watcher import HouseWatcherCollector
        asyncio.run(_run_collector(HouseWatcherCollector))
    except Exception as e:
        logger.warning("House Watcher S3 failed (%s), falling back to House Clerk scraper", e)
        from src.ingestion.trades.house_clerk import HouseClerkCollector
        asyncio.run(_run_collector(HouseClerkCollector))


@celery_app.task(name="src.tasks.ingestion_tasks.collect_senate_trades")
def collect_senate_trades():
    """Collect Senate trades: try S3 watcher first, fall back to GitHub CSV."""
    try:
        from src.ingestion.trades.senate_watcher import SenateWatcherCollector
        asyncio.run(_run_collector(SenateWatcherCollector))
    except Exception as e:
        logger.warning("Senate Watcher S3 failed (%s), falling back to GitHub CSV", e)
        from src.ingestion.trades.github_senate import GitHubSenateCollector
        asyncio.run(_run_collector(GitHubSenateCollector))


@celery_app.task(name="src.tasks.ingestion_tasks.collect_house_clerk_trades")
def collect_house_clerk_trades():
    from src.ingestion.trades.house_clerk import HouseClerkCollector
    asyncio.run(_run_collector(HouseClerkCollector))


@celery_app.task(name="src.tasks.ingestion_tasks.collect_fmp_house_trades")
def collect_fmp_house_trades():
    from src.ingestion.trades.fmp_client import FMPHouseCollector
    asyncio.run(_run_collector(FMPHouseCollector))


@celery_app.task(name="src.tasks.ingestion_tasks.collect_fmp_senate_trades")
def collect_fmp_senate_trades():
    from src.ingestion.trades.fmp_client import FMPSenateCollector
    asyncio.run(_run_collector(FMPSenateCollector))


@celery_app.task(name="src.tasks.ingestion_tasks.collect_market_data")
def collect_market_data():
    """Fetch market data for all tickers found in trade disclosures."""
    from src.ingestion.market.yahoo_finance import fetch_stock_history
    from src.ingestion.loader import get_unique_tickers, upsert_stock_daily
    from datetime import date, timedelta

    async def _run():
        factory, engine = _get_async_session()
        try:
            async with factory() as session:
                tickers = await get_unique_tickers(session)

            logger.info("Fetching market data for %d tickers", len(tickers))
            start = date.today() - timedelta(days=7)

            for ticker in tickers:
                records = fetch_stock_history(ticker, start_date=start)
                if records:
                    async with factory() as session:
                        await upsert_stock_daily(session, records)
        finally:
            await engine.dispose()

    asyncio.run(_run())
