#!/usr/bin/env python3
"""Seed the database with historical trade data and market data.

Usage:
    python -m scripts.seed_initial_data [--skip-market-data]

This script:
1. Creates all database tables
2. Fetches historical trades from multiple sources (with fallbacks)
3. Optionally fetches FMP data if API key is configured
4. Backfills market data for all traded tickers

Trade data sources (tried in order):
- House Stock Watcher (S3 JSON) — free, may be down
- Senate Stock Watcher (S3 JSON) — free, may be down
- GitHub Senate CSV (jeremiak dataset) — free fallback, 2012-2024
- House Clerk PTR scraper — free fallback, scrapes clerk.house.gov
- FMP House/Senate — paid, requires API key
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, timedelta

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.config import settings
from src.db.postgres import Base
from src.ingestion.trades.house_watcher import HouseWatcherCollector
from src.ingestion.trades.senate_watcher import SenateWatcherCollector
from src.ingestion.trades.github_senate import GitHubSenateCollector
from src.ingestion.trades.house_clerk import HouseClerkCollector
from src.ingestion.trades.fmp_client import FMPHouseCollector, FMPSenateCollector
from src.ingestion.loader import upsert_trades, upsert_stock_daily, get_unique_tickers
from src.ingestion.market.yahoo_finance import fetch_stock_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def create_tables(engine):
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified")


async def collect_and_load(factory, collector_cls, **kwargs):
    """Run a collector and load results into the database."""
    collector = collector_cls(**kwargs) if kwargs else collector_cls()
    try:
        records = await collector.run()
        if records:
            async with factory() as session:
                count = await upsert_trades(session, records)
                logger.info("[%s] Loaded %d new trades", collector.source_name, count)
        return len(records)
    except Exception as e:
        logger.warning("[%s] Collection failed (skipping): %s", collector.source_name, e)
        return 0
    finally:
        await collector.close()


async def backfill_market_data(factory):
    """Fetch historical market data for all tickers in trade disclosures."""
    async with factory() as session:
        tickers = await get_unique_tickers(session)

    logger.info("Backfilling market data for %d tickers", len(tickers))
    start_date = date(2016, 1, 1)

    loaded = 0
    for i, ticker in enumerate(tickers, 1):
        if i % 50 == 0:
            logger.info("Market data progress: %d/%d tickers", i, len(tickers))

        records = fetch_stock_history(ticker, start_date=start_date)
        if records:
            async with factory() as session:
                await upsert_stock_daily(session, records)
            loaded += 1

        # Respect yfinance rate limits
        await asyncio.sleep(0.5)

    logger.info("Market data backfilled for %d tickers", loaded)


async def main(skip_market_data: bool = False):
    engine = create_async_engine(settings.database_url, pool_size=10)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    try:
        # Step 1: Create tables
        await create_tables(engine)

        # Step 2: Collect trades from primary sources
        total = 0
        senate_count = 0
        house_count = 0

        # Try primary sources first (S3 JSON endpoints)
        logger.info("Trying primary sources (Stock Watcher S3)...")
        house_count = await collect_and_load(factory, HouseWatcherCollector)
        total += house_count
        senate_count = await collect_and_load(factory, SenateWatcherCollector)
        total += senate_count

        # Step 3: Fallback to alternative free sources if primary failed
        if senate_count == 0:
            logger.info("Senate Stock Watcher unavailable, trying GitHub CSV fallback...")
            count = await collect_and_load(factory, GitHubSenateCollector)
            total += count
            if count > 0:
                logger.info("GitHub Senate CSV fallback loaded %d trades", count)

        if house_count == 0:
            logger.info("House Stock Watcher unavailable, trying House Clerk scraper...")
            count = await collect_and_load(factory, HouseClerkCollector)
            total += count
            if count > 0:
                logger.info("House Clerk scraper loaded %d trades", count)

        # Step 4: Try FMP if configured (supplements other sources)
        if settings.fmp_api_key:
            for cls in [FMPHouseCollector, FMPSenateCollector]:
                count = await collect_and_load(factory, cls)
                total += count
        else:
            logger.info("FMP API key not configured, skipping FMP sources")

        logger.info("Total trade records processed: %d", total)

        # Step 5: Backfill market data
        if not skip_market_data:
            await backfill_market_data(factory)
        else:
            logger.info("Skipping market data backfill (--skip-market-data)")

    finally:
        await engine.dispose()

    logger.info("Seed complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed initial data")
    parser.add_argument("--skip-market-data", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(skip_market_data=args.skip_market_data))
