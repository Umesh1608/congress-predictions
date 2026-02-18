#!/usr/bin/env python3
"""Seed the database with historical trade data and market data.

Usage:
    python -m scripts.seed_initial_data [--skip-market-data]

This script:
1. Creates all database tables
2. Fetches historical trades from House and Senate Stock Watchers
3. Optionally fetches FMP data if API key is configured
4. Backfills market data for all traded tickers
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


async def collect_and_load(factory, collector_cls):
    """Run a collector and load results into the database."""
    collector = collector_cls()
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

        # Step 2: Collect trades from all sources
        total = 0
        for cls in [HouseWatcherCollector, SenateWatcherCollector]:
            count = await collect_and_load(factory, cls)
            total += count

        # Step 3: Try FMP if configured
        if settings.fmp_api_key:
            for cls in [FMPHouseCollector, FMPSenateCollector]:
                count = await collect_and_load(factory, cls)
                total += count
        else:
            logger.info("FMP API key not configured, skipping FMP sources")

        logger.info("Total trade records processed: %d", total)

        # Step 4: Backfill market data
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
