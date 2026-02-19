"""Backfill House Clerk PTR trade data year-by-year.

Processes PDFs with semaphore-bounded concurrency (chunks of 200) and
inserts to DB after each chunk to avoid memory issues.

Usage:
    python -m scripts.backfill_house_clerk                    # 2020-2026
    python -m scripts.backfill_house_clerk --years 2023 2024  # specific years
    python -m scripts.backfill_house_clerk --start 2015       # 2015 to current
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.config import settings
from src.ingestion.base import RateLimiter
from src.ingestion.loader import get_existing_filing_urls, upsert_trades
from src.ingestion.trades.house_clerk import HouseClerkCollector, _MAX_CONCURRENT_DOWNLOADS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("pypdf").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

CHUNK_SIZE = 200


async def backfill_year(year: int, factory, existing_urls: set[str]) -> int:
    """Backfill a single year in streaming chunks with concurrent downloads."""
    collector = HouseClerkCollector(years=[year], skip_urls=existing_urls)
    # Use a faster rate limit for backfill (10 req/sec instead of 5)
    collector.rate_limiter = RateLimiter(max_calls=10, period_seconds=1.0)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_DOWNLOADS)
    total_inserted = 0
    total_trades = 0

    try:
        # Step 1: Get filing list
        filings = await collector._search_filings(year)
        ptr_filings = [f for f in filings if "PTR" in f.get("filing_type", "")]

        # Filter out already-processed
        before = len(ptr_filings)
        ptr_filings = [
            f for f in ptr_filings if f.get("pdf_url", "") not in existing_urls
        ]
        logger.info(
            "[%d] %d PTR filings (%d new, %d already processed)",
            year, before, len(ptr_filings), before - len(ptr_filings),
        )

        if not ptr_filings:
            return 0

        # Step 2: Process in chunks with semaphore-bounded concurrency
        for chunk_start in range(0, len(ptr_filings), CHUNK_SIZE):
            chunk = ptr_filings[chunk_start:chunk_start + CHUNK_SIZE]
            tasks = []
            for filing in chunk:
                pdf_url = filing.get("pdf_url", "")
                if pdf_url:
                    tasks.append(
                        collector._safe_download(pdf_url, filing.get("name", ""), semaphore)
                    )

            results = await asyncio.gather(*tasks)

            # Collect, transform, and insert immediately
            raw_trades = []
            for trades in results:
                raw_trades.extend(trades)

            if raw_trades:
                transformed = []
                for raw in raw_trades:
                    result = collector.transform(raw)
                    if result is not None:
                        transformed.append(result)

                if transformed:
                    async with factory() as session:
                        count = await upsert_trades(session, transformed)
                        total_inserted += count
                        total_trades += len(transformed)

            processed = min(chunk_start + CHUNK_SIZE, len(ptr_filings))
            logger.info(
                "[%d] %d/%d filings | %d trades extracted | %d inserted",
                year, processed, len(ptr_filings), total_trades, total_inserted,
            )

    finally:
        await collector.close()

    logger.info(
        "[%d] Done: %d trades extracted, %d inserted",
        year, total_trades, total_inserted,
    )
    return total_inserted


async def main(years: list[int]) -> None:
    engine = create_async_engine(settings.database_url, pool_size=5)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with factory() as session:
            existing = await get_existing_filing_urls(session, "house_clerk")
        logger.info("Found %d existing filing URLs to skip", len(existing))

        grand_total = 0
        for year in years:
            logger.info("=" * 60)
            count = await backfill_year(year, factory, existing)
            grand_total += count

            # Refresh existing URLs for next year
            async with factory() as session:
                existing = await get_existing_filing_urls(session, "house_clerk")

        logger.info("=" * 60)
        logger.info(
            "Backfill complete: %d total new records across %d years",
            grand_total, len(years),
        )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill House Clerk PTR data")
    parser.add_argument(
        "--years", nargs="+", type=int,
        help="Specific years to backfill (e.g., --years 2023 2024)",
    )
    parser.add_argument(
        "--start", type=int, default=2020,
        help="Start year for range (default: 2020, goes to current year)",
    )
    args = parser.parse_args()

    if args.years:
        years = sorted(args.years, reverse=True)
    else:
        current = date.today().year
        years = list(range(current, args.start - 1, -1))

    logger.info("Will backfill years: %s", years)
    asyncio.run(main(years))
