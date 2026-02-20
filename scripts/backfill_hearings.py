"""Backfill committee hearings from Congress.gov API for multiple congresses.

Collects hearing data with proper committee_code and hearing_date from the
Congress.gov API v3. The hearing detail endpoint returns:
- committees: [{name, systemCode, url}]  (plural list)
- dates: [{date: "YYYY-MM-DD"}]  (list of date objects)

Run: python -m scripts.backfill_hearings [--congresses 114 115 116 117 118 119]
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.config import settings
from src.ingestion.legislation.congress_gov import CongressHearingCollector
from src.ingestion.loader import upsert_hearings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def backfill_hearings(congresses: list[int]) -> None:
    engine = create_async_engine(settings.database_url, pool_size=3)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    total_inserted = 0

    for congress in congresses:
        logger.info("Collecting hearings for Congress %d...", congress)
        start = time.time()

        collector = CongressHearingCollector(congress=congress)
        try:
            records = await collector.run()
        except Exception as e:
            logger.error("Error collecting Congress %d hearings: %s", congress, e)
            continue

        if not records:
            logger.info("  Congress %d: 0 hearings collected", congress)
            continue

        # Filter out records without committee_code (useless for matching)
        valid = [r for r in records if r.get("committee_code")]
        logger.info(
            "  Congress %d: %d hearings collected, %d with committee codes",
            congress, len(records), len(valid),
        )

        if valid:
            async with factory() as session:
                inserted = await upsert_hearings(session, valid)
                total_inserted += inserted
                logger.info("  Congress %d: inserted %d hearings (%.0fs)", congress, inserted, time.time() - start)

    # Summary
    async with factory() as session:
        r = await session.execute(text("SELECT COUNT(*) FROM committee_hearing"))
        total_db = r.scalar()
        r = await session.execute(text(
            "SELECT COUNT(*) FROM committee_hearing WHERE committee_code IS NOT NULL AND hearing_date IS NOT NULL"
        ))
        complete = r.scalar()

    logger.info("Backfill complete: %d inserted, %d total in DB, %d with full data", total_inserted, total_db, complete)
    await engine.dispose()


async def main():
    args = sys.argv[1:]
    if "--congresses" in args:
        idx = args.index("--congresses")
        congresses = [int(x) for x in args[idx + 1:]]
    else:
        # Default: congresses 114-119 (2015-2026, covering all trades)
        congresses = [114, 115, 116, 117, 118, 119]

    logger.info("Target congresses: %s", congresses)
    await backfill_hearings(congresses)


if __name__ == "__main__":
    asyncio.run(main())
