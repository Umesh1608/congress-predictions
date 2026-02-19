"""Backfill historical bills from Congress.gov API.

Fetches bills for congresses 114-118 (2015-2024) to provide
legislative context for historical trades.

Run: python -m scripts.backfill_bills
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.config import settings
from src.ingestion.legislation.congress_gov import CongressBillCollector
from src.ingestion.loader import upsert_bills

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Congresses covering trade data range (2016-2024)
CONGRESSES = [114, 115, 116, 117, 118]
BILL_TYPES = ["hr", "s", "hjres", "sjres"]


async def main():
    engine = create_async_engine(settings.database_url, pool_size=3)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    total_bills = 0

    for congress in CONGRESSES:
        congress_total = 0
        for bill_type in BILL_TYPES:
            logger.info("Collecting %s bills for Congress %d...", bill_type.upper(), congress)
            collector = CongressBillCollector(congress=congress, bill_type=bill_type)
            try:
                records = await collector.run()
                if records:
                    async with factory() as session:
                        count = await upsert_bills(session, records)
                        congress_total += count
                        logger.info(
                            "  Congress %d %s: %d bills upserted",
                            congress, bill_type.upper(), count,
                        )
                else:
                    logger.info("  Congress %d %s: no records", congress, bill_type.upper())
            except Exception:
                logger.exception("  Failed: Congress %d %s", congress, bill_type.upper())
            finally:
                await collector.close()

        total_bills += congress_total
        logger.info("Congress %d complete: %d bills", congress, congress_total)

    logger.info("Backfill complete: %d total bills across congresses %s", total_bills, CONGRESSES)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
