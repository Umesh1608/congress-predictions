"""Backfill campaign finance data: committees + contributions + entity resolution.

Run: python -m scripts.backfill_campaign_finance [--cycle YEAR]
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.config import settings
from src.ingestion.network.campaign_finance import (
    FECCommitteeCollector,
    FECContributionCollector,
)
from src.ingestion.loader import upsert_campaign_committees, upsert_campaign_contributions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def main():
    api_key = settings.fec_api_key
    if not api_key:
        logger.error("No FEC_API_KEY configured")
        return

    # Parse cycle year from args
    cycle = None
    for a in sys.argv[1:]:
        if a.startswith("--cycle="):
            cycle = int(a.split("=")[1])

    engine = create_async_engine(settings.database_url, pool_size=5)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    t0 = time.time()

    # Step 1: Collect committees for multiple cycles
    cycles = [cycle] if cycle else [2020, 2022, 2024]
    total_committees = 0
    for cy in cycles:
        logger.info("Collecting FEC committees for cycle %d...", cy)
        collector = FECCommitteeCollector(api_key=api_key, cycle=cy)
        try:
            records = await collector.run()
            if records:
                async with factory() as session:
                    count = await upsert_campaign_committees(session, records)
                    total_committees += count
                    logger.info("  Cycle %d: upserted %d committees (total: %d)",
                                cy, count, total_committees)
        except Exception as e:
            logger.error("  Error collecting committees for %d: %s", cy, e)
        finally:
            await collector.close()

    # Step 2: Get all committee IDs from DB
    async with factory() as session:
        r = await session.execute(text(
            "SELECT fec_committee_id FROM campaign_committee ORDER BY id"
        ))
        committee_ids = [row[0] for row in r.fetchall()]

    logger.info("Total committees in DB: %d", len(committee_ids))

    # Step 3: Link committees to members via candidate FEC IDs
    # FEC candidate IDs start with H (House) or S (Senate) + state + district + year
    # Bioguide mapping is tricky — do it after collection
    async with factory() as session:
        # Try to link via name matching
        r = await session.execute(text("""
            UPDATE campaign_committee cc
            SET member_bioguide_id = cm.bioguide_id
            FROM congress_member cm
            WHERE cc.member_bioguide_id IS NULL
            AND LOWER(cc.name) LIKE '%' || LOWER(cm.last_name) || '%'
            AND LOWER(cc.name) LIKE '%' || LOWER(cm.first_name) || '%'
            AND cm.first_name IS NOT NULL AND cm.last_name IS NOT NULL
        """))
        await session.commit()
        logger.info("Linked committees to members via name matching")

    # Step 4: Collect contributions for all committees (in batches)
    BATCH = 50
    total_contributions = 0
    errors = 0

    for cy in cycles:
        logger.info("Collecting contributions for cycle %d...", cy)
        for i in range(0, len(committee_ids), BATCH):
            batch_ids = committee_ids[i:i + BATCH]
            collector = FECContributionCollector(
                api_key=api_key,
                committee_ids=batch_ids,
                cycle=cy,
            )
            try:
                records = await collector.run()
                if records:
                    async with factory() as session:
                        count = await upsert_campaign_contributions(session, records)
                        total_contributions += count
                        if (i + BATCH) % 200 == 0 or (i + BATCH) >= len(committee_ids):
                            elapsed = time.time() - t0
                            logger.info(
                                "  Cycle %d progress: %d/%d committees, %d contributions (%.0fs elapsed)",
                                cy, min(i + BATCH, len(committee_ids)),
                                len(committee_ids), total_contributions, elapsed,
                            )
            except Exception as e:
                errors += 1
                logger.error("  Error for batch starting at %d: %s", i, e)
            finally:
                await collector.close()

    # Step 5: Run entity resolution
    logger.info("Running entity resolution on campaign contributions...")
    try:
        from src.processing.normalizer import resolve_campaign_employers
        async with factory() as session:
            resolved = await resolve_campaign_employers(session)
            logger.info("Entity resolution: %d campaign employers resolved", resolved)
    except Exception as e:
        logger.error("Entity resolution error: %s", e)

    elapsed = time.time() - t0
    logger.info(
        "Campaign finance backfill complete in %.1fmin: %d committees, %d contributions, %d errors",
        elapsed / 60, total_committees, total_contributions, errors,
    )
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
