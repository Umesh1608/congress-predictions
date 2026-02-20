"""Backfill bill detail (sponsor + introduced_date) from Congress.gov.

Run: python -m scripts.backfill_bill_sponsors [--concurrency N]
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import date as date_type

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

CONGRESS_API_BASE = "https://api.congress.gov/v3"


async def fetch_one(client, api_key, congress, bill_type, bill_number, sem):
    async with sem:
        url = f"{CONGRESS_API_BASE}/bill/{congress}/{bill_type}/{bill_number}"
        try:
            r = await client.get(url, params={"api_key": api_key, "format": "json"})
            if r.status_code == 429:
                await asyncio.sleep(30)
                r = await client.get(url, params={"api_key": api_key, "format": "json"})
            if r.status_code != 200:
                return None
            return r.json().get("bill", {})
        except Exception:
            return None


async def main():
    concurrency = 5
    for a in sys.argv[1:]:
        if a.startswith("--concurrency="):
            concurrency = int(a.split("=")[1])

    api_key = settings.congress_gov_api_key
    if not api_key:
        logger.error("No CONGRESS_GOV_API_KEY configured")
        return

    engine = create_async_engine(settings.database_url, pool_size=10)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    # Get ALL bills missing introduced_date (the critical missing field)
    async with factory() as session:
        r = await session.execute(text(
            "SELECT bill_id, congress_number, bill_type, bill_number FROM bill "
            "WHERE introduced_date IS NULL ORDER BY congress_number DESC"
        ))
        bills = r.fetchall()

    # Also get the set of valid bioguide_ids to avoid FK violations
    async with factory() as session:
        r = await session.execute(text("SELECT bioguide_id FROM congress_member"))
        valid_bios = {row[0] for row in r.fetchall()}

    logger.info("Bills needing introduced_date: %d (valid members: %d)", len(bills), len(valid_bios))
    if not bills:
        await engine.dispose()
        return

    sem = asyncio.Semaphore(concurrency)
    updated = 0
    errors = 0
    t0 = time.time()

    BATCH = 50
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(0, len(bills), BATCH):
            batch = bills[i:i + BATCH]
            tasks = [fetch_one(client, api_key, c, bt, bn, sem) for _, c, bt, bn in batch]
            details = await asyncio.gather(*tasks)

            async with factory() as session:
                for (bill_id, _, _, _), detail in zip(batch, details):
                    if detail is None:
                        errors += 1
                        continue

                    intro_str = (detail.get("introducedDate") or "")[:10]
                    intro = None
                    if intro_str:
                        try:
                            y, m, d = intro_str.split("-")
                            intro = date_type(int(y), int(m), int(d))
                        except (ValueError, TypeError):
                            pass
                    pa = (detail.get("policyArea") or {}).get("name")
                    sponsors = detail.get("sponsors", [])
                    bio = sponsors[0].get("bioguideId") if sponsors else None
                    sname = sponsors[0].get("fullName") if sponsors else None

                    # Only set sponsor_bioguide_id if it's a valid FK
                    if bio and bio not in valid_bios:
                        bio = None

                    parts = []
                    params = {"bid": bill_id}
                    if intro:
                        parts.append("introduced_date = :intro")
                        params["intro"] = intro
                    if pa:
                        parts.append("policy_area = :pa")
                        params["pa"] = pa
                    if bio:
                        parts.append("sponsor_bioguide_id = :bio")
                        params["bio"] = bio
                    if sname:
                        parts.append("sponsor_name = :sname")
                        params["sname"] = sname

                    if parts:
                        await session.execute(
                            text(f"UPDATE bill SET {', '.join(parts)} WHERE bill_id = :bid"),
                            params,
                        )
                        updated += 1

                await session.commit()

            done = i + len(batch)
            elapsed = time.time() - t0
            rate = done / elapsed * 60 if elapsed > 0 else 0
            eta = (len(bills) - done) / rate if rate > 0 else 0

            if done % 500 == 0 or done >= len(bills):
                logger.info("Progress: %d/%d (updated=%d, errors=%d) %.0f/min ~%.0fmin left",
                            done, len(bills), updated, errors, rate, eta)

            await asyncio.sleep(0.3)

    logger.info("Done in %.1fmin: updated=%d, errors=%d / %d",
                (time.time() - t0) / 60, updated, errors, len(bills))
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
