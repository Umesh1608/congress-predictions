"""Backfill historical committee assignments from Congress.gov API.

Fetches committee membership for congresses 114-118 (2015-2025).
Currently we only have Congress 119 assignments.

The Congress.gov API provides committee membership via:
  /member/{bioguide_id} → member detail with committee info

Run: python -m scripts.backfill_committee_assignments [--concurrency N] [--congresses 114,115,116]
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import Any

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

# Congress number to year range mapping
CONGRESS_YEARS = {
    114: (2015, 2016),
    115: (2017, 2018),
    116: (2019, 2020),
    117: (2021, 2022),
    118: (2023, 2024),
    119: (2025, 2026),
}


async def fetch_committee_members(
    client: httpx.AsyncClient,
    api_key: str,
    congress: int,
    chamber: str,
    sem: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    """Fetch all committees and their members for a congress/chamber."""
    assignments = []

    # First get list of committees
    async with sem:
        url = f"{CONGRESS_API_BASE}/committee/{chamber}"
        try:
            r = await client.get(url, params={
                "api_key": api_key, "format": "json", "limit": 250,
            })
            if r.status_code == 429:
                retry = int(r.headers.get("Retry-After", 60))
                logger.warning("Rate limited, waiting %ds...", retry)
                await asyncio.sleep(retry)
                r = await client.get(url, params={
                    "api_key": api_key, "format": "json", "limit": 250,
                })
            if r.status_code != 200:
                logger.error("Failed to get %s committees: %d", chamber, r.status_code)
                return []
            committees = r.json().get("committees", [])
        except Exception as e:
            logger.error("Error fetching %s committees: %s", chamber, e)
            return []

    await asyncio.sleep(0.3)

    # For each committee, try to get members for this specific congress
    for comm in committees:
        system_code = comm.get("systemCode", "")
        comm_name = comm.get("name", "")
        if not system_code:
            continue

        # Try the congress-specific committee endpoint
        async with sem:
            url = f"{CONGRESS_API_BASE}/committee/{congress}/{chamber}/{system_code.lower()}"
            try:
                r = await client.get(url, params={
                    "api_key": api_key, "format": "json",
                })
                if r.status_code == 429:
                    retry = int(r.headers.get("Retry-After", 60))
                    logger.warning("Rate limited, waiting %ds...", retry)
                    await asyncio.sleep(retry)
                    r = await client.get(url, params={
                        "api_key": api_key, "format": "json",
                    })
                if r.status_code != 200:
                    continue

                data = r.json().get("committee", {})
                # Check for members in the response
                # The API may return them under different keys
                members = []
                for key in ["members", "committeeMembers", "currentMembers"]:
                    if key in data and isinstance(data[key], list):
                        members = data[key]
                        break

                # If no direct member list, check for member URL
                if not members and "url" in data:
                    # Some committees have a members sub-resource
                    member_url = data["url"]
                    if not member_url.endswith("/"):
                        member_url += "/"
                    member_url += "members"

                for m in members:
                    bio_id = m.get("bioguideId", "")
                    if bio_id:
                        assignments.append({
                            "member_bioguide_id": bio_id,
                            "committee_code": system_code,
                            "committee_name": comm_name,
                            "role": _extract_role(m),
                            "congress_number": congress,
                        })

            except Exception:
                pass

        await asyncio.sleep(0.3)

    return assignments


def _extract_role(member_data: dict) -> str:
    """Extract member role from committee member data."""
    role = member_data.get("role", "")
    if not role:
        role = member_data.get("rankInParty", "")
    role_lower = role.lower() if role else ""
    if "chair" in role_lower and "ranking" not in role_lower:
        return "chair"
    if "ranking" in role_lower:
        return "ranking"
    return "member"


async def fetch_member_committees(
    client: httpx.AsyncClient,
    api_key: str,
    bioguide_id: str,
    sem: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    """Fetch committee assignments for a specific member from their detail page."""
    async with sem:
        url = f"{CONGRESS_API_BASE}/member/{bioguide_id}"
        try:
            r = await client.get(url, params={
                "api_key": api_key, "format": "json",
            })
            if r.status_code == 429:
                retry = int(r.headers.get("Retry-After", 60))
                logger.warning("Rate limited on %s, waiting %ds...", bioguide_id, retry)
                await asyncio.sleep(retry)
                r = await client.get(url, params={
                    "api_key": api_key, "format": "json",
                })
            if r.status_code != 200:
                return []

            member = r.json().get("member", {})
            assignments = []

            # Check for terms with committee info
            terms = member.get("terms", [])
            for term in terms:
                congress_num = term.get("congress")
                if not congress_num:
                    continue
                try:
                    congress_num = int(congress_num)
                except (ValueError, TypeError):
                    continue

                # Only care about 114-119
                if congress_num < 114 or congress_num > 119:
                    continue

                # Look for committees in term data
                committees = term.get("committees", [])
                for c in committees:
                    code = c.get("systemCode", "") or c.get("code", "")
                    name = c.get("name", "")
                    if code:
                        assignments.append({
                            "member_bioguide_id": bioguide_id,
                            "committee_code": code,
                            "committee_name": name,
                            "role": "member",
                            "congress_number": congress_num,
                        })

            return assignments

        except Exception:
            return []


async def main():
    concurrency = 3
    target_congresses = [114, 115, 116, 117, 118]

    for a in sys.argv[1:]:
        if a.startswith("--concurrency="):
            concurrency = int(a.split("=")[1])
        elif a.startswith("--congresses="):
            target_congresses = [int(x) for x in a.split("=")[1].split(",")]

    api_key = settings.congress_gov_api_key
    if not api_key:
        logger.error("No CONGRESS_GOV_API_KEY configured")
        return

    engine = create_async_engine(settings.database_url, pool_size=5)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    # Get existing assignment count
    async with factory() as session:
        r = await session.execute(text(
            "SELECT congress_number, COUNT(*) FROM committee_assignment "
            "GROUP BY congress_number ORDER BY congress_number"
        ))
        existing = dict(r.fetchall())
        logger.info("Existing assignments: %s", existing)

    # Get all trading members (these are the ones we care about)
    async with factory() as session:
        r = await session.execute(text("""
            SELECT DISTINCT m.bioguide_id
            FROM congress_member m
            JOIN trade_disclosure t ON t.member_bioguide_id = m.bioguide_id
            WHERE m.bioguide_id IS NOT NULL
        """))
        trading_members = [row[0] for row in r.fetchall()]

    logger.info("Trading members to fetch committee data for: %d", len(trading_members))
    logger.info("Target congresses: %s", target_congresses)

    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    total_inserted = 0
    total_errors = 0

    # Strategy: Fetch each member's detail page to get their committee history
    # This is more efficient than fetching each committee's member list
    BATCH = 20
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(0, len(trading_members), BATCH):
            batch = trading_members[i:i + BATCH]
            tasks = [
                fetch_member_committees(client, api_key, bio, sem)
                for bio in batch
            ]
            results = await asyncio.gather(*tasks)

            # Filter to target congresses and insert
            all_assignments = []
            for member_assignments in results:
                for a in member_assignments:
                    if a["congress_number"] in target_congresses:
                        all_assignments.append(a)

            if all_assignments:
                async with factory() as session:
                    for a in all_assignments:
                        try:
                            await session.execute(text("""
                                INSERT INTO committee_assignment
                                    (member_bioguide_id, committee_code, committee_name,
                                     role, congress_number)
                                VALUES (:bio, :code, :name, :role, :congress)
                                ON CONFLICT DO NOTHING
                            """), {
                                "bio": a["member_bioguide_id"],
                                "code": a["committee_code"],
                                "name": a["committee_name"],
                                "role": a["role"],
                                "congress": a["congress_number"],
                            })
                            total_inserted += 1
                        except Exception:
                            total_errors += 1
                    await session.commit()

            done = min(i + BATCH, len(trading_members))
            if done % 100 == 0 or done >= len(trading_members):
                elapsed = time.time() - t0
                rate = done / elapsed * 60 if elapsed > 0 else 0
                eta = (len(trading_members) - done) / rate if rate > 0 else 0
                logger.info(
                    "Progress: %d/%d members (assignments=%d, errors=%d) %.0f/min ~%.0fmin left",
                    done, len(trading_members), total_inserted, total_errors, rate, eta,
                )

            await asyncio.sleep(0.2)

    # If member detail approach yielded nothing, try committee-level approach
    if total_inserted == 0:
        logger.info("Member detail approach yielded no results. Trying committee-level fetch...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            for congress in target_congresses:
                for chamber in ["house", "senate"]:
                    logger.info("Fetching %s committees for Congress %d...", chamber, congress)
                    assignments = await fetch_committee_members(
                        client, api_key, congress, chamber, sem
                    )
                    if assignments:
                        async with factory() as session:
                            for a in assignments:
                                try:
                                    await session.execute(text("""
                                        INSERT INTO committee_assignment
                                            (member_bioguide_id, committee_code,
                                             committee_name, role, congress_number)
                                        VALUES (:bio, :code, :name, :role, :congress)
                                        ON CONFLICT DO NOTHING
                                    """), {
                                        "bio": a["member_bioguide_id"],
                                        "code": a["committee_code"],
                                        "name": a["committee_name"],
                                        "role": a["role"],
                                        "congress": a["congress_number"],
                                    })
                                    total_inserted += 1
                                except Exception:
                                    total_errors += 1
                            await session.commit()
                        logger.info("  %s/%d: %d assignments", chamber, congress, len(assignments))

    # Report final counts
    async with factory() as session:
        r = await session.execute(text(
            "SELECT congress_number, COUNT(*) FROM committee_assignment "
            "GROUP BY congress_number ORDER BY congress_number"
        ))
        final = dict(r.fetchall())
        logger.info("Final assignments by congress: %s", final)

    elapsed = time.time() - t0
    logger.info(
        "Committee assignment backfill complete in %.1fmin: %d inserted, %d errors",
        elapsed / 60, total_inserted, total_errors,
    )
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
