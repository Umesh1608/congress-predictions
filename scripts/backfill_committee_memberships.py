"""Backfill historical committee assignments from GovTrack XML data.

Uses GovTrack's historical-committee-membership repository:
  https://github.com/govtrack/historical-committee-membership

Mapping files from unitedstates/congress-legislators:
  https://github.com/unitedstates/congress-legislators

Data files expected in data/govtrack/:
  - 114.xml, 115.xml, ..., 118.xml  (committee membership snapshots)
  - legislators-current.json         (current legislator ID mapping)
  - legislators-historical.json      (historical legislator ID mapping)

Run: python -m scripts.backfill_committee_memberships
"""

from __future__ import annotations

import asyncio
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "govtrack"


def build_govtrack_to_bioguide_map() -> dict[int, str]:
    """Build mapping from GovTrack numeric ID to bioguide_id."""
    mapping: dict[int, str] = {}

    for filename in ["legislators-current.json", "legislators-historical.json"]:
        path = DATA_DIR / filename
        if not path.exists():
            logger.warning("Missing %s", path)
            continue
        with open(path) as f:
            legislators = json.load(f)
        for leg in legislators:
            ids = leg.get("id", {})
            govtrack_id = ids.get("govtrack")
            bioguide_id = ids.get("bioguide")
            if govtrack_id and bioguide_id:
                mapping[int(govtrack_id)] = bioguide_id

    logger.info("Built GovTrack→bioguide mapping: %d entries", len(mapping))
    return mapping


def parse_committee_xml(congress: int, id_map: dict[int, str]) -> list[dict]:
    """Parse a GovTrack committee membership XML file."""
    path = DATA_DIR / f"{congress}.xml"
    if not path.exists():
        logger.warning("Missing %s", path)
        return []

    tree = ET.parse(path)
    root = tree.getroot()
    assignments: list[dict] = []

    for committee_el in root.findall("committee"):
        committee_code = committee_el.get("code", "").lower()
        committee_name = committee_el.get("displayname", "")
        committee_type = committee_el.get("type", "")

        if not committee_code:
            continue

        # Add "00" suffix for main committees to match congress.gov format
        # GovTrack uses "HSAG", congress.gov uses "hsag00"
        if len(committee_code) == 4:
            full_code = committee_code + "00"
        else:
            full_code = committee_code

        # Process committee members
        for member_el in committee_el.findall("member"):
            govtrack_id_str = member_el.get("id", "")
            if not govtrack_id_str:
                continue
            try:
                govtrack_id = int(govtrack_id_str)
            except ValueError:
                continue

            bioguide_id = id_map.get(govtrack_id)
            if not bioguide_id:
                continue

            role = member_el.get("role", "member").lower()
            if role in ("chair", "chairman"):
                role = "chair"
            elif role in ("ranking member",):
                role = "ranking"
            elif role in ("vice chair", "vice chairman"):
                role = "vice_chair"
            else:
                role = "member"

            assignments.append({
                "member_bioguide_id": bioguide_id,
                "committee_code": full_code,
                "committee_name": committee_name,
                "role": role,
                "congress_number": congress,
            })

        # Process subcommittee members
        for sub_el in committee_el.findall("subcommittee"):
            sub_code = sub_el.get("code", "")
            sub_name = sub_el.get("displayname", "")
            if not sub_code:
                continue

            # Subcommittee full code = parent code + sub code
            sub_full_code = committee_code + sub_code
            if len(sub_full_code) < 6:
                sub_full_code = sub_full_code.ljust(6, "0")

            for member_el in sub_el.findall("member"):
                govtrack_id_str = member_el.get("id", "")
                if not govtrack_id_str:
                    continue
                try:
                    govtrack_id = int(govtrack_id_str)
                except ValueError:
                    continue

                bioguide_id = id_map.get(govtrack_id)
                if not bioguide_id:
                    continue

                role = member_el.get("role", "member").lower()
                if role in ("chair", "chairman"):
                    role = "chair"
                elif role in ("ranking member",):
                    role = "ranking"
                else:
                    role = "member"

                assignments.append({
                    "member_bioguide_id": bioguide_id,
                    "committee_code": sub_full_code,
                    "committee_name": sub_name or f"{committee_name} - Subcommittee",
                    "role": role,
                    "congress_number": congress,
                })

    logger.info("Congress %d: parsed %d assignments from XML", congress, len(assignments))
    return assignments


async def upsert_assignments(session, assignments: list[dict], known_members: set[str]) -> int:
    """Insert committee assignments, skipping duplicates and unknown members."""
    if not assignments:
        return 0

    inserted = 0
    skipped_fk = 0
    for a in assignments:
        # Skip members not in our congress_member table
        if a["member_bioguide_id"] not in known_members:
            skipped_fk += 1
            continue

        # Check if already exists
        r = await session.execute(text("""
            SELECT 1 FROM committee_assignment
            WHERE member_bioguide_id = :bio AND committee_code = :code AND congress_number = :congress
            LIMIT 1
        """), {"bio": a["member_bioguide_id"], "code": a["committee_code"], "congress": a["congress_number"]})
        if r.fetchone():
            continue

        await session.execute(text("""
            INSERT INTO committee_assignment (member_bioguide_id, committee_code, committee_name, role, congress_number)
            VALUES (:member_bioguide_id, :committee_code, :committee_name, :role, :congress_number)
        """), a)
        inserted += 1

    await session.commit()
    if skipped_fk:
        logger.info("  Skipped %d assignments (member not in DB)", skipped_fk)
    return inserted


async def main():
    id_map = build_govtrack_to_bioguide_map()

    engine = create_async_engine(settings.database_url, pool_size=3)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    # Get known members from DB
    async with factory() as session:
        r = await session.execute(text("SELECT bioguide_id FROM congress_member"))
        known_members = {row[0] for row in r.fetchall()}
        logger.info("Known members in DB: %d", len(known_members))

        r = await session.execute(text(
            "SELECT congress_number, COUNT(*) FROM committee_assignment GROUP BY congress_number ORDER BY congress_number"
        ))
        existing = dict(r.fetchall())
        logger.info("Existing assignments: %s", existing)

    congresses = [114, 115, 116, 117, 118]
    total_inserted = 0

    for congress in congresses:
        if existing.get(congress, 0) > 100:
            logger.info("Congress %d already has %d assignments, skipping", congress, existing[congress])
            continue

        assignments = parse_committee_xml(congress, id_map)
        if assignments:
            async with factory() as session:
                inserted = await upsert_assignments(session, assignments, known_members)
                total_inserted += inserted
                logger.info("Congress %d: inserted %d new assignments", congress, inserted)

    # Final summary
    async with factory() as session:
        r = await session.execute(text(
            "SELECT congress_number, COUNT(*) FROM committee_assignment GROUP BY congress_number ORDER BY congress_number"
        ))
        final = dict(r.fetchall())
        total = sum(final.values())
        logger.info("Final assignments by congress: %s", final)
        logger.info("Total assignments: %d (inserted %d new)", total, total_inserted)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
