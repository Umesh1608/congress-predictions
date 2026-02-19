"""Link trade_disclosure records to congress_member via member_bioguide_id.

Parses member_name from trades (formats: "Last, Hon.. First" for house_clerk,
"Last, First" for github_senate) and matches against congress_member.full_name.

Run: python -m scripts.link_trades_to_members
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict

from sqlalchemy import text, update

from src.db.postgres import async_session_factory

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Suffixes to strip from first name portion
_SUFFIX_RE = re.compile(
    r"\s+(Jr\.?|Sr\.?|III\.?|II\.?|IV\.?|Mr\.?|Mrs\.?|Ms\.?|Dr\.?|MD|FACS|Hon)\s*$",
    re.IGNORECASE,
)

# Manual overrides for names that can't be auto-matched
_MANUAL_OVERRIDES: dict[str, str] = {
    "McConnell, A. Mitchell Jr.": "M000355",      # Mitch McConnell
    "Allen, Hon.. Richard W.": "A000372",          # Rick Allen
    "Banks, Hon.. James E Hon": "B001311",         # Jim Banks
    "Barragan, Hon.. Nanette": "B001300",          # Nanette Barragan
    "Casey, Robert P. Jr.": "C001070",             # Bob Casey
    "Crapo, Michael D.": "C000880",                # Mike Crapo
    "Crenshaw, Hon.. Daniel": "C001120",           # Dan Crenshaw
    "Delaney, Hon.. April McClain": "D000632",     # April McClain Delaney (née McClain)
    "Dunn, MD, FACS, Hon.. Neal Patrick": "D000628",  # Neal Dunn
    "Fallon, Hon.. Patrick": "F000246",            # Pat Fallon
    "Franklin, Hon.. C. Scott": "F000472",         # Scott Franklin
    "Garcia, Hon.. Michael": "G000061",            # Mike Garcia
    "Green, Hon.. Mark Dr": "G000590",             # Mark Green
    "Hill, Hon.. James French": "H001082",         # French Hill
    "Jacobs, Hon.. Christopher L.": "J000020",     # Chris Jacobs
    "Keating, Hon.. William R.": "K000375",        # Bill Keating
    "Kennedy, John": "K000393",                    # John Kennedy (senator, LA)
    "McCormick, Hon.. Richard Dean Dr": "M001224",  # Rich McCormick
    "McGuire, Hon.. John": "M001225",              # John McGuire
    "Moore, Hon.. Felix Barry": "M001212",         # Barry Moore
    "Reed, John F.": "R000122",                    # Jack Reed
    "Webster, Hon.. Daniel": "W000806",            # Daniel Webster
}


def _normalize_trade_name(member_name: str) -> tuple[str, str] | None:
    """Parse trade member_name into (last_name, first_name).

    Returns None if the name can't be parsed (e.g. just a state name).
    """
    if "," not in member_name:
        return None

    parts = member_name.split(",", 1)
    last_raw = parts[0].strip()
    first_raw = parts[1].strip()

    # Remove "Hon.." prefix (house clerk format)
    first_raw = re.sub(r"Hon\.+\s*", "", first_raw).strip()

    # Remove trailing suffixes
    first_raw = _SUFFIX_RE.sub("", first_raw).strip()

    # Remove trailing periods
    first_raw = first_raw.rstrip(".")

    return (last_raw, first_raw)


def _build_member_lookup(
    members: list[tuple],
) -> dict[str, list[tuple[str, str, str | None, str | None]]]:
    """Build last_name -> [(bioguide_id, full_name, last_name, first_name)] lookup."""
    by_last: dict[str, list[tuple[str, str, str | None, str | None]]] = defaultdict(list)

    for bio, full, last, first, _chamber in members:
        if last:
            key = last.lower().strip()
        else:
            # Extract last word of full_name
            name_parts = full.strip().split()
            if not name_parts:
                continue
            key = name_parts[-1].lower()
        by_last[key].append((bio, full, last, first))

    return dict(by_last)


def _match_name(
    last_raw: str,
    first_raw: str,
    lookup: dict[str, list[tuple[str, str, str | None, str | None]]],
) -> str | None:
    """Find bioguide_id for a (last, first) name pair."""
    last_key = last_raw.lower().strip()
    candidates = lookup.get(last_key, [])

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0][0]

    # Multiple candidates — narrow by first name
    first_lower = first_raw.split()[0].lower() if first_raw else ""
    if not first_lower:
        return None

    matched = []
    for bio, full, last, first in candidates:
        full_lower = full.lower()
        first_db = (first or "").lower()
        if first_lower == first_db or first_lower in full_lower.split():
            matched.append(bio)

    if len(matched) == 1:
        return matched[0]

    # Try substring match on full_name
    if not matched:
        for bio, full, last, first in candidates:
            if first_lower in full.lower():
                matched.append(bio)
        if len(matched) == 1:
            return matched[0]

    return None


async def link_trades() -> None:
    """Link all unlinked trades to congress members."""
    async with async_session_factory() as session:
        # Load all members
        r = await session.execute(
            text("SELECT bioguide_id, full_name, last_name, first_name, chamber FROM congress_member")
        )
        members = r.fetchall()
        lookup = _build_member_lookup(members)
        logger.info("Built member lookup: %d last names, %d total members", len(lookup), len(members))

        # Get all distinct unlinked member_names
        r = await session.execute(
            text("""
                SELECT DISTINCT member_name
                FROM trade_disclosure
                WHERE member_bioguide_id IS NULL
                AND member_name IS NOT NULL
            """)
        )
        trade_names = [row[0] for row in r.fetchall()]
        logger.info("Found %d distinct unlinked trade member names", len(trade_names))

        # Match each name
        matched = 0
        unmatched = []
        name_to_bio: dict[str, str] = {}

        for name in trade_names:
            # Check manual overrides first
            if name in _MANUAL_OVERRIDES:
                name_to_bio[name] = _MANUAL_OVERRIDES[name]
                matched += 1
                continue

            parsed = _normalize_trade_name(name)
            if not parsed:
                unmatched.append(name)
                continue

            last_raw, first_raw = parsed
            bio = _match_name(last_raw, first_raw, lookup)
            if bio:
                name_to_bio[name] = bio
                matched += 1
            else:
                unmatched.append(name)

        logger.info("Matched %d / %d names", matched, len(trade_names))
        if unmatched:
            logger.info("Unmatched names (%d):", len(unmatched))
            for n in sorted(unmatched):
                logger.info("  %s", n)

        # Update trades in bulk per member_name
        total_updated = 0
        for name, bio in name_to_bio.items():
            r = await session.execute(
                text("""
                    UPDATE trade_disclosure
                    SET member_bioguide_id = :bio
                    WHERE member_name = :name
                    AND member_bioguide_id IS NULL
                """),
                {"bio": bio, "name": name},
            )
            total_updated += r.rowcount

        await session.commit()
        logger.info("Updated %d trade records with member_bioguide_id", total_updated)

        # Verify
        r = await session.execute(
            text("SELECT COUNT(*) FROM trade_disclosure WHERE member_bioguide_id IS NOT NULL")
        )
        logger.info("Total trades now linked: %d", r.scalar())

        r = await session.execute(
            text("SELECT COUNT(*) FROM trade_disclosure WHERE member_bioguide_id IS NULL")
        )
        logger.info("Total trades still unlinked: %d", r.scalar())


if __name__ == "__main__":
    asyncio.run(link_trades())
