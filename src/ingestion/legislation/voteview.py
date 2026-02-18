"""Voteview integration for DW-NOMINATE ideology scores.

Downloads HSall_members.csv from voteview.com and extracts
ideology scores (nominate_dim1, nominate_dim2) for congress members.

Source: https://voteview.com/data
"""

from __future__ import annotations

import io
import logging
from typing import Any

import pandas as pd

from src.ingestion.base import BaseCollector

logger = logging.getLogger(__name__)

VOTEVIEW_MEMBERS_URL = "https://voteview.com/static/data/out/members/HSall_members.csv"


class VoteviewCollector(BaseCollector):
    """Collect DW-NOMINATE ideology scores from Voteview."""

    source_name = "voteview"

    def __init__(self, congress: int | None = None) -> None:
        super().__init__()
        self.congress = congress  # None = get latest for each member

    async def collect(self) -> list[dict[str, Any]]:
        """Download and parse Voteview CSV."""
        try:
            response = await self.client.get(VOTEVIEW_MEMBERS_URL, timeout=120.0)
            response.raise_for_status()
        except Exception:
            logger.exception("Failed to download Voteview data")
            return []

        try:
            df = pd.read_csv(io.StringIO(response.text), low_memory=False)
        except Exception:
            logger.exception("Failed to parse Voteview CSV")
            return []

        # Filter to rows with bioguide_id
        df = df[df["bioguide_id"].notna() & (df["bioguide_id"] != "")]

        if self.congress:
            # Get specific congress
            df = df[df["congress"] == self.congress]
        else:
            # Get the latest congress entry for each member
            df = df.sort_values("congress", ascending=False)
            df = df.drop_duplicates(subset=["bioguide_id"], keep="first")

        logger.info("Parsed %d member records from Voteview", len(df))
        return df.to_dict("records")

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        bioguide_id = str(raw.get("bioguide_id", "")).strip()
        if not bioguide_id:
            return None

        dim1 = raw.get("nominate_dim1")
        dim2 = raw.get("nominate_dim2")

        # Skip if both scores are missing
        if pd.isna(dim1) and pd.isna(dim2):
            return None

        # Normalize chamber
        chamber_code = raw.get("chamber", "")
        if chamber_code == "Senate":
            chamber = "senate"
        elif chamber_code == "House":
            chamber = "house"
        else:
            chamber = None

        # Parse party
        party_code = raw.get("party_code")
        if party_code == 100:
            party = "Democratic"
        elif party_code == 200:
            party = "Republican"
        elif party_code == 328:
            party = "Independent"
        else:
            party = str(party_code) if party_code else None

        # Build name from bioname (format: "LASTNAME, Firstname")
        bioname = raw.get("bioname", "")
        full_name = bioname
        if ", " in bioname:
            parts = bioname.split(", ", 1)
            last = parts[0].title()
            first = parts[1].title() if len(parts) > 1 else ""
            full_name = f"{first} {last}".strip()

        return {
            "bioguide_id": bioguide_id,
            "nominate_dim1": float(dim1) if not pd.isna(dim1) else None,
            "nominate_dim2": float(dim2) if not pd.isna(dim2) else None,
            "congress_number": int(raw.get("congress", 0)),
            "chamber": chamber,
            "party": party,
            "state": raw.get("state_abbrev", ""),
            "full_name": full_name,
        }
