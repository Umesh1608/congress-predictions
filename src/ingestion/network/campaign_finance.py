"""FEC campaign finance data collector.

FEC provides bulk data downloads and an API for campaign contributions.
We use the FEC API (https://api.open.fec.gov/v1/) to fetch:
1. Candidate committees (link candidates to their campaign committees)
2. Individual contributions to those committees (link company employees to members)

FEC API key: free, sign up at https://api.data.gov/signup/
Rate limit: 1000 requests/hour per API key.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from src.ingestion.base import BaseCollector, RateLimiter

logger = logging.getLogger(__name__)

FEC_API_BASE = "https://api.open.fec.gov/v1"

# Be conservative — FEC allows 1000/hr, we use well under that
_fec_rate_limiter = RateLimiter(max_calls=3, period_seconds=1.0)


class FECCommitteeCollector(BaseCollector):
    """Collect campaign committees linked to congressional candidates.

    Maps FEC committee IDs to candidate IDs so we can link contributions
    to specific congress members.
    """

    source_name = "fec_committees"
    rate_limiter = _fec_rate_limiter

    def __init__(self, api_key: str = "", cycle: int | None = None) -> None:
        super().__init__()
        self.api_key = api_key
        self.cycle = cycle or (date.today().year if date.today().year % 2 == 0 else date.today().year - 1)
        self.max_pages = 20

    async def collect(self) -> list[dict[str, Any]]:
        """Fetch congressional candidate committees from FEC API."""
        if not self.api_key:
            logger.warning("[%s] No FEC API key configured, skipping", self.source_name)
            return []

        all_committees: list[dict[str, Any]] = []

        for committee_type in ["H", "S"]:  # House and Senate
            url = f"{FEC_API_BASE}/committees/"
            page = 1

            while page <= self.max_pages:
                params = {
                    "api_key": self.api_key,
                    "committee_type": committee_type,
                    "cycle": self.cycle,
                    "per_page": 100,
                    "page": page,
                    "is_active": True,
                }

                try:
                    data = await self.fetch_json(url, params=params)
                    if not data:
                        break

                    results = data.get("results", [])
                    if not results:
                        break

                    all_committees.extend(results)
                    pagination = data.get("pagination", {})
                    if page >= pagination.get("pages", 0):
                        break

                    page += 1
                except Exception:
                    logger.exception("[%s] Error fetching page %d", self.source_name, page)
                    break

        return all_committees

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        """Transform FEC committee record."""
        committee_id = raw.get("committee_id", "")
        if not committee_id:
            return None

        # Get the linked candidate
        candidate_ids = raw.get("candidate_ids", []) or []
        candidate_id = candidate_ids[0] if candidate_ids else None

        return {
            "fec_committee_id": committee_id,
            "name": raw.get("name", ""),
            "committee_type": raw.get("committee_type"),
            "party": raw.get("party"),
            "state": raw.get("state"),
            "candidate_fec_id": candidate_id,
        }


class FECContributionCollector(BaseCollector):
    """Collect individual contributions to congressional campaign committees.

    Focuses on itemized contributions ($200+) which include employer information.
    This employer data is key for linking companies to congress members through
    their employees' donations.
    """

    source_name = "fec_contributions"
    rate_limiter = _fec_rate_limiter

    def __init__(
        self,
        api_key: str = "",
        committee_ids: list[str] | None = None,
        min_amount: int = 200,
        cycle: int | None = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.committee_ids = committee_ids or []
        self.min_amount = min_amount
        self.cycle = cycle or (date.today().year if date.today().year % 2 == 0 else date.today().year - 1)
        self.max_pages = 10  # Per committee

    async def collect(self) -> list[dict[str, Any]]:
        """Fetch itemized contributions for specified committees."""
        if not self.api_key:
            logger.warning("[%s] No FEC API key configured, skipping", self.source_name)
            return []

        all_contributions: list[dict[str, Any]] = []

        for committee_id in self.committee_ids:
            url = f"{FEC_API_BASE}/schedules/schedule_a/"
            page = 1

            while page <= self.max_pages:
                params = {
                    "api_key": self.api_key,
                    "committee_id": committee_id,
                    "two_year_transaction_period": self.cycle,
                    "min_amount": self.min_amount,
                    "per_page": 100,
                    "sort": "-contribution_receipt_date",
                    "page": page,
                    "is_individual": True,
                }

                try:
                    data = await self.fetch_json(url, params=params)
                    if not data:
                        break

                    results = data.get("results", [])
                    if not results:
                        break

                    # Attach committee_id for linking
                    for r in results:
                        r["_committee_id"] = committee_id

                    all_contributions.extend(results)
                    pagination = data.get("pagination", {})
                    if page >= pagination.get("pages", 0):
                        break

                    page += 1
                except Exception:
                    logger.exception(
                        "[%s] Error fetching contributions for %s page %d",
                        self.source_name, committee_id, page,
                    )
                    break

        return all_contributions

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        """Transform FEC contribution record."""
        contributor_name = raw.get("contributor_name", "")
        if not contributor_name:
            return None

        amount = raw.get("contribution_receipt_amount")
        if amount is None:
            return None

        # Parse contribution date
        contrib_date = None
        date_str = raw.get("contribution_receipt_date")
        if date_str:
            try:
                contrib_date = date.fromisoformat(date_str[:10])
            except (ValueError, TypeError):
                pass

        return {
            "fec_transaction_id": raw.get("transaction_id"),
            "fec_committee_id": raw.get("_committee_id") or raw.get("committee_id"),
            "contributor_name": contributor_name,
            "contributor_employer": raw.get("contributor_employer"),
            "contributor_occupation": raw.get("contributor_occupation"),
            "contributor_city": raw.get("contributor_city"),
            "contributor_state": raw.get("contributor_state"),
            "contributor_zip": raw.get("contributor_zip_code"),
            "amount": float(amount),
            "contribution_date": contrib_date,
            "transaction_type": raw.get("receipt_type"),
        }
