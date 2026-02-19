"""Collectors for Congress.gov API v3.

API docs: https://api.congress.gov/
Rate limit: 5,000 requests/hour
Authentication: api_key query parameter
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from src.config import settings
from src.ingestion.base import BaseCollector, RateLimiter

logger = logging.getLogger(__name__)

CONGRESS_API_BASE = "https://api.congress.gov/v3"

# Shared rate limiter: 5000/hr ≈ 1.4/sec, use 1/sec for safety
_congress_rate_limiter = RateLimiter(max_calls=1, period_seconds=1.0)


def _current_congress() -> int:
    """Compute the current congress number. The 119th Congress started Jan 2025."""
    year = date.today().year
    return (year - 1789) // 2 + 1


class CongressMemberCollector(BaseCollector):
    """Collect current congress member data from Congress.gov API."""

    source_name = "congress_gov_members"
    rate_limiter = _congress_rate_limiter

    def __init__(self, current_only: bool = True) -> None:
        super().__init__()
        self.current_only = current_only

    async def collect(self) -> list[dict[str, Any]]:
        if not settings.congress_gov_api_key:
            logger.warning("Congress.gov API key not configured, skipping")
            return []

        all_members: list[dict] = []
        offset = 0
        limit = 250  # max allowed

        while True:
            params = {
                "api_key": settings.congress_gov_api_key,
                "format": "json",
                "offset": offset,
                "limit": limit,
            }
            if self.current_only:
                params["currentMember"] = "true"

            data = await self.fetch_json(f"{CONGRESS_API_BASE}/member", params=params)
            if not data or "members" not in data:
                break

            members = data["members"]
            if not members:
                break

            all_members.extend(members)
            offset += limit

            # Check if there are more pages
            pagination = data.get("pagination", {})
            if not pagination.get("next"):
                break

        return all_members

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        bioguide_id = raw.get("bioguideId", "").strip()
        if not bioguide_id:
            return None

        name = raw.get("name", "")
        # Congress.gov returns "LastName, FirstName" format
        first_name = raw.get("firstName", "")
        last_name = raw.get("lastName", "")
        if not first_name and ", " in name:
            parts = name.split(", ", 1)
            last_name = parts[0]
            first_name = parts[1] if len(parts) > 1 else ""

        full_name = f"{first_name} {last_name}".strip() or name

        # Normalize chamber
        chamber_raw = raw.get("terms", {})
        # Try to get from the latest term
        terms = raw.get("terms", []) if isinstance(raw.get("terms"), list) else []
        chamber = "house"
        if terms:
            latest_term = terms[-1] if isinstance(terms[-1], dict) else {}
            chamber_val = latest_term.get("chamber", "")
        else:
            chamber_val = raw.get("chamber", "")

        if "senate" in str(chamber_val).lower():
            chamber = "senate"
        else:
            chamber = "house"

        party = raw.get("partyName", raw.get("party", ""))
        state = raw.get("state", "")
        # State might be full name — extract postal code
        if len(state) > 2:
            state = raw.get("stateCode", state[:2])

        district = raw.get("district")

        return {
            "bioguide_id": bioguide_id,
            "full_name": full_name,
            "first_name": first_name,
            "last_name": last_name,
            "chamber": chamber,
            "state": state[:2] if state else None,
            "district": str(district) if district else None,
            "party": party,
            "in_office": raw.get("currentMember", True),
            "social_accounts": {},
            "raw_data": raw,
        }


class CongressBillCollector(BaseCollector):
    """Collect bills from Congress.gov API for a specific congress session."""

    source_name = "congress_gov_bills"
    rate_limiter = _congress_rate_limiter

    def __init__(self, congress: int | None = None, bill_type: str = "hr") -> None:
        super().__init__()
        self.congress = congress or _current_congress()
        self.bill_type = bill_type

    async def collect(self) -> list[dict[str, Any]]:
        if not settings.congress_gov_api_key:
            logger.warning("Congress.gov API key not configured, skipping")
            return []

        all_bills: list[dict] = []
        offset = 0
        limit = 250

        while True:
            params = {
                "api_key": settings.congress_gov_api_key,
                "format": "json",
                "offset": offset,
                "limit": limit,
            }
            url = f"{CONGRESS_API_BASE}/bill/{self.congress}/{self.bill_type}"
            data = await self.fetch_json(url, params=params)
            if not data or "bills" not in data:
                break

            bills = data["bills"]
            if not bills:
                break

            all_bills.extend(bills)
            offset += limit

            if not data.get("pagination", {}).get("next"):
                break

            # Safety: limit to 2000 bills per type to avoid very long runs
            if offset >= 2000:
                logger.info("Reached 2000 bill limit for %s-%d", self.bill_type, self.congress)
                break

        return all_bills

    async def collect_bill_detail(self, congress: int, bill_type: str, bill_number: int) -> dict | None:
        """Fetch detailed info for a single bill (sponsors, subjects, actions)."""
        if not settings.congress_gov_api_key:
            return None

        params = {"api_key": settings.congress_gov_api_key, "format": "json"}
        url = f"{CONGRESS_API_BASE}/bill/{congress}/{bill_type}/{bill_number}"
        data = await self.fetch_json(url, params=params)
        return data.get("bill") if data else None

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        congress = raw.get("congress")
        bill_type = (raw.get("type") or raw.get("billType", "")).lower()
        number = raw.get("number")

        if not all([congress, bill_type, number]):
            return None

        bill_id = f"{bill_type}{number}-{congress}"

        introduced = _parse_date(raw.get("introducedDate", ""))
        latest_action = raw.get("latestAction") or {}
        latest_action_date = _parse_date(latest_action.get("actionDate", ""))
        latest_action_text = latest_action.get("text", "")

        # Sponsors
        sponsors = raw.get("sponsors", [])
        sponsor_id = None
        sponsor_name = None
        if sponsors and isinstance(sponsors, list):
            sponsor = sponsors[0]
            sponsor_id = sponsor.get("bioguideId")
            sponsor_name = sponsor.get("fullName", sponsor.get("name", ""))

        # Policy area
        policy_area = None
        if raw.get("policyArea"):
            policy_area = raw["policyArea"].get("name")

        # Subjects
        subjects = []
        if raw.get("subjects") and isinstance(raw["subjects"], dict):
            subjects = raw["subjects"].get("legislativeSubjects", [])
            if isinstance(subjects, list):
                subjects = [s.get("name", s) if isinstance(s, dict) else s for s in subjects]

        # Committees
        committees = []
        if raw.get("committees") and isinstance(raw["committees"], list):
            committees = [
                {"code": c.get("systemCode", ""), "name": c.get("name", "")}
                for c in raw["committees"]
            ]

        return {
            "bill_id": bill_id,
            "congress_number": int(congress),
            "bill_type": bill_type,
            "bill_number": int(number),
            "title": raw.get("title", ""),
            "short_title": (raw.get("title", "") or "")[:500],
            "introduced_date": introduced,
            "sponsor_bioguide_id": sponsor_id,
            "sponsor_name": sponsor_name,
            "status": latest_action_text[:100] if latest_action_text else None,
            "subjects": subjects,
            "committees": committees,
            "actions": [],  # Populated from detail endpoint
            "latest_action_date": latest_action_date,
            "latest_action_text": latest_action_text,
            "policy_area": policy_area,
            "raw_data": raw,
        }


class CongressCommitteeCollector(BaseCollector):
    """Collect committee data from Congress.gov API."""

    source_name = "congress_gov_committees"
    rate_limiter = _congress_rate_limiter

    def __init__(self, congress: int | None = None) -> None:
        super().__init__()
        self.congress = congress or _current_congress()

    async def collect(self) -> list[dict[str, Any]]:
        if not settings.congress_gov_api_key:
            logger.warning("Congress.gov API key not configured, skipping")
            return []

        all_committees: list[dict] = []
        offset = 0
        limit = 250

        for chamber_code in ["house", "senate", "joint"]:
            offset = 0
            while True:
                params = {
                    "api_key": settings.congress_gov_api_key,
                    "format": "json",
                    "offset": offset,
                    "limit": limit,
                }
                url = f"{CONGRESS_API_BASE}/committee/{chamber_code}"
                data = await self.fetch_json(url, params=params)
                if not data or "committees" not in data:
                    break

                committees = data["committees"]
                if not committees:
                    break

                all_committees.extend(committees)
                offset += limit

                if not data.get("pagination", {}).get("next"):
                    break

        return all_committees

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        system_code = raw.get("systemCode", "").strip()
        if not system_code:
            return None

        name = raw.get("name", "").strip()
        if not name:
            return None

        chamber_raw = raw.get("chamber", "").lower()
        if "senate" in chamber_raw:
            chamber = "senate"
        elif "house" in chamber_raw:
            chamber = "house"
        else:
            chamber = "joint"

        parent = raw.get("parent", {})
        parent_code = parent.get("systemCode") if parent else None

        return {
            "system_code": system_code,
            "name": name,
            "chamber": chamber,
            "parent_code": parent_code,
            "url": raw.get("url"),
            "is_current": raw.get("isCurrent", True),
        }


class CongressHearingCollector(BaseCollector):
    """Collect hearing data from Congress.gov API."""

    source_name = "congress_gov_hearings"
    rate_limiter = _congress_rate_limiter

    def __init__(self, congress: int | None = None) -> None:
        super().__init__()
        self.congress = congress or _current_congress()

    async def collect(self) -> list[dict[str, Any]]:
        if not settings.congress_gov_api_key:
            logger.warning("Congress.gov API key not configured, skipping")
            return []

        all_hearings: list[dict] = []
        offset = 0
        limit = 250

        while True:
            params = {
                "api_key": settings.congress_gov_api_key,
                "format": "json",
                "offset": offset,
                "limit": limit,
            }
            url = f"{CONGRESS_API_BASE}/hearing/{self.congress}"
            data = await self.fetch_json(url, params=params)
            if not data or "hearings" not in data:
                break

            hearings = data["hearings"]
            if not hearings:
                break

            # Each hearing in the list only has summary fields.
            # Fetch detail to get title, date, etc.
            for hearing in hearings:
                detail_url = hearing.get("url")
                if not detail_url:
                    continue
                detail = await self.fetch_json(
                    detail_url,
                    params={"api_key": settings.congress_gov_api_key, "format": "json"},
                )
                if detail and "hearing" in detail:
                    all_hearings.append(detail["hearing"])

            offset += limit

            if not data.get("pagination", {}).get("next"):
                break

            if offset >= 2000:
                break

        return all_hearings

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        title = (raw.get("title") or "").strip()
        if not title:
            return None

        hearing_date = _parse_date(raw.get("date", ""))

        chamber_raw = (raw.get("chamber") or "").lower()
        if "senate" in chamber_raw:
            chamber = "senate"
        elif "house" in chamber_raw:
            chamber = "house"
        else:
            chamber = None

        # Committee info
        committee = raw.get("committee", {}) or {}
        committee_code = committee.get("systemCode")

        # Related bills
        associated_bills = []
        for bill_ref in (raw.get("associatedBills") or []):
            if isinstance(bill_ref, dict):
                associated_bills.append({
                    "congress": bill_ref.get("congress"),
                    "type": bill_ref.get("type", "").lower(),
                    "number": bill_ref.get("number"),
                })

        return {
            "committee_code": committee_code,
            "title": title,
            "hearing_date": hearing_date,
            "chamber": chamber,
            "congress_number": raw.get("congress") or self.congress,
            "url": raw.get("url"),
            "transcript_url": raw.get("transcriptUrl"),
            "related_bills": associated_bills,
        }


def _parse_date(date_str: str) -> date | None:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(date_str.strip()[:19], fmt).date()
        except ValueError:
            continue
    return None
