"""Senate Lobbying Disclosure Act (LDA) data collector.

Senate publishes lobbying disclosure data as bulk XML downloads:
https://lda.senate.gov/api/

The API provides JSON endpoints for filings, registrants, clients, and lobbyists.
We use the JSON API (no XML parsing needed) with pagination support.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from src.ingestion.base import BaseCollector, RateLimiter

logger = logging.getLogger(__name__)

# Senate LDA API base URL
LDA_API_BASE = "https://lda.senate.gov/api/v1"

# Rate limit: be conservative with Senate servers
_lda_rate_limiter = RateLimiter(max_calls=2, period_seconds=1.0)


class LobbyingFilingCollector(BaseCollector):
    """Collect lobbying disclosure filings from the Senate LDA API.

    The API provides JSON endpoints with pagination. We collect filings
    for a given year range and extract registrants, clients, and lobbyists.
    """

    source_name = "senate_lda"
    rate_limiter = _lda_rate_limiter

    def __init__(self, filing_year: int | None = None) -> None:
        super().__init__()
        self.filing_year = filing_year or date.today().year
        self.max_pages = 50  # Safety limit

    async def collect(self) -> list[dict[str, Any]]:
        """Fetch filings from Senate LDA API with pagination."""
        all_filings: list[dict[str, Any]] = []
        url = f"{LDA_API_BASE}/filings/"
        params: dict[str, Any] = {
            "filing_year": self.filing_year,
            "page_size": 25,
        }

        for page in range(1, self.max_pages + 1):
            params["page"] = page
            try:
                data = await self.fetch_json(url, params=params)
                if not data:
                    break

                results = data.get("results", [])
                if not results:
                    break

                all_filings.extend(results)
                logger.info(
                    "[%s] Page %d: fetched %d filings (total: %d)",
                    self.source_name, page, len(results), len(all_filings),
                )

                if not data.get("next"):
                    break

            except Exception:
                logger.exception("[%s] Error fetching page %d", self.source_name, page)
                break

        return all_filings

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        """Transform a raw LDA filing into structured data.

        Returns a dict with filing info plus nested registrant, client, and lobbyists.
        """
        filing_uuid = raw.get("filing_uuid", "")
        if not filing_uuid:
            return None

        # Extract registrant info
        registrant = raw.get("registrant", {}) or {}
        registrant_data = {
            "senate_id": str(registrant.get("id", "")),
            "name": registrant.get("name", ""),
            "description": registrant.get("description"),
        }

        # Extract client info
        client = raw.get("client", {}) or {}
        client_data = {
            "name": client.get("name", ""),
            "description": client.get("general_description"),
            "country": client.get("country"),
            "state": client.get("state"),
        }

        # Extract issues, bills, and lobbyists from lobbying_activities
        activities = raw.get("lobbying_activities", []) or []
        issue_codes = []
        specific_issues_list = []
        lobbied_bills = []
        gov_entities = []
        lobbyists_data = []
        seen_lobbyist_names: set[str] = set()

        for activity in activities:
            code = activity.get("general_issue_code")
            if code:
                issue_codes.append(code)

            desc = activity.get("description", "")
            if desc:
                specific_issues_list.append(desc)

            for entity in (activity.get("government_entities", []) or []):
                entity_name = entity.get("name", "")
                if entity_name:
                    gov_entities.append(entity_name)

            # Lobbyists are nested inside each activity
            for lob in (activity.get("lobbyists", []) or []):
                covered = lob.get("covered_official_position", "")
                name = _normalize_lobbyist_name(lob)
                if name and name not in seen_lobbyist_names:
                    seen_lobbyist_names.add(name)
                    lobbyists_data.append({
                        "name": name,
                        "covered_position": covered if covered else None,
                        "is_former_congress": _is_former_congress(covered),
                        "is_former_executive": _is_former_executive(covered),
                    })

        # Parse lobbied bills from specific issues text
        # LDA filings sometimes list bill numbers in the description
        for issue_text in specific_issues_list:
            lobbied_bills.extend(_extract_bill_references(issue_text))

        # Parse filing date
        filing_date_str = raw.get("filing_date") or raw.get("dt_posted")
        filing_date = None
        if filing_date_str:
            try:
                filing_date = date.fromisoformat(filing_date_str[:10])
            except (ValueError, TypeError):
                pass

        # Parse amount
        amount = None
        income = raw.get("income")
        expenses = raw.get("expenses")
        if income:
            try:
                amount = float(str(income).replace(",", "").replace("$", ""))
            except (ValueError, TypeError):
                pass
        elif expenses:
            try:
                amount = float(str(expenses).replace(",", "").replace("$", ""))
            except (ValueError, TypeError):
                pass

        return {
            "filing_uuid": filing_uuid,
            "filing_type": raw.get("filing_type_display", raw.get("filing_type", "")),
            "filing_year": self.filing_year,
            "filing_period": raw.get("filing_period"),
            "filing_date": filing_date,
            "amount": amount,
            "registrant": registrant_data,
            "client": client_data,
            "lobbyists": lobbyists_data,
            "specific_issues": specific_issues_list,
            "general_issue_codes": list(set(issue_codes)),
            "government_entities": list(set(gov_entities)),
            "lobbied_bills": lobbied_bills,
        }


def _normalize_lobbyist_name(lob: dict[str, Any]) -> str:
    """Build a consistent name from lobbyist fields."""
    first = (lob.get("lobbyist", {}) or {}).get("first_name", "")
    last = (lob.get("lobbyist", {}) or {}).get("last_name", "")
    if first and last:
        return f"{first} {last}".strip()
    # Fallback to prefix field
    prefix = (lob.get("lobbyist", {}) or {}).get("prefix", "")
    name = f"{prefix} {first} {last}".strip()
    return name if name else "Unknown"


def _is_former_congress(covered_position: str) -> bool:
    """Check if a covered position indicates former congressional service."""
    if not covered_position:
        return False
    pos_lower = covered_position.lower()
    congress_keywords = [
        "senator", "representative", "congressman", "congresswoman",
        "house", "senate", "staff director", "chief of staff",
        "legislative director", "counsel to", "chief counsel",
    ]
    return any(kw in pos_lower for kw in congress_keywords)


def _is_former_executive(covered_position: str) -> bool:
    """Check if a covered position indicates former executive branch service."""
    if not covered_position:
        return False
    pos_lower = covered_position.lower()
    exec_keywords = [
        "white house", "executive office", "department of",
        "secretary", "administrator", "commissioner",
        "director of", "assistant secretary",
    ]
    return any(kw in pos_lower for kw in exec_keywords)


def _extract_bill_references(text: str) -> list[str]:
    """Extract bill references like H.R. 1234, S. 5678 from text."""
    import re

    bills = []
    # Match patterns like H.R. 1234, S. 567, H.R.1234, S.567
    pattern = r'\b(?:H\.?\s*R\.?|S\.?|H\.?\s*J\.?\s*Res\.?|S\.?\s*J\.?\s*Res\.?)\s*(\d+)\b'
    for match in re.finditer(pattern, text, re.IGNORECASE):
        bills.append(match.group(0).strip())

    return bills
