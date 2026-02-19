"""GovInfo hearing transcript collector.

Uses the GovInfo API to fetch full committee hearing transcripts.
API docs: https://api.govinfo.gov/docs/

Collections used:
- CHRG: Congressional Hearings (full text)
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any

from src.config import settings
from src.ingestion.base import BaseCollector, RateLimiter

logger = logging.getLogger(__name__)

GOVINFO_API_BASE = "https://api.govinfo.gov"

_govinfo_rate_limiter = RateLimiter(max_calls=1, period_seconds=1.0)


class GovInfoHearingCollector(BaseCollector):
    """Collect committee hearing transcripts from the GovInfo API.

    Fetches hearing packages from the CHRG collection and retrieves
    their full text content (HTML or plain text).
    """

    source_name = "govinfo_hearings"
    rate_limiter = _govinfo_rate_limiter

    def __init__(
        self,
        api_key: str | None = None,
        congress: int = 119,
        offset: int = 0,
        page_size: int = 100,
    ) -> None:
        super().__init__()
        self.api_key = api_key or settings.govinfo_api_key
        self.congress = congress
        self.offset = offset
        self.page_size = page_size
        # Congress N starts in odd year: 2025 for 119th, 2023 for 118th, etc.
        self.congress_start_year = 2025 - 2 * (119 - congress)

    async def collect(self) -> list[dict[str, Any]]:
        if not self.api_key:
            logger.info("[%s] No GovInfo API key configured, skipping", self.source_name)
            return []

        results: list[dict[str, Any]] = []
        # Use the collections endpoint: /collections/CHRG/{startDate}
        start_date = f"{self.congress_start_year}-01-01T00:00:00Z"
        params = {
            "api_key": self.api_key,
            "offset": self.offset,
            "pageSize": self.page_size,
        }

        data = await self.fetch_json(
            f"{GOVINFO_API_BASE}/collections/CHRG/{start_date}",
            params=params,
        )
        if not data:
            return []

        packages = data.get("packages", [])
        for pkg in packages:
            package_id = pkg.get("packageId", "")
            if not package_id:
                continue

            # Fetch the package summary for metadata
            summary = await self.fetch_json(
                f"{GOVINFO_API_BASE}/packages/{package_id}/summary",
                params={"api_key": self.api_key},
            )
            if summary:
                pkg["_summary"] = summary

            # Fetch the full text (htm format)
            try:
                text_data = await self.fetch_json(
                    f"{GOVINFO_API_BASE}/packages/{package_id}/htm",
                    params={"api_key": self.api_key},
                )
                if text_data:
                    pkg["_full_text"] = text_data
            except Exception:
                logger.debug("No HTM content for %s", package_id)

            results.append(pkg)

        logger.info("[%s] Fetched %d hearing packages", self.source_name, len(results))
        return results

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        package_id = raw.get("packageId", "")
        if not package_id:
            return None

        title = raw.get("title", "")
        if not title:
            return None

        # Extract full text from HTM content or summary
        content = ""
        full_text = raw.get("_full_text")
        if isinstance(full_text, str):
            content = _strip_html(full_text)
        elif isinstance(full_text, dict):
            content = _strip_html(full_text.get("body", ""))

        summary = raw.get("_summary", {})

        # Parse date
        published_date = None
        date_str = raw.get("dateIssued") or summary.get("dateIssued", "")
        if date_str:
            published_date = _parse_date(date_str)

        # Extract committee from title or summary
        committee = summary.get("committee", "")
        congress_num = summary.get("congress", self.congress)

        return {
            "source_type": "hearing_transcript",
            "source_id": package_id,
            "title": title,
            "content": content,
            "summary": summary.get("description", ""),
            "url": f"https://www.govinfo.gov/app/details/{package_id}",
            "published_date": published_date,
            "raw_metadata": {
                "package_id": package_id,
                "committee": committee,
                "congress": congress_num,
                "collection": "CHRG",
                "category": raw.get("category", ""),
            },
        }


def _strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _parse_date(date_str: str) -> date | None:
    """Parse various date formats from GovInfo."""
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(date_str[:19], fmt).date()
        except (ValueError, IndexError):
            continue
    return None
