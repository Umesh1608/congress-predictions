"""Member press release RSS collector.

Fetches press releases from congress member websites via RSS feeds.
Uses feedparser + beautifulsoup4 (both already in project dependencies).
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any

import feedparser

from src.ingestion.base import BaseCollector, RateLimiter

logger = logging.getLogger(__name__)

_press_rate_limiter = RateLimiter(max_calls=1, period_seconds=2.0)

# Curated RSS feeds for active congressional traders
# Format: {bioguide_id: (member_name, rss_url)}
# These are members known for high trading activity
MEMBER_RSS_FEEDS: dict[str, tuple[str, str]] = {
    "H000601": (
        "Bill Hagerty",
        "https://www.hagerty.senate.gov/feed/",
    ),
    "R000618": (
        "Pete Ricketts",
        "https://www.ricketts.senate.gov/feed/",
    ),
}


class PressReleaseCollector(BaseCollector):
    """Collect press releases from congress member website RSS feeds.

    Maintains a curated dictionary of member RSS feed URLs.
    New feeds can be added by updating MEMBER_RSS_FEEDS.
    """

    source_name = "press_release"
    rate_limiter = _press_rate_limiter

    def __init__(
        self,
        member_feeds: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self.member_feeds = member_feeds or MEMBER_RSS_FEEDS

    async def collect(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for bioguide_id, (member_name, feed_url) in self.member_feeds.items():
            try:
                if self.rate_limiter:
                    await self.rate_limiter.acquire()
                response = await self.client.get(feed_url)
                response.raise_for_status()

                feed = feedparser.parse(response.text)
                for entry in feed.entries:
                    entry_dict = dict(entry)
                    entry_dict["_bioguide_id"] = bioguide_id
                    entry_dict["_member_name"] = member_name
                    results.append(entry_dict)

            except Exception:
                logger.warning(
                    "[%s] Failed to fetch press releases for %s (%s)",
                    self.source_name,
                    member_name,
                    bioguide_id,
                )

        logger.info("[%s] Collected %d press release entries", self.source_name, len(results))
        return results

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        title = raw.get("title", "")
        if not title:
            return None

        source_id = raw.get("id") or raw.get("link", "")
        if not source_id:
            return None

        bioguide_id = raw.get("_bioguide_id", "")

        # Parse date
        published_date = None
        pub_str = raw.get("published", "") or raw.get("updated", "")
        if pub_str:
            published_date = _parse_rss_date(pub_str)

        # Extract and clean content
        content = ""
        if raw.get("content"):
            content = raw["content"][0].get("value", "") if raw["content"] else ""
        elif raw.get("summary"):
            content = raw["summary"]
        content = _strip_html(content)

        return {
            "source_type": "press_release",
            "source_id": source_id,
            "title": title,
            "content": content,
            "url": raw.get("link", ""),
            "author": raw.get("_member_name", ""),
            "published_date": published_date,
            "member_bioguide_ids": [bioguide_id] if bioguide_id else [],
            "raw_metadata": {
                "bioguide_id": bioguide_id,
                "member_name": raw.get("_member_name", ""),
            },
        }


def _strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"&[a-zA-Z]+;", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _parse_rss_date(date_str: str) -> date | None:
    """Parse various RSS date formats."""
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None
