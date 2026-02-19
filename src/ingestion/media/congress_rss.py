"""Congress.gov RSS feed collector.

Fetches congressional activity updates from public RSS feeds.
No API key required. Uses feedparser library.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any

import feedparser

from src.ingestion.base import BaseCollector, RateLimiter

logger = logging.getLogger(__name__)

_rss_rate_limiter = RateLimiter(max_calls=2, period_seconds=1.0)

# Congress.gov RSS feed URLs
DEFAULT_FEEDS: dict[str, str] = {
    "house_floor": "https://www.congress.gov/rss/house-floor-today.xml",
    "senate_floor": "https://www.congress.gov/rss/senate-floor-today.xml",
    "most_viewed_bills": "https://www.congress.gov/rss/most-viewed-bills.xml",
}


class CongressRSSCollector(BaseCollector):
    """Collect congressional activity from Congress.gov RSS feeds.

    Fetches floor updates, bill activity, and committee news.
    Free, no API key required.
    """

    source_name = "congress_rss"
    rate_limiter = _rss_rate_limiter

    def __init__(self, feeds: dict[str, str] | None = None) -> None:
        super().__init__()
        self.feeds = feeds or DEFAULT_FEEDS

    async def collect(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for feed_name, feed_url in self.feeds.items():
            try:
                if self.rate_limiter:
                    await self.rate_limiter.acquire()
                response = await self.client.get(feed_url)
                response.raise_for_status()

                feed = feedparser.parse(response.text)
                for entry in feed.entries:
                    entry_dict = dict(entry)
                    entry_dict["_feed_name"] = feed_name
                    results.append(entry_dict)

            except Exception:
                logger.exception(
                    "[%s] Failed to fetch feed %s", self.source_name, feed_name
                )

        logger.info("[%s] Collected %d RSS entries", self.source_name, len(results))
        return results

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        title = raw.get("title", "")
        if not title:
            return None

        # Use guid or link as source_id
        source_id = raw.get("id") or raw.get("link", "")
        if not source_id:
            return None

        # Parse published date
        published_date = None
        pub_str = raw.get("published", "") or raw.get("updated", "")
        if pub_str:
            published_date = _parse_rss_date(pub_str)

        # Clean HTML from summary
        summary = raw.get("summary", "")
        content = _strip_html(summary)

        return {
            "source_type": "congress_rss",
            "source_id": source_id,
            "title": title,
            "content": content,
            "url": raw.get("link", ""),
            "published_date": published_date,
            "raw_metadata": {
                "feed_name": raw.get("_feed_name", ""),
                "tags": [tag.get("term", "") for tag in raw.get("tags", [])],
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
