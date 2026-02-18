"""NewsData.io collector.

Fetches news articles related to congressional activity from NewsData.io.
Free tier: 2000 articles/day.
API docs: https://newsdata.io/documentation
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from src.config import settings
from src.ingestion.base import BaseCollector, RateLimiter

logger = logging.getLogger(__name__)

NEWSDATA_API_BASE = "https://newsdata.io/api/1"

_newsdata_rate_limiter = RateLimiter(max_calls=1, period_seconds=1.0)

DEFAULT_QUERY = "congress stock trading OR congressional insider trading OR STOCK Act"


class NewsDataCollector(BaseCollector):
    """Collect news articles from NewsData.io API.

    Searches for political and business news related to congressional trading.
    Skips entirely if newsdata_api_key is not configured.
    """

    source_name = "newsdata"
    rate_limiter = _newsdata_rate_limiter

    def __init__(
        self,
        api_key: str | None = None,
        query: str | None = None,
        language: str = "en",
        country: str = "us",
    ) -> None:
        super().__init__()
        self.api_key = api_key or settings.newsdata_api_key
        self.query = query or DEFAULT_QUERY
        self.language = language
        self.country = country

    async def collect(self) -> list[dict[str, Any]]:
        if not self.api_key:
            logger.info("[%s] No NewsData API key configured, skipping", self.source_name)
            return []

        results: list[dict[str, Any]] = []

        data = await self.fetch_json(
            f"{NEWSDATA_API_BASE}/news",
            params={
                "apikey": self.api_key,
                "q": self.query,
                "language": self.language,
                "country": self.country,
                "category": "politics,business",
            },
        )
        if not data:
            return []

        for article in data.get("results", []):
            results.append(article)

        logger.info("[%s] Collected %d articles", self.source_name, len(results))
        return results

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        title = raw.get("title", "")
        if not title:
            return None

        # Use article_id or link as source_id
        source_id = raw.get("article_id") or raw.get("link", "")
        if not source_id:
            return None

        # Parse published date
        published_date = None
        pub_str = raw.get("pubDate", "")
        if pub_str:
            try:
                published_date = datetime.fromisoformat(
                    pub_str.replace("Z", "+00:00")
                ).date()
            except (ValueError, TypeError):
                # Try alternate format "YYYY-MM-DD HH:MM:SS"
                try:
                    published_date = datetime.strptime(
                        pub_str[:19], "%Y-%m-%d %H:%M:%S"
                    ).date()
                except (ValueError, IndexError):
                    pass

        return {
            "source_type": "newsdata",
            "source_id": source_id,
            "title": title,
            "content": raw.get("description", "") or raw.get("content", ""),
            "url": raw.get("link", ""),
            "author": ", ".join(raw.get("creator", []) or []),
            "published_date": published_date,
            "raw_metadata": {
                "source_id": raw.get("source_id", ""),
                "source_name": raw.get("source_name", ""),
                "categories": raw.get("category", []),
                "keywords": raw.get("keywords", []),
                "country": raw.get("country", []),
                "image_url": raw.get("image_url", ""),
            },
        }
