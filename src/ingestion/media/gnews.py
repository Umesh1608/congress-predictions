"""GNews API collector.

Fetches news articles related to congressional trading from GNews.io.
Free tier: 100 requests/day.
API docs: https://gnews.io/docs/v4
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from src.config import settings
from src.ingestion.base import BaseCollector, RateLimiter

logger = logging.getLogger(__name__)

GNEWS_API_BASE = "https://gnews.io/api/v4"

_gnews_rate_limiter = RateLimiter(max_calls=1, period_seconds=1.0)

# Default search queries for congressional trading news
DEFAULT_QUERIES = [
    "congress stock trading",
    "congressional insider trading",
    "senator stock trade",
    "STOCK Act congress",
]


class GNewsCollector(BaseCollector):
    """Collect news articles from GNews API.

    Searches for congressional trading-related articles.
    Skips entirely if gnews_api_key is not configured.
    """

    source_name = "gnews"
    rate_limiter = _gnews_rate_limiter

    def __init__(
        self,
        api_key: str | None = None,
        queries: list[str] | None = None,
        max_articles: int = 10,
        language: str = "en",
        country: str = "us",
    ) -> None:
        super().__init__()
        self.api_key = api_key or settings.gnews_api_key
        self.queries = queries or DEFAULT_QUERIES
        self.max_articles = max_articles
        self.language = language
        self.country = country

    async def collect(self) -> list[dict[str, Any]]:
        if not self.api_key:
            logger.info("[%s] No GNews API key configured, skipping", self.source_name)
            return []

        results: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        for query in self.queries:
            data = await self.fetch_json(
                f"{GNEWS_API_BASE}/search",
                params={
                    "q": query,
                    "token": self.api_key,
                    "lang": self.language,
                    "country": self.country,
                    "max": self.max_articles,
                },
            )
            if not data:
                continue

            for article in data.get("articles", []):
                url = article.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append(article)

        logger.info("[%s] Collected %d articles", self.source_name, len(results))
        return results

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        title = raw.get("title", "")
        if not title:
            return None

        url = raw.get("url", "")
        if not url:
            return None

        # Parse published date
        published_date = None
        pub_str = raw.get("publishedAt", "")
        if pub_str:
            try:
                published_date = datetime.fromisoformat(
                    pub_str.replace("Z", "+00:00")
                ).date()
            except (ValueError, TypeError):
                pass

        source_info = raw.get("source", {})

        return {
            "source_type": "gnews",
            "source_id": url,
            "title": title,
            "content": raw.get("description", "") or raw.get("content", ""),
            "url": url,
            "author": source_info.get("name", ""),
            "published_date": published_date,
            "raw_metadata": {
                "source_name": source_info.get("name", ""),
                "source_url": source_info.get("url", ""),
                "image": raw.get("image", ""),
            },
        }
