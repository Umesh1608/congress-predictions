"""X/Twitter collector (stub — requires paid API).

Full implementation ready to activate by adding TWITTER_BEARER_TOKEN.
Uses Twitter API v2 user timeline endpoint.

When no API key is configured, collect() returns [] immediately.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from src.config import settings
from src.ingestion.base import BaseCollector, RateLimiter

logger = logging.getLogger(__name__)

TWITTER_API_BASE = "https://api.twitter.com/2"

_twitter_rate_limiter = RateLimiter(max_calls=1, period_seconds=1.0)

# Congress member X/Twitter accounts
# Format: {twitter_user_id: (screen_name, bioguide_id)}
# Populate when API key is available
CONGRESS_TWITTER_ACCOUNTS: dict[str, tuple[str, str]] = {
    "18aborar26495649": ("SpeakerPelosi", "P000197"),
    "23818800": ("SenTuberville", "T000278"),
    "951649396770480128": ("SenTedCruz", "C001098"),
}


class TwitterCollector(BaseCollector):
    """Collect tweets from congress member accounts.

    Requires TWITTER_BEARER_TOKEN to be configured.
    When not configured, gracefully skips (returns empty list).
    """

    source_name = "twitter"
    rate_limiter = _twitter_rate_limiter

    def __init__(
        self,
        bearer_token: str | None = None,
        accounts: dict[str, tuple[str, str]] | None = None,
        max_tweets_per_user: int = 20,
    ) -> None:
        super().__init__()
        self.bearer_token = bearer_token or settings.twitter_bearer_token
        self.accounts = accounts or CONGRESS_TWITTER_ACCOUNTS
        self.max_tweets_per_user = max_tweets_per_user

        if self.bearer_token:
            self.client.headers["Authorization"] = f"Bearer {self.bearer_token}"

    async def collect(self) -> list[dict[str, Any]]:
        if not self.bearer_token:
            logger.info(
                "[%s] No Twitter bearer token configured, skipping. "
                "Set TWITTER_BEARER_TOKEN to enable.",
                self.source_name,
            )
            return []

        results: list[dict[str, Any]] = []

        for user_id, (screen_name, bioguide_id) in self.accounts.items():
            try:
                data = await self.fetch_json(
                    f"{TWITTER_API_BASE}/users/{user_id}/tweets",
                    params={
                        "max_results": self.max_tweets_per_user,
                        "tweet.fields": "created_at,public_metrics,author_id",
                        "expansions": "author_id",
                    },
                )
                if not data:
                    continue

                for tweet in data.get("data", []):
                    tweet["_screen_name"] = screen_name
                    tweet["_bioguide_id"] = bioguide_id
                    results.append(tweet)

            except Exception:
                logger.warning(
                    "[%s] Failed to fetch tweets for @%s",
                    self.source_name,
                    screen_name,
                )

        logger.info("[%s] Collected %d tweets", self.source_name, len(results))
        return results

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        tweet_id = raw.get("id", "")
        if not tweet_id:
            return None

        text = raw.get("text", "")
        if not text:
            return None

        # Parse date
        published_date = None
        created_at = raw.get("created_at", "")
        if created_at:
            try:
                published_date = datetime.fromisoformat(
                    created_at.replace("Z", "+00:00")
                ).date()
            except (ValueError, TypeError):
                pass

        metrics = raw.get("public_metrics", {})
        bioguide_id = raw.get("_bioguide_id", "")

        return {
            "source_type": "twitter",
            "source_id": tweet_id,
            "title": text[:100] + ("..." if len(text) > 100 else ""),
            "content": text,
            "author": raw.get("_screen_name", ""),
            "published_date": published_date,
            "member_bioguide_ids": [bioguide_id] if bioguide_id else [],
            "raw_metadata": {
                "tweet_id": tweet_id,
                "screen_name": raw.get("_screen_name", ""),
                "bioguide_id": bioguide_id,
                "retweet_count": metrics.get("retweet_count", 0),
                "like_count": metrics.get("like_count", 0),
                "reply_count": metrics.get("reply_count", 0),
                "quote_count": metrics.get("quote_count", 0),
            },
        }
