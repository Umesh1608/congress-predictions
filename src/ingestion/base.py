from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, max_calls: int, period_seconds: float) -> None:
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls: list[float] = []

    async def acquire(self) -> None:
        now = time.monotonic()
        self.calls = [t for t in self.calls if now - t < self.period]
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0])
            logger.debug("Rate limit hit, sleeping %.1fs", sleep_time)
            await asyncio.sleep(sleep_time)
        self.calls.append(time.monotonic())


class BaseCollector(ABC):
    """Abstract base class for all data collectors.

    Subclasses implement collect() to fetch raw data and transform() to
    convert each raw record into a validated form ready for database upsert.
    """

    source_name: str = "unknown"
    rate_limiter: RateLimiter | None = None
    max_retries: int = 3
    retry_delay: float = 2.0

    def __init__(self) -> None:
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        await self.client.aclose()

    async def fetch_json(self, url: str, **kwargs: Any) -> Any:
        """Fetch JSON from a URL with rate limiting and retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.rate_limiter:
                    await self.rate_limiter.acquire()
                response = await self.client.get(url, **kwargs)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = self.retry_delay * attempt * 2
                    logger.warning("429 rate limited on %s, waiting %.1fs", url, wait)
                    await asyncio.sleep(wait)
                    continue
                if attempt == self.max_retries:
                    logger.error("HTTP %d on %s after %d attempts", e.response.status_code, url, attempt)
                    raise
                await asyncio.sleep(self.retry_delay * attempt)
            except httpx.RequestError as e:
                if attempt == self.max_retries:
                    logger.error("Request failed for %s after %d attempts: %s", url, attempt, e)
                    raise
                await asyncio.sleep(self.retry_delay * attempt)
        return None

    @abstractmethod
    async def collect(self) -> list[dict[str, Any]]:
        """Fetch raw records from the data source."""
        ...

    @abstractmethod
    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        """Transform a raw record into a dict ready for DB upsert.

        Return None to skip a record (e.g., if it fails validation).
        """
        ...

    async def run(self) -> list[dict[str, Any]]:
        """Execute the full collect -> transform pipeline."""
        logger.info("[%s] Starting collection", self.source_name)
        raw_records = await self.collect()
        logger.info("[%s] Fetched %d raw records", self.source_name, len(raw_records))

        transformed = []
        skipped = 0
        for raw in raw_records:
            try:
                result = self.transform(raw)
                if result is not None:
                    transformed.append(result)
                else:
                    skipped += 1
            except Exception:
                logger.exception("[%s] Failed to transform record: %s", self.source_name, raw)
                skipped += 1

        logger.info(
            "[%s] Transformed %d records, skipped %d",
            self.source_name, len(transformed), skipped,
        )
        return transformed
