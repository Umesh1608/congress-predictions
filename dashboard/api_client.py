"""API client for Streamlit dashboard — wraps FastAPI backend with caching."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
DEFAULT_TIMEOUT = 30.0


class _Cache:
    """Simple TTL cache for API responses."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, Any]] = {}

    def get(self, key: str, ttl: float = 60.0) -> Any | None:
        if key in self._data:
            ts, value = self._data[key]
            if time.time() - ts < ttl:
                return value
            del self._data[key]
        return None

    def set(self, key: str, value: Any) -> None:
        self._data[key] = (time.time(), value)

    def clear(self) -> None:
        self._data.clear()


_cache = _Cache()


class CongressAPI:
    """Synchronous HTTP client for the Congress Predictions API."""

    def __init__(self, base_url: str = API_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=DEFAULT_TIMEOUT)

    def _get(self, path: str, params: dict | None = None, cache_ttl: float = 0) -> Any:
        cache_key = f"{path}:{params}" if params else path
        if cache_ttl > 0:
            cached = _cache.get(cache_key, cache_ttl)
            if cached is not None:
                return cached

        resp = self.client.get(path, params=params)
        resp.raise_for_status()
        data = resp.json()

        if cache_ttl > 0:
            _cache.set(cache_key, data)
        return data

    # Members
    def get_members(self, **params: Any) -> list[dict]:
        return self._get("/members", params=params, cache_ttl=300)

    def get_member(self, bioguide_id: str) -> dict:
        return self._get(f"/members/{bioguide_id}", cache_ttl=300)

    # Trades
    def get_trades(self, **params: Any) -> list[dict]:
        return self._get("/trades", params=params)

    def get_trade_stats(self) -> dict:
        return self._get("/trades/stats", cache_ttl=60)

    def get_trade_context(self, trade_id: int) -> dict:
        return self._get(f"/trades/{trade_id}/legislative-context")

    # Predictions
    def get_predictions(self, **params: Any) -> list[dict]:
        return self._get("/predictions", params=params)

    def get_trade_predictions(self, trade_id: int) -> list[dict]:
        return self._get(f"/predictions/{trade_id}")

    def get_model_performance(self) -> list[dict]:
        return self._get("/predictions/model-performance", cache_ttl=120)

    def get_prediction_stats(self) -> dict:
        return self._get("/predictions/stats", cache_ttl=60)

    def get_leaderboard(self, limit: int = 20) -> list[dict]:
        return self._get("/predictions/leaderboard", params={"limit": limit}, cache_ttl=120)

    # Signals
    def get_signals(self, **params: Any) -> list[dict]:
        return self._get("/signals", params=params)

    def get_signal(self, signal_id: int) -> dict:
        return self._get(f"/signals/{signal_id}")

    def get_signal_stats(self) -> dict:
        return self._get("/signals/stats", cache_ttl=60)

    # Media
    def get_media(self, **params: Any) -> list[dict]:
        return self._get("/media", params=params)

    def get_media_stats(self) -> dict:
        return self._get("/media/stats", cache_ttl=60)

    def get_member_sentiment_timeline(self, bioguide_id: str, days: int = 90) -> dict:
        return self._get(
            f"/members/{bioguide_id}/sentiment-timeline",
            params={"days": days},
        )

    # Network
    def get_network_stats(self) -> dict:
        return self._get("/network/stats", cache_ttl=120)

    def get_member_network(self, bioguide_id: str, max_depth: int = 2) -> dict:
        return self._get(
            f"/network/member/{bioguide_id}",
            params={"max_depth": max_depth},
        )

    def get_member_paths(self, bioguide_id: str, ticker: str) -> dict:
        return self._get(f"/network/member/{bioguide_id}/paths-to/{ticker}")

    # Recommendations
    def get_recommendations(
        self,
        portfolio_size: float = 3000,
        top_n: int = 5,
        min_signal_strength: float = 0.3,
    ) -> dict:
        return self._get(
            "/recommendations",
            params={
                "portfolio_size": portfolio_size,
                "top_n": top_n,
                "min_signal_strength": min_signal_strength,
            },
            cache_ttl=60,
        )

    # Health
    def get_health(self) -> dict:
        return self._get("/health" if self.base_url.endswith("/v1") else "/../health")
