"""In-memory response cache with TTL for expensive API endpoints."""

from __future__ import annotations

import functools
import hashlib
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ResponseCache:
    """Simple in-memory TTL cache for API responses."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str, ttl: int) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if time.monotonic() - stored_at > ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.monotonic(), value)

    def invalidate(self, prefix: str) -> int:
        """Remove all entries matching a key prefix. Returns count removed."""
        to_remove = [k for k in self._store if k.startswith(prefix)]
        for k in to_remove:
            del self._store[k]
        return len(to_remove)

    def clear(self) -> None:
        self._store.clear()


# Module-level singleton
_cache = ResponseCache()


def get_cache() -> ResponseCache:
    return _cache


def _make_cache_key(prefix: str, args: tuple, kwargs: dict) -> str:
    """Build a deterministic cache key from function arguments."""
    raw = f"{prefix}:{args!r}:{sorted(kwargs.items())!r}"
    return f"{prefix}:{hashlib.md5(raw.encode()).hexdigest()}"


def cached(ttl: int = 60, key_prefix: str = "") -> Callable:
    """Decorator that caches an async endpoint's return value.

    Usage::

        @cached(ttl=60, key_prefix="trades_stats")
        async def get_trades_stats(db: AsyncSession):
            ...
    """

    def decorator(func: Callable) -> Callable:
        prefix = key_prefix or func.__qualname__

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_key = _make_cache_key(prefix, args, kwargs)
            hit = _cache.get(cache_key, ttl)
            if hit is not None:
                logger.debug("Cache hit: %s", cache_key)
                return hit
            result = await func(*args, **kwargs)
            _cache.set(cache_key, result)
            return result

        # Expose invalidation helper on the wrapper
        wrapper.cache_prefix = prefix  # type: ignore[attr-defined]
        wrapper.invalidate_cache = lambda: _cache.invalidate(prefix)  # type: ignore[attr-defined]

        return wrapper

    return decorator
