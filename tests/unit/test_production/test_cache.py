"""Tests for the response cache."""

import pytest

from src.api.cache import ResponseCache, _make_cache_key, cached


class TestResponseCache:
    def test_set_and_get(self):
        cache = ResponseCache()
        cache.set("key1", {"data": 42})
        assert cache.get("key1", ttl=60) == {"data": 42}

    def test_expired_returns_none(self):
        cache = ResponseCache()
        cache.set("key1", "value")
        assert cache.get("key1", ttl=0) is None

    def test_missing_key_returns_none(self):
        cache = ResponseCache()
        assert cache.get("missing", ttl=60) is None

    def test_invalidate_by_prefix(self):
        cache = ResponseCache()
        cache.set("stats:a", 1)
        cache.set("stats:b", 2)
        cache.set("other:c", 3)
        removed = cache.invalidate("stats")
        assert removed == 2
        assert cache.get("stats:a", ttl=60) is None
        assert cache.get("other:c", ttl=60) == 3

    def test_clear(self):
        cache = ResponseCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a", ttl=60) is None
        assert cache.get("b", ttl=60) is None


class TestCacheKey:
    def test_deterministic(self):
        k1 = _make_cache_key("prefix", (1, 2), {"a": "b"})
        k2 = _make_cache_key("prefix", (1, 2), {"a": "b"})
        assert k1 == k2

    def test_different_args_different_keys(self):
        k1 = _make_cache_key("prefix", (1,), {})
        k2 = _make_cache_key("prefix", (2,), {})
        assert k1 != k2


class TestCachedDecorator:
    @pytest.mark.asyncio
    async def test_caches_result(self):
        call_count = 0

        @cached(ttl=60, key_prefix="test_fn_a")
        async def my_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = await my_func(5)
        result2 = await my_func(5)
        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Second call served from cache

    @pytest.mark.asyncio
    async def test_invalidate_cache(self):
        call_count = 0

        @cached(ttl=60, key_prefix="test_fn_b")
        async def my_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        await my_func(3)
        my_func.invalidate_cache()
        await my_func(3)
        assert call_count == 2  # Called again after invalidation
