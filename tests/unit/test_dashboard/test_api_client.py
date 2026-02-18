"""Tests for dashboard API client."""

from dashboard.api_client import CongressAPI, _Cache


class TestCache:
    def test_set_and_get(self):
        cache = _Cache()
        cache.set("key", "value")
        assert cache.get("key", ttl=60) == "value"

    def test_expired_returns_none(self):
        cache = _Cache()
        cache.set("key", "value")
        # Request with 0 TTL means it's always expired
        assert cache.get("key", ttl=0) is None

    def test_clear(self):
        cache = _Cache()
        cache.set("key", "value")
        cache.clear()
        assert cache.get("key", ttl=60) is None

    def test_missing_key(self):
        cache = _Cache()
        assert cache.get("nonexistent", ttl=60) is None


class TestCongressAPI:
    def test_default_base_url(self):
        api = CongressAPI()
        assert "localhost:8000" in api.base_url

    def test_custom_base_url(self):
        api = CongressAPI(base_url="http://myserver:9000/api/v1")
        assert api.base_url == "http://myserver:9000/api/v1"

    def test_trailing_slash_stripped(self):
        api = CongressAPI(base_url="http://localhost:8000/api/v1/")
        assert not api.base_url.endswith("/")
