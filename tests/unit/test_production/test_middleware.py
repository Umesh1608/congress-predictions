"""Tests for production middleware."""

from unittest.mock import patch

from src.api.middleware import _SlidingWindow, _is_valid_api_key


class TestSlidingWindow:
    def test_allows_within_limit(self):
        sw = _SlidingWindow()
        for _ in range(5):
            assert sw.is_allowed("client1", limit=5)

    def test_blocks_over_limit(self):
        sw = _SlidingWindow()
        for _ in range(10):
            sw.is_allowed("client1", limit=10)
        assert not sw.is_allowed("client1", limit=10)

    def test_separate_keys_independent(self):
        sw = _SlidingWindow()
        for _ in range(5):
            sw.is_allowed("client1", limit=5)
        # client2 should still be allowed
        assert sw.is_allowed("client2", limit=5)

    def test_expired_entries_pruned(self):
        sw = _SlidingWindow()
        # Fill with entries
        for _ in range(5):
            sw.is_allowed("client1", limit=5, window_seconds=0)
        # With window_seconds=0, all old entries are expired, so next should be allowed
        assert sw.is_allowed("client1", limit=5, window_seconds=0)


class TestApiKeyValidation:
    @patch("src.api.middleware.settings")
    def test_valid_key(self, mock_settings):
        mock_settings.api_keys = "key1,key2,key3"
        assert _is_valid_api_key("key2") is True

    @patch("src.api.middleware.settings")
    def test_invalid_key(self, mock_settings):
        mock_settings.api_keys = "key1,key2"
        assert _is_valid_api_key("bad_key") is False

    @patch("src.api.middleware.settings")
    def test_empty_config(self, mock_settings):
        mock_settings.api_keys = ""
        assert _is_valid_api_key("any_key") is False

    @patch("src.api.middleware.settings")
    def test_whitespace_handling(self, mock_settings):
        mock_settings.api_keys = " key1 , key2 "
        assert _is_valid_api_key("key1") is True
