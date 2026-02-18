"""Tests for alert dispatch and matching logic."""

from unittest.mock import MagicMock

from src.signals.alerting import _matches_config, _within_rate_limit, _dispatch_counts


class TestAlertMatching:
    def _make_signal(self, **kwargs):
        signal = MagicMock()
        signal.signal_type = kwargs.get("signal_type", "trade_follow")
        signal.ticker = kwargs.get("ticker", "NVDA")
        signal.member_bioguide_id = kwargs.get("member_bioguide_id", "P000197")
        signal.strength = kwargs.get("strength", 0.8)
        return signal

    def _make_config(self, **kwargs):
        config = MagicMock()
        config.signal_types = kwargs.get("signal_types", [])
        config.min_strength = kwargs.get("min_strength", 0.5)
        config.tickers = kwargs.get("tickers", [])
        config.members = kwargs.get("members", [])
        return config

    def test_matches_all_empty_filters(self):
        signal = self._make_signal()
        config = self._make_config()
        assert _matches_config(signal, config) is True

    def test_matches_signal_type(self):
        signal = self._make_signal(signal_type="trade_follow")
        config = self._make_config(signal_types=["trade_follow", "anomaly_alert"])
        assert _matches_config(signal, config) is True

    def test_rejects_wrong_signal_type(self):
        signal = self._make_signal(signal_type="insider_cluster")
        config = self._make_config(signal_types=["trade_follow"])
        assert _matches_config(signal, config) is False

    def test_rejects_below_min_strength(self):
        signal = self._make_signal(strength=0.3)
        config = self._make_config(min_strength=0.5)
        assert _matches_config(signal, config) is False

    def test_matches_ticker_filter(self):
        signal = self._make_signal(ticker="NVDA")
        config = self._make_config(tickers=["NVDA", "AAPL"])
        assert _matches_config(signal, config) is True

    def test_rejects_wrong_ticker(self):
        signal = self._make_signal(ticker="MSFT")
        config = self._make_config(tickers=["NVDA", "AAPL"])
        assert _matches_config(signal, config) is False

    def test_matches_member_filter(self):
        signal = self._make_signal(member_bioguide_id="P000197")
        config = self._make_config(members=["P000197"])
        assert _matches_config(signal, config) is True


class TestRateLimiting:
    def test_within_limit(self):
        # Clear any previous state
        _dispatch_counts.clear()
        assert _within_rate_limit(9999) is True

    def test_tracks_dispatches(self):
        _dispatch_counts.clear()
        for _ in range(10):
            _within_rate_limit(8888)
        assert _within_rate_limit(8888) is False
