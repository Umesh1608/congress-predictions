"""Tests for signal scoring logic."""

from src.signals.scorer import score_signal


class TestSignalScorer:
    def test_basic_scoring(self):
        data = {
            "signal_type": "trade_follow",
            "confidence": 0.8,
            "evidence": {"trade_id": 1, "prediction_id": 2},
        }
        result = score_signal(data)
        assert 0 <= result["strength"] <= 1.0
        assert result["confidence"] == 0.8

    def test_freshness_bonus(self):
        fresh = {
            "signal_type": "trade_follow",
            "confidence": 0.5,
            "disclosure_lag_days": 3,
            "evidence": {"trade_id": 1},
        }
        stale = {
            "signal_type": "trade_follow",
            "confidence": 0.5,
            "disclosure_lag_days": 30,
            "evidence": {"trade_id": 1},
        }
        fresh_score = score_signal(fresh)
        stale_score = score_signal(stale)
        assert fresh_score["strength"] > stale_score["strength"]

    def test_lag_penalty(self):
        no_lag = {
            "signal_type": "trade_follow",
            "confidence": 0.8,
            "evidence": {"trade_id": 1},
        }
        high_lag = {
            "signal_type": "trade_follow",
            "confidence": 0.8,
            "disclosure_lag_days": 40,
            "evidence": {"trade_id": 1},
        }
        no_lag_score = score_signal(no_lag)
        lag_score = score_signal(high_lag)
        assert lag_score["strength"] < no_lag_score["strength"]

    def test_corroboration_bonus(self):
        single = {
            "signal_type": "trade_follow",
            "confidence": 0.5,
            "evidence": {"trade_id": 1},
        }
        multi = {
            "signal_type": "trade_follow",
            "confidence": 0.5,
            "evidence": {
                "trade_id": 1,
                "prediction_id": 2,
                "avg_sentiment_30d": 0.3,
            },
        }
        single_score = score_signal(single)
        multi_score = score_signal(multi)
        assert multi_score["strength"] > single_score["strength"]

    def test_cap_at_one(self):
        data = {
            "signal_type": "insider_cluster",
            "confidence": 0.99,
            "evidence": {
                "trade_id": 1,
                "prediction_id": 2,
                "anomaly_score": 0.9,
                "avg_sentiment_30d": 0.5,
                "cluster_size": 10,
            },
        }
        result = score_signal(data)
        assert result["strength"] <= 1.0

    def test_cluster_size_bonus(self):
        small = {
            "signal_type": "insider_cluster",
            "confidence": 0.5,
            "evidence": {"cluster_size": 2},
        }
        large = {
            "signal_type": "insider_cluster",
            "confidence": 0.5,
            "evidence": {"cluster_size": 5},
        }
        small_score = score_signal(small)
        large_score = score_signal(large)
        assert large_score["strength"] > small_score["strength"]

    def test_zero_confidence(self):
        data = {
            "signal_type": "trade_follow",
            "confidence": 0,
            "evidence": {},
        }
        result = score_signal(data)
        assert result["confidence"] == 0
        assert result["strength"] >= 0
