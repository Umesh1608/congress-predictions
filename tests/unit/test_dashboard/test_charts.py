"""Tests for dashboard chart functions."""

from dashboard.charts import (
    model_performance_chart,
    sentiment_timeline_chart,
    signal_strength_chart,
    top_tickers_chart,
    trade_timeline_chart,
)


class TestTradeTimelineChart:
    def test_basic_chart(self):
        trades = [
            {"transaction_date": "2024-03-01", "transaction_type": "purchase",
             "amount_range_low": 15000, "ticker": "NVDA", "member_name": "Pelosi"},
            {"transaction_date": "2024-03-05", "transaction_type": "sale",
             "amount_range_low": 50000, "ticker": "AAPL", "member_name": "Tuberville"},
        ]
        fig = trade_timeline_chart(trades)
        assert fig is not None
        assert len(fig.data) == 2  # purchases + sales traces

    def test_empty_trades(self):
        fig = trade_timeline_chart([])
        assert fig is not None
        assert len(fig.data) == 0


class TestSentimentTimelineChart:
    def test_basic_chart(self):
        timeline = [
            {"date": "2024-03-01", "avg_score": 0.5, "count": 3},
            {"date": "2024-03-02", "avg_score": -0.2, "count": 1},
        ]
        fig = sentiment_timeline_chart(timeline, "Test Member")
        assert fig is not None
        assert len(fig.data) >= 1


class TestSignalStrengthChart:
    def test_basic_chart(self):
        signals = [
            {"signal_type": "trade_follow", "strength": 0.8},
            {"signal_type": "trade_follow", "strength": 0.6},
            {"signal_type": "anomaly_alert", "strength": 0.9},
        ]
        fig = signal_strength_chart(signals)
        assert fig is not None

    def test_empty_signals(self):
        fig = signal_strength_chart([])
        assert fig is not None


class TestModelPerformanceChart:
    def test_basic_chart(self):
        models = [
            {"model_name": "trade_predictor", "metrics": {"accuracy": 0.7, "f1": 0.65}},
            {"model_name": "return_predictor", "metrics": {"mae": 0.02, "r2": 0.4}},
        ]
        fig = model_performance_chart(models)
        assert fig is not None

    def test_empty_models(self):
        fig = model_performance_chart([])
        assert fig is not None


class TestTopTickersChart:
    def test_basic_chart(self):
        stats = {
            "top_tickers": [
                {"ticker": "NVDA", "count": 50},
                {"ticker": "AAPL", "count": 30},
            ]
        }
        fig = top_tickers_chart(stats)
        assert fig is not None

    def test_empty_stats(self):
        fig = top_tickers_chart({})
        assert fig is not None
