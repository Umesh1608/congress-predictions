"""Tests for Senate Stock Watcher collector."""

from datetime import date
from decimal import Decimal

from src.ingestion.trades.senate_watcher import SenateWatcherCollector


class TestSenateWatcherTransform:
    def setup_method(self):
        self.collector = SenateWatcherCollector()

    def test_transform_basic_purchase(self, sample_senate_trade_raw):
        result = self.collector.transform(sample_senate_trade_raw)

        assert result is not None
        assert result["member_name"] == "Tommy Tuberville"
        assert result["ticker"] == "MSFT"
        assert result["transaction_type"] == "purchase"
        assert result["transaction_date"] == date(2024, 2, 10)
        assert result["disclosure_date"] == date(2024, 3, 1)
        assert result["amount_range_low"] == Decimal("1001")
        assert result["amount_range_high"] == Decimal("15000")
        assert result["chamber"] == "senate"
        assert result["filer_type"] == "member"

    def test_transform_spouse(self, sample_senate_trade_raw):
        sample_senate_trade_raw["owner"] = "Spouse"
        result = self.collector.transform(sample_senate_trade_raw)
        assert result["filer_type"] == "spouse"

    def test_transform_no_date(self, sample_senate_trade_raw):
        sample_senate_trade_raw["transaction_date"] = ""
        result = self.collector.transform(sample_senate_trade_raw)
        assert result is None
