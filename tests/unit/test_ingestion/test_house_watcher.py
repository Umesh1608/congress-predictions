"""Tests for House Stock Watcher collector."""

from datetime import date
from decimal import Decimal

from src.ingestion.trades.house_watcher import HouseWatcherCollector


class TestHouseWatcherTransform:
    def setup_method(self):
        self.collector = HouseWatcherCollector()

    def test_transform_basic_purchase(self, sample_house_trade_raw):
        result = self.collector.transform(sample_house_trade_raw)

        assert result is not None
        assert result["member_name"] == "Hon. Nancy Pelosi"
        assert result["ticker"] == "NVDA"
        assert result["transaction_type"] == "purchase"
        assert result["transaction_date"] == date(2024, 1, 2)
        assert result["disclosure_date"] == date(2024, 1, 15)
        assert result["amount_range_low"] == Decimal("15001")
        assert result["amount_range_high"] == Decimal("50000")
        assert result["chamber"] == "house"
        assert result["source"] == "house_watcher"

    def test_transform_sale(self, sample_house_trade_raw):
        sample_house_trade_raw["type"] = "sale_full"
        result = self.collector.transform(sample_house_trade_raw)
        assert result["transaction_type"] == "sale_full"

    def test_transform_partial_sale(self, sample_house_trade_raw):
        sample_house_trade_raw["type"] = "sale_partial"
        result = self.collector.transform(sample_house_trade_raw)
        assert result["transaction_type"] == "sale_partial"

    def test_transform_spouse_owner(self, sample_house_trade_raw):
        sample_house_trade_raw["owner"] = "spouse"
        result = self.collector.transform(sample_house_trade_raw)
        assert result["filer_type"] == "spouse"

    def test_transform_dependent_owner(self, sample_house_trade_raw):
        sample_house_trade_raw["owner"] = "dependent child"
        result = self.collector.transform(sample_house_trade_raw)
        assert result["filer_type"] == "dependent"

    def test_transform_no_ticker(self, sample_house_trade_raw):
        sample_house_trade_raw["ticker"] = "--"
        result = self.collector.transform(sample_house_trade_raw)
        assert result is not None
        assert result["ticker"] is None

    def test_transform_bad_date_returns_none(self, sample_house_trade_raw):
        sample_house_trade_raw["transaction_date"] = "invalid-date"
        result = self.collector.transform(sample_house_trade_raw)
        assert result is None

    def test_transform_empty_date_returns_none(self, sample_house_trade_raw):
        sample_house_trade_raw["transaction_date"] = ""
        result = self.collector.transform(sample_house_trade_raw)
        assert result is None
