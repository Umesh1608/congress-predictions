"""Tests for GitHub Senate CSV collector."""

from datetime import date
from decimal import Decimal

from src.ingestion.trades.github_senate import GitHubSenateCollector


class TestGitHubSenateTransform:
    def setup_method(self):
        self.collector = GitHubSenateCollector()

    def test_basic_transform(self):
        raw = {
            "report-id": "abc-123",
            "filer": "Hoeven, John",
            "filed-date": "05/03/2017",
            "owner": "Self",
            "asset-name": "International Business Machines Corporation",
            "ticker": "IBM",
            "transaction-type": "Purchase",
            "transaction-date": "04/24/2017",
            "amount": "$50,001 - $100,000",
            "asset-type": "Stock",
            "type": "Purchase",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["member_name"] == "Hoeven, John"
        assert result["ticker"] == "IBM"
        assert result["transaction_type"] == "purchase"
        assert result["transaction_date"] == date(2017, 4, 24)
        assert result["disclosure_date"] == date(2017, 5, 3)
        assert result["amount_range_low"] == Decimal("50001")
        assert result["amount_range_high"] == Decimal("100000")
        assert result["chamber"] == "senate"
        assert result["source"] == "github_senate"
        assert result["filer_type"] == "member"

    def test_sale_type(self):
        raw = {
            "filer": "Wyden, Ron",
            "ticker": "BYND",
            "transaction-type": "Sale (Full)",
            "transaction-date": "11/10/2020",
            "owner": "Spouse",
            "asset-name": "Beyond Meat",
            "amount": "$50,001 - $100,000",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["transaction_type"] == "sale_full"
        assert result["filer_type"] == "spouse"

    def test_partial_sale(self):
        raw = {
            "filer": "Test Senator",
            "ticker": "AAPL",
            "transaction-type": "Sale (Partial)",
            "transaction-date": "2023-01-15",
            "owner": "Self",
            "asset-name": "Apple Inc.",
            "amount": "$1,001 - $15,000",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["transaction_type"] == "sale_partial"

    def test_no_ticker(self):
        raw = {
            "filer": "Test Senator",
            "ticker": "--",
            "transaction-date": "2023-01-15",
            "owner": "Self",
            "asset-name": "Municipal Bond",
            "amount": "$1,001 - $15,000",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["ticker"] is None

    def test_long_ticker_nullified(self):
        raw = {
            "filer": "Test Senator",
            "ticker": "Wynn Resorts Ltd. (stock) NASDAQ",
            "transaction-date": "2023-01-15",
            "owner": "Self",
            "asset-name": "Wynn Resorts",
            "amount": "$1,001 - $15,000",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["ticker"] is None

    def test_no_transaction_date_returns_none(self):
        raw = {
            "filer": "Test Senator",
            "ticker": "AAPL",
            "transaction-date": "",
            "owner": "Self",
            "asset-name": "Apple",
        }
        result = self.collector.transform(raw)
        assert result is None

    def test_no_filer_returns_none(self):
        raw = {
            "filer": "",
            "ticker": "AAPL",
            "transaction-date": "2023-01-15",
            "owner": "Self",
            "asset-name": "Apple",
        }
        result = self.collector.transform(raw)
        assert result is None

    def test_dependent_owner(self):
        raw = {
            "filer": "Test Senator",
            "ticker": "MSFT",
            "transaction-date": "2023-01-15",
            "owner": "Dependent Child",
            "asset-name": "Microsoft",
            "amount": "$1,001 - $15,000",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["filer_type"] == "dependent"

    def test_joint_owner(self):
        raw = {
            "filer": "Test Senator",
            "ticker": "GOOG",
            "transaction-date": "2023-01-15",
            "owner": "Joint",
            "asset-name": "Alphabet",
            "amount": "$15,001 - $50,000",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["filer_type"] == "joint"

    def test_date_format_slash(self):
        raw = {
            "filer": "Test Senator",
            "ticker": "AAPL",
            "transaction-date": "01/15/2023",
            "owner": "Self",
            "asset-name": "Apple",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["transaction_date"] == date(2023, 1, 15)

    def test_raw_data_preserved(self):
        raw = {
            "filer": "Test Senator",
            "ticker": "AAPL",
            "transaction-date": "2023-01-15",
            "owner": "Self",
            "asset-name": "Apple",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["raw_data"] == raw
