"""Tests for entity resolution / name normalization."""

from src.processing.normalizer import (
    normalize_company_name,
    match_name_to_ticker,
    MANUAL_OVERRIDES,
)


class TestNormalizeCompanyName:
    def test_strips_corporate_suffixes(self):
        assert normalize_company_name("Microsoft Corp.") == "microsoft"
        assert normalize_company_name("Apple Inc.") == "apple"
        assert normalize_company_name("Tesla, Inc.") == "tesla"

    def test_strips_common_suffixes(self):
        name = normalize_company_name("Alphabet Holdings LLC")
        assert "llc" not in name
        assert "holdings" not in name

    def test_handles_empty_string(self):
        assert normalize_company_name("") == ""

    def test_handles_none_gracefully(self):
        assert normalize_company_name(None) == ""

    def test_normalizes_whitespace(self):
        result = normalize_company_name("  Some   Company   Inc.  ")
        assert "  " not in result

    def test_removes_punctuation(self):
        result = normalize_company_name("Johnson & Johnson")
        # & gets removed, words split
        assert "johnson" in result


class TestMatchNameToTicker:
    def setup_method(self):
        """Build a sample ticker lookup."""
        self.known_tickers = {
            "nvidia": "NVDA",
            "microsoft": "MSFT",
            "apple": "AAPL",
            "tesla": "TSLA",
            "pfizer": "PFE",
            "boeing": "BA",
            "jpmorgan chase": "JPM",
            "general electric": "GE",
        }

    def test_manual_override_google(self):
        ticker, confidence, method = match_name_to_ticker("Google LLC", self.known_tickers)
        assert ticker == "GOOGL"
        assert method == "manual"
        assert confidence >= 0.95

    def test_manual_override_amazon(self):
        ticker, confidence, method = match_name_to_ticker("Amazon.com Inc.", self.known_tickers)
        assert ticker == "AMZN"
        assert method == "manual"

    def test_manual_override_meta(self):
        ticker, confidence, method = match_name_to_ticker("Meta Platforms", self.known_tickers)
        assert ticker == "META"
        assert method == "manual"

    def test_exact_match(self):
        ticker, confidence, method = match_name_to_ticker("NVIDIA Corp.", self.known_tickers)
        assert ticker == "NVDA"
        # Could be exact or manual since nvidia is in overrides
        assert confidence >= 0.9

    def test_fuzzy_match_close_name(self):
        ticker, confidence, method = match_name_to_ticker(
            "Pfizer Pharmaceuticals", self.known_tickers
        )
        # Should match Pfizer either via known_tickers or manual
        assert ticker == "PFE"
        assert confidence >= 0.8

    def test_no_match_returns_none(self):
        ticker, confidence, method = match_name_to_ticker(
            "Some Totally Unknown Company XYZ", self.known_tickers
        )
        assert ticker is None
        assert confidence == 0.0
        assert method == "none"

    def test_empty_name_returns_none(self):
        ticker, confidence, method = match_name_to_ticker("", self.known_tickers)
        assert ticker is None

    def test_manual_overrides_have_common_companies(self):
        """Verify manual overrides include key companies."""
        assert "google" in MANUAL_OVERRIDES
        assert "amazon" in MANUAL_OVERRIDES
        assert "microsoft" in MANUAL_OVERRIDES
        assert "apple" in MANUAL_OVERRIDES
        assert "boeing" in MANUAL_OVERRIDES

    def test_defense_companies_in_overrides(self):
        """Defense sector is important for congressional trading."""
        assert "lockheed martin" in MANUAL_OVERRIDES
        assert "raytheon" in MANUAL_OVERRIDES
        assert "general dynamics" in MANUAL_OVERRIDES
        assert "northrop grumman" in MANUAL_OVERRIDES
