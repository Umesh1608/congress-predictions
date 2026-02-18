"""Tests for NLP text processing utilities.

Heavy ML model functions (FinBERT, spaCy, zero-shot) are mocked to
avoid loading models in CI. Only utility functions are tested directly.
"""

from unittest.mock import MagicMock, patch

from src.processing.text_processing import (
    extract_ticker_mentions,
    strip_html,
    truncate_for_model,
)


class TestExtractTickerMentions:
    def test_finds_dollar_prefixed_tickers(self):
        text = "Bought $NVDA and $AAPL today"
        tickers = extract_ticker_mentions(text, known_tickers={"NVDA", "AAPL"})
        assert "NVDA" in tickers
        assert "AAPL" in tickers

    def test_finds_bare_tickers(self):
        text = "Trading MSFT and GOOGL"
        tickers = extract_ticker_mentions(text, known_tickers={"MSFT", "GOOGL"})
        assert "MSFT" in tickers
        assert "GOOGL" in tickers

    def test_filters_false_positives(self):
        text = "The CEO said HR and PR teams are working on IT"
        tickers = extract_ticker_mentions(text)
        assert "CEO" not in tickers
        assert "HR" not in tickers
        assert "PR" not in tickers
        assert "IT" not in tickers

    def test_filters_against_known_tickers(self):
        text = "Looking at NVDA FAKE and AAPL"
        tickers = extract_ticker_mentions(text, known_tickers={"NVDA", "AAPL"})
        assert "NVDA" in tickers
        assert "AAPL" in tickers
        assert "FAKE" not in tickers

    def test_empty_text(self):
        assert extract_ticker_mentions("") == []

    def test_no_duplicates(self):
        text = "$NVDA is great. NVDA will rise. $NVDA again."
        tickers = extract_ticker_mentions(text, known_tickers={"NVDA"})
        assert tickers.count("NVDA") == 1

    def test_no_known_tickers_allows_all(self):
        text = "Looking at NVDA today"
        tickers = extract_ticker_mentions(text)
        assert "NVDA" in tickers


class TestStripHtml:
    def test_removes_tags(self):
        assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_removes_entities(self):
        result = strip_html("&amp; &lt; &gt;")
        assert "&amp;" not in result
        assert "&lt;" not in result

    def test_empty(self):
        assert strip_html("") == ""

    def test_none_returns_empty(self):
        assert strip_html(None) == ""

    def test_collapses_whitespace(self):
        result = strip_html("<p>Hello</p>   <p>World</p>")
        assert "  " not in result


class TestTruncateForModel:
    def test_short_text_unchanged(self):
        assert truncate_for_model("short") == "short"

    def test_long_text_truncated(self):
        text = "a" * 5000
        result = truncate_for_model(text, max_chars=100)
        assert len(result) == 100

    def test_empty_text(self):
        assert truncate_for_model("") == ""


class TestAnalyzeSentimentMocked:
    @patch("src.processing.text_processing.get_finbert")
    def test_positive_sentiment(self, mock_get_finbert):
        from src.processing.text_processing import analyze_sentiment

        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"label": "positive", "score": 0.95}]
        mock_get_finbert.return_value = mock_pipe

        result = analyze_sentiment("Great earnings report!")
        assert result["label"] == "positive"
        assert result["score"] > 0
        assert result["confidence"] == 0.95

    @patch("src.processing.text_processing.get_finbert")
    def test_negative_sentiment(self, mock_get_finbert):
        from src.processing.text_processing import analyze_sentiment

        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"label": "negative", "score": 0.85}]
        mock_get_finbert.return_value = mock_pipe

        result = analyze_sentiment("Stock crashed after scandal.")
        assert result["label"] == "negative"
        assert result["score"] < 0
        assert result["confidence"] == 0.85

    def test_empty_text_returns_neutral(self):
        from src.processing.text_processing import analyze_sentiment

        result = analyze_sentiment("")
        assert result["label"] == "neutral"
        assert result["score"] == 0.0


class TestExtractEntitiesMocked:
    @patch("src.processing.text_processing.get_spacy")
    def test_extracts_entities(self, mock_get_spacy):
        from src.processing.text_processing import extract_entities

        # Create mock spaCy doc with entities
        mock_nlp = MagicMock()
        mock_ent = MagicMock()
        mock_ent.text = "Nancy Pelosi"
        mock_ent.label_ = "PERSON"
        mock_ent.start_char = 0
        mock_ent.end_char = 12

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc
        mock_get_spacy.return_value = mock_nlp

        result = extract_entities("Nancy Pelosi trades stocks")
        assert len(result) >= 1
        assert result[0]["text"] == "Nancy Pelosi"
        assert result[0]["label"] == "PERSON"

    def test_empty_text(self):
        from src.processing.text_processing import extract_entities

        assert extract_entities("") == []
