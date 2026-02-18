"""Tests for NewsData.io collector transform logic."""

from datetime import date

from src.ingestion.media.newsdata import NewsDataCollector


class TestNewsDataTransform:
    def setup_method(self):
        self.collector = NewsDataCollector(api_key="test")

    def test_transform_basic_article(self):
        raw = {
            "article_id": "nd-article-001",
            "title": "Senator Trades Tech Stocks Before Hearing",
            "description": "A senator purchased technology stocks days before a committee hearing.",
            "link": "https://newssite.com/article/001",
            "pubDate": "2024-07-20T08:00:00Z",
            "creator": ["Jane Reporter"],
            "source_id": "newssite",
            "source_name": "News Site",
            "category": ["politics", "business"],
            "keywords": ["congress", "stocks"],
            "country": ["us"],
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["source_type"] == "newsdata"
        assert result["source_id"] == "nd-article-001"
        assert result["title"] == "Senator Trades Tech Stocks Before Hearing"
        assert result["published_date"] == date(2024, 7, 20)
        assert result["author"] == "Jane Reporter"
        assert result["raw_metadata"]["categories"] == ["politics", "business"]

    def test_transform_no_title_returns_none(self):
        raw = {"title": "", "article_id": "test"}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_no_source_id_returns_none(self):
        raw = {"title": "Has Title", "article_id": "", "link": ""}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_alternate_date_format(self):
        raw = {
            "article_id": "nd-002",
            "title": "Article",
            "link": "https://example.com/2",
            "pubDate": "2024-07-20 12:00:00",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["published_date"] == date(2024, 7, 20)

    def test_transform_null_creator_handled(self):
        raw = {
            "article_id": "nd-003",
            "title": "No Creator",
            "link": "https://example.com/3",
            "creator": None,
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["author"] == ""
