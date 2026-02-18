"""Tests for GNews API collector transform logic."""

from datetime import date

from src.ingestion.media.gnews import GNewsCollector


class TestGNewsTransform:
    def setup_method(self):
        self.collector = GNewsCollector(api_key="test")

    def test_transform_basic_article(self):
        raw = {
            "title": "Congress Members Stock Trades Under Scrutiny",
            "description": "New report reveals pattern of trading ahead of legislation.",
            "url": "https://example.com/article/123",
            "publishedAt": "2024-06-15T10:30:00Z",
            "source": {
                "name": "Test News",
                "url": "https://example.com",
            },
            "image": "https://example.com/img.jpg",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["source_type"] == "gnews"
        assert result["source_id"] == "https://example.com/article/123"
        assert result["title"] == "Congress Members Stock Trades Under Scrutiny"
        assert "pattern of trading" in result["content"]
        assert result["published_date"] == date(2024, 6, 15)
        assert result["author"] == "Test News"

    def test_transform_no_title_returns_none(self):
        raw = {"title": "", "url": "https://example.com/1"}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_no_url_returns_none(self):
        raw = {"title": "Has Title", "url": ""}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_bad_date(self):
        raw = {
            "title": "Article",
            "url": "https://example.com/2",
            "publishedAt": "not-a-date",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["published_date"] is None

    def test_transform_uses_content_fallback(self):
        raw = {
            "title": "Article",
            "url": "https://example.com/3",
            "description": "",
            "content": "Full article content here.",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["content"] == "Full article content here."
