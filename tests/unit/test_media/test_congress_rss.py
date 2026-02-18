"""Tests for Congress.gov RSS feed collector transform logic."""

from datetime import date

from src.ingestion.media.congress_rss import CongressRSSCollector, _strip_html, _parse_rss_date


class TestCongressRSSTransform:
    def setup_method(self):
        self.collector = CongressRSSCollector(feeds={"test": "http://example.com/rss"})

    def test_transform_basic_entry(self):
        raw = {
            "title": "H.R. 1234 - Introduced in House",
            "id": "https://www.congress.gov/bill/118th-congress/house-bill/1234",
            "link": "https://www.congress.gov/bill/118th-congress/house-bill/1234",
            "summary": "<p>A bill to regulate something.</p>",
            "published": "Mon, 15 Jul 2024 10:00:00 +0000",
            "_feed_name": "new_bills",
            "tags": [{"term": "legislation"}],
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["source_type"] == "congress_rss"
        assert "1234" in result["source_id"]
        assert result["title"] == "H.R. 1234 - Introduced in House"
        assert "regulate something" in result["content"]
        assert "<p>" not in result["content"]

    def test_transform_no_title_returns_none(self):
        raw = {"title": "", "id": "test"}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_no_id_returns_none(self):
        raw = {"title": "Has Title", "id": "", "link": ""}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_uses_link_as_fallback_id(self):
        raw = {
            "title": "Floor Update",
            "link": "https://congress.gov/floor/123",
            "summary": "Session began at 10 AM.",
            "_feed_name": "house_floor",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["source_id"] == "https://congress.gov/floor/123"

    def test_transform_preserves_feed_name(self):
        raw = {
            "title": "Test",
            "id": "test-id",
            "_feed_name": "most_viewed_bills",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["raw_metadata"]["feed_name"] == "most_viewed_bills"


class TestRSSStripHtml:
    def test_strips_tags_and_entities(self):
        assert "Hello world" in _strip_html("<p>Hello &amp; world</p>")

    def test_empty(self):
        assert _strip_html("") == ""


class TestRSSParseDate:
    def test_rfc822(self):
        result = _parse_rss_date("Mon, 15 Jul 2024 10:00:00 +0000")
        assert result == date(2024, 7, 15)

    def test_iso(self):
        assert _parse_rss_date("2024-07-15") == date(2024, 7, 15)

    def test_invalid(self):
        assert _parse_rss_date("garbage") is None
