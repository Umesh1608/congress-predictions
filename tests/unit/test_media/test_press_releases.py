"""Tests for member press release collector transform logic."""

from datetime import date

from src.ingestion.media.press_releases import PressReleaseCollector, _strip_html


class TestPressReleaseTransform:
    def setup_method(self):
        self.collector = PressReleaseCollector(
            member_feeds={"P000197": ("Nancy Pelosi", "http://example.com/rss")}
        )

    def test_transform_basic_press_release(self):
        raw = {
            "title": "Pelosi Statement on Technology Regulation",
            "id": "https://pelosi.house.gov/news/press-releases/tech-regulation",
            "link": "https://pelosi.house.gov/news/press-releases/tech-regulation",
            "summary": "Speaker Pelosi issued the following statement on tech regulation.",
            "published": "Wed, 20 Mar 2024 12:00:00 +0000",
            "_bioguide_id": "P000197",
            "_member_name": "Nancy Pelosi",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["source_type"] == "press_release"
        assert result["title"] == "Pelosi Statement on Technology Regulation"
        assert result["author"] == "Nancy Pelosi"
        assert result["member_bioguide_ids"] == ["P000197"]
        assert "tech regulation" in result["content"]

    def test_transform_no_title_returns_none(self):
        raw = {"title": "", "id": "test", "_bioguide_id": "P000197"}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_no_id_returns_none(self):
        raw = {"title": "Has Title", "id": "", "link": "", "_bioguide_id": "P000197"}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_html_content_stripped(self):
        raw = {
            "title": "Statement",
            "id": "test-id",
            "_bioguide_id": "P000197",
            "_member_name": "Nancy Pelosi",
            "summary": "<p><b>Washington</b>, D.C. &mdash; Speaker Pelosi said...</p>",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert "<p>" not in result["content"]
        assert "<b>" not in result["content"]

    def test_transform_content_from_content_field(self):
        raw = {
            "title": "Statement",
            "id": "test-id-2",
            "_bioguide_id": "P000197",
            "_member_name": "Nancy Pelosi",
            "content": [{"value": "<div>Full text of the release.</div>"}],
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert "Full text of the release" in result["content"]

    def test_transform_no_bioguide_empty_list(self):
        raw = {
            "title": "Statement",
            "id": "test-id-3",
            "_bioguide_id": "",
            "_member_name": "",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["member_bioguide_ids"] == []
