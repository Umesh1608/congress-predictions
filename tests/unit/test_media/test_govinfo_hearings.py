"""Tests for GovInfo hearing transcript collector transform logic."""

from datetime import date

from src.ingestion.media.govinfo_hearings import GovInfoHearingCollector, _strip_html, _parse_date


class TestGovInfoHearingTransform:
    def setup_method(self):
        self.collector = GovInfoHearingCollector(api_key="test", congress=118)

    def test_transform_basic_hearing(self):
        raw = {
            "packageId": "CHRG-118shrg12345",
            "title": "Hearing on AI Regulation",
            "dateIssued": "2024-03-15",
            "category": "Hearing",
            "_summary": {
                "committee": "Senate Commerce Committee",
                "congress": 118,
                "description": "Hearing examining AI policy.",
            },
            "_full_text": "<p>Good morning. We are here to discuss AI.</p>",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["source_type"] == "hearing_transcript"
        assert result["source_id"] == "CHRG-118shrg12345"
        assert result["title"] == "Hearing on AI Regulation"
        assert result["published_date"] == date(2024, 3, 15)
        assert "AI" in result["content"]
        assert result["raw_metadata"]["committee"] == "Senate Commerce Committee"

    def test_transform_empty_package_id_returns_none(self):
        raw = {"packageId": "", "title": "Something"}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_missing_package_id_returns_none(self):
        raw = {"title": "No ID here"}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_no_title_returns_none(self):
        raw = {"packageId": "CHRG-123", "title": ""}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_no_full_text_still_works(self):
        raw = {
            "packageId": "CHRG-118test",
            "title": "Test Hearing",
            "dateIssued": "2024-01-01",
            "_summary": {"description": "A test hearing."},
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["content"] == ""
        assert result["summary"] == "A test hearing."

    def test_transform_generates_govinfo_url(self):
        raw = {
            "packageId": "CHRG-118shrg99",
            "title": "Budget Hearing",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert "govinfo.gov" in result["url"]
        assert "CHRG-118shrg99" in result["url"]


class TestStripHtml:
    def test_removes_tags(self):
        assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_empty_string(self):
        assert _strip_html("") == ""

    def test_no_html(self):
        assert _strip_html("plain text") == "plain text"


class TestParseDate:
    def test_iso_date(self):
        assert _parse_date("2024-03-15") == date(2024, 3, 15)

    def test_iso_datetime(self):
        assert _parse_date("2024-03-15T10:30:00") == date(2024, 3, 15)

    def test_invalid_date(self):
        assert _parse_date("not-a-date") is None
