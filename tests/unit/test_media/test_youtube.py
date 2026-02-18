"""Tests for YouTube transcript collector transform logic."""

from datetime import date

from src.ingestion.media.youtube import YouTubeTranscriptCollector


class TestYouTubeTranscriptTransform:
    def setup_method(self):
        self.collector = YouTubeTranscriptCollector(channels={"test_id": "Test Channel"})

    def test_transform_basic_video(self):
        raw = {
            "video_id": "dQw4w9WgXcQ",
            "title": "Congressional Hearing on Tech",
            "published": "2024-05-10T14:00:00+00:00",
            "transcript": "Today we discuss technology regulation in Congress.",
            "channel_id": "test_id",
            "channel_name": "Test Channel",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["source_type"] == "youtube"
        assert result["source_id"] == "dQw4w9WgXcQ"
        assert result["title"] == "Congressional Hearing on Tech"
        assert "technology regulation" in result["content"]
        assert result["published_date"] == date(2024, 5, 10)
        assert result["url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_transform_missing_video_id_returns_none(self):
        raw = {"video_id": "", "title": "Missing ID"}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_empty_transcript(self):
        raw = {
            "video_id": "abc123",
            "title": "No Transcript Available",
            "transcript": "",
            "channel_id": "test",
            "channel_name": "Test",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["content"] == ""
        assert result["raw_metadata"]["has_transcript"] is False

    def test_transform_with_transcript_flag(self):
        raw = {
            "video_id": "xyz789",
            "title": "Has Transcript",
            "transcript": "Some text here",
            "channel_id": "test",
            "channel_name": "Test",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["raw_metadata"]["has_transcript"] is True

    def test_transform_bad_date_handled(self):
        raw = {
            "video_id": "date_test",
            "title": "Date Test",
            "published": "invalid-date",
            "transcript": "",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["published_date"] is None

    def test_transform_missing_date(self):
        raw = {
            "video_id": "no_date",
            "title": "No Date",
            "transcript": "content",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["published_date"] is None
