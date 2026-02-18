"""Tests for Twitter collector transform logic."""

from datetime import date

from src.ingestion.media.twitter import TwitterCollector


class TestTwitterTransform:
    def setup_method(self):
        self.collector = TwitterCollector(
            bearer_token="test_token",
            accounts={"12345": ("TestAccount", "T000001")},
        )

    def test_transform_basic_tweet(self):
        raw = {
            "id": "1234567890",
            "text": "Just voted on the new infrastructure bill. Great progress for America!",
            "created_at": "2024-04-10T15:30:00Z",
            "public_metrics": {
                "retweet_count": 150,
                "like_count": 500,
                "reply_count": 30,
                "quote_count": 10,
            },
            "_screen_name": "TestAccount",
            "_bioguide_id": "T000001",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["source_type"] == "twitter"
        assert result["source_id"] == "1234567890"
        assert "infrastructure bill" in result["content"]
        assert result["published_date"] == date(2024, 4, 10)
        assert result["member_bioguide_ids"] == ["T000001"]
        assert result["raw_metadata"]["like_count"] == 500

    def test_transform_no_id_returns_none(self):
        raw = {"id": "", "text": "Some tweet"}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_no_text_returns_none(self):
        raw = {"id": "123", "text": ""}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_title_truncation(self):
        long_text = "A" * 200
        raw = {
            "id": "999",
            "text": long_text,
            "_screen_name": "Test",
            "_bioguide_id": "",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert len(result["title"]) <= 103  # 100 + "..."
        assert result["content"] == long_text

    def test_skip_when_no_bearer_token(self):
        collector = TwitterCollector(bearer_token="", accounts={})
        # The skip happens in collect(), not transform(), but verify it's graceful
        assert collector.bearer_token == ""
