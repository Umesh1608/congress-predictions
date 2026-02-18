"""Tests for Voteview collector."""

import math

from src.ingestion.legislation.voteview import VoteviewCollector


class TestVoteviewTransform:
    def setup_method(self):
        self.collector = VoteviewCollector()

    def test_transform_basic_record(self):
        raw = {
            "bioguide_id": "P000197",
            "congress": 118,
            "chamber": "House",
            "party_code": 100,
            "state_abbrev": "CA",
            "bioname": "PELOSI, Nancy",
            "nominate_dim1": -0.523,
            "nominate_dim2": 0.112,
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["bioguide_id"] == "P000197"
        assert result["nominate_dim1"] == -0.523
        assert result["nominate_dim2"] == 0.112
        assert result["party"] == "Democratic"
        assert result["chamber"] == "house"
        assert result["full_name"] == "Nancy Pelosi"

    def test_transform_republican(self):
        raw = {
            "bioguide_id": "T000278",
            "congress": 118,
            "chamber": "Senate",
            "party_code": 200,
            "state_abbrev": "AL",
            "bioname": "TUBERVILLE, Tommy",
            "nominate_dim1": 0.654,
            "nominate_dim2": -0.231,
        }
        result = self.collector.transform(raw)
        assert result["party"] == "Republican"
        assert result["chamber"] == "senate"
        assert result["nominate_dim1"] == 0.654

    def test_transform_independent(self):
        raw = {
            "bioguide_id": "S000033",
            "congress": 118,
            "chamber": "Senate",
            "party_code": 328,
            "state_abbrev": "VT",
            "bioname": "SANDERS, Bernard",
            "nominate_dim1": -0.721,
            "nominate_dim2": 0.044,
        }
        result = self.collector.transform(raw)
        assert result["party"] == "Independent"

    def test_transform_missing_bioguide_returns_none(self):
        raw = {
            "bioguide_id": "",
            "congress": 118,
            "nominate_dim1": 0.1,
            "nominate_dim2": 0.2,
        }
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_both_scores_nan_returns_none(self):
        raw = {
            "bioguide_id": "X000001",
            "congress": 118,
            "chamber": "House",
            "nominate_dim1": float("nan"),
            "nominate_dim2": float("nan"),
        }
        result = self.collector.transform(raw)
        assert result is None
