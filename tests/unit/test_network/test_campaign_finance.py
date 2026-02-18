"""Tests for FEC campaign finance collector transform logic."""

from datetime import date

from src.ingestion.network.campaign_finance import (
    FECCommitteeCollector,
    FECContributionCollector,
)


class TestFECCommitteeTransform:
    def setup_method(self):
        self.collector = FECCommitteeCollector(api_key="test", cycle=2024)

    def test_transform_basic_committee(self):
        raw = {
            "committee_id": "C00123456",
            "name": "PELOSI FOR CONGRESS",
            "committee_type": "H",
            "party": "DEM",
            "state": "CA",
            "candidate_ids": ["H6CA05245"],
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["fec_committee_id"] == "C00123456"
        assert result["name"] == "PELOSI FOR CONGRESS"
        assert result["committee_type"] == "H"
        assert result["candidate_fec_id"] == "H6CA05245"

    def test_transform_no_candidate(self):
        raw = {
            "committee_id": "C00789012",
            "name": "Some PAC",
            "committee_type": "Q",
            "candidate_ids": [],
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["candidate_fec_id"] is None

    def test_transform_empty_id_returns_none(self):
        raw = {"committee_id": "", "name": "Nothing"}
        result = self.collector.transform(raw)
        assert result is None


class TestFECContributionTransform:
    def setup_method(self):
        self.collector = FECContributionCollector(api_key="test")

    def test_transform_basic_contribution(self):
        raw = {
            "transaction_id": "SA11AI.12345",
            "_committee_id": "C00123456",
            "contributor_name": "DOE, JOHN",
            "contributor_employer": "MICROSOFT CORPORATION",
            "contributor_occupation": "SOFTWARE ENGINEER",
            "contributor_city": "SEATTLE",
            "contributor_state": "WA",
            "contributor_zip_code": "98101",
            "contribution_receipt_amount": 2800.0,
            "contribution_receipt_date": "2024-03-15",
            "receipt_type": "15",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["fec_transaction_id"] == "SA11AI.12345"
        assert result["fec_committee_id"] == "C00123456"
        assert result["contributor_name"] == "DOE, JOHN"
        assert result["contributor_employer"] == "MICROSOFT CORPORATION"
        assert result["amount"] == 2800.0
        assert result["contribution_date"] == date(2024, 3, 15)

    def test_transform_no_name_returns_none(self):
        raw = {
            "transaction_id": "test",
            "contributor_name": "",
            "contribution_receipt_amount": 500,
        }
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_no_amount_returns_none(self):
        raw = {
            "transaction_id": "test",
            "contributor_name": "DOE, JANE",
            "contribution_receipt_amount": None,
        }
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_missing_date_handled(self):
        raw = {
            "transaction_id": "test",
            "_committee_id": "C00111",
            "contributor_name": "DOE, JOHN",
            "contribution_receipt_amount": 1000,
            "contribution_receipt_date": None,
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["contribution_date"] is None

    def test_transform_bad_date_handled(self):
        raw = {
            "transaction_id": "test",
            "_committee_id": "C00111",
            "contributor_name": "DOE, JOHN",
            "contribution_receipt_amount": 1000,
            "contribution_receipt_date": "not-a-date",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["contribution_date"] is None
