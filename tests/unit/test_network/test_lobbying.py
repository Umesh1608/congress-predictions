"""Tests for lobbying data collector transform logic."""

from datetime import date

from src.ingestion.network.lobbying import (
    LobbyingFilingCollector,
    _extract_bill_references,
    _is_former_congress,
    _is_former_executive,
    _normalize_lobbyist_name,
)


class TestLobbyingFilingTransform:
    def setup_method(self):
        self.collector = LobbyingFilingCollector(filing_year=2024)

    def test_transform_basic_filing(self):
        raw = {
            "filing_uuid": "abc-123-def",
            "filing_type": "RR",
            "filing_type_display": "Registration",
            "filing_year": 2024,
            "filing_period": "Q1",
            "filing_date": "2024-04-15",
            "income": "50000",
            "registrant": {
                "id": "REG001",
                "name": "Acme Lobbying Group",
                "description": "Government relations",
            },
            "client": {
                "name": "Tech Corp Inc",
                "general_description": "Technology company",
                "country": "USA",
                "state": "CA",
            },
            "lobbying_activities": [
                {
                    "general_issue_code": "CPT",
                    "description": "Issues related to H.R. 1234 and S. 567",
                    "government_entities": [{"name": "U.S. Senate"}],
                    "lobbyists": [
                        {
                            "lobbyist": {"first_name": "John", "last_name": "Smith"},
                            "covered_official_position": "Former Senator",
                        },
                    ],
                },
            ],
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["filing_uuid"] == "abc-123-def"
        assert result["filing_year"] == 2024
        assert result["amount"] == 50000.0
        assert result["registrant"]["name"] == "Acme Lobbying Group"
        assert result["client"]["name"] == "Tech Corp Inc"
        assert len(result["lobbyists"]) == 1
        assert result["lobbyists"][0]["name"] == "John Smith"
        assert result["lobbyists"][0]["is_former_congress"] is True
        assert "CPT" in result["general_issue_codes"]
        assert "U.S. Senate" in result["government_entities"]

    def test_transform_empty_uuid_returns_none(self):
        raw = {"filing_uuid": "", "registrant": {}}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_missing_uuid_returns_none(self):
        raw = {"registrant": {"id": "123"}}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_expenses_used_when_no_income(self):
        raw = {
            "filing_uuid": "xyz-789",
            "filing_type": "report",
            "filing_year": 2024,
            "income": None,
            "expenses": "75000",
            "registrant": {},
            "client": {},
            "lobbyists": [],
            "lobbying_activities": [],
        }
        result = self.collector.transform(raw)
        assert result["amount"] == 75000.0

    def test_transform_filing_date_parsed(self):
        raw = {
            "filing_uuid": "date-test",
            "filing_type": "registration",
            "filing_year": 2024,
            "filing_date": "2024-06-15T00:00:00",
            "registrant": {},
            "client": {},
            "lobbyists": [],
            "lobbying_activities": [],
        }
        result = self.collector.transform(raw)
        assert result["filing_date"] == date(2024, 6, 15)


class TestBillReferenceExtraction:
    def test_extract_hr_bill(self):
        bills = _extract_bill_references("Issues related to H.R. 1234")
        assert len(bills) >= 1
        assert any("1234" in b for b in bills)

    def test_extract_senate_bill(self):
        bills = _extract_bill_references("Regarding S. 567 provisions")
        assert len(bills) >= 1
        assert any("567" in b for b in bills)

    def test_extract_multiple_bills(self):
        bills = _extract_bill_references("Discussing H.R. 100 and S. 200 and H.R. 300")
        assert len(bills) >= 3

    def test_no_bills_returns_empty(self):
        bills = _extract_bill_references("General policy discussion")
        assert bills == []


class TestFormerPositionDetection:
    def test_former_senator(self):
        assert _is_former_congress("Former Senator from California") is True

    def test_former_representative(self):
        assert _is_former_congress("Former Representative, 5th District") is True

    def test_former_staff_director(self):
        assert _is_former_congress("Staff Director, Senate Finance Committee") is True

    def test_not_former_congress(self):
        assert _is_former_congress("CEO, Private Company") is False

    def test_empty_position(self):
        assert _is_former_congress("") is False

    def test_none_position(self):
        assert _is_former_congress(None) is False

    def test_former_executive(self):
        assert _is_former_executive("Assistant Secretary, Department of Defense") is True

    def test_former_white_house(self):
        assert _is_former_executive("White House Staff") is True

    def test_not_former_executive(self):
        assert _is_former_executive("Former Senator") is False


class TestLobbyistNameNormalization:
    def test_first_last_name(self):
        lob = {"lobbyist": {"first_name": "Jane", "last_name": "Doe"}}
        assert _normalize_lobbyist_name(lob) == "Jane Doe"

    def test_empty_lobbyist_dict(self):
        lob = {"lobbyist": {}}
        name = _normalize_lobbyist_name(lob)
        assert name == "Unknown"

    def test_missing_lobbyist_key(self):
        lob = {}
        name = _normalize_lobbyist_name(lob)
        # Should handle gracefully
        assert isinstance(name, str)
