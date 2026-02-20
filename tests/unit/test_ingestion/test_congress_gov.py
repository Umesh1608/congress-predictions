"""Tests for Congress.gov API collectors."""

from datetime import date

from src.ingestion.legislation.congress_gov import (
    CongressBillCollector,
    CongressCommitteeCollector,
    CongressHearingCollector,
    CongressMemberCollector,
    _current_congress,
)


class TestCongressMemberTransform:
    def setup_method(self):
        self.collector = CongressMemberCollector()

    def test_transform_basic_member(self):
        raw = {
            "bioguideId": "P000197",
            "name": "Pelosi, Nancy",
            "firstName": "Nancy",
            "lastName": "Pelosi",
            "partyName": "Democratic",
            "state": "CA",
            "district": 11,
            "currentMember": True,
            "terms": [{"chamber": "House of Representatives"}],
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["bioguide_id"] == "P000197"
        assert result["full_name"] == "Nancy Pelosi"
        assert result["chamber"] == "house"
        assert result["party"] == "Democratic"
        assert result["state"] == "CA"
        assert result["in_office"] is True

    def test_transform_senator(self):
        raw = {
            "bioguideId": "T000278",
            "firstName": "Tommy",
            "lastName": "Tuberville",
            "partyName": "Republican",
            "state": "AL",
            "currentMember": True,
            "terms": [{"chamber": "Senate"}],
        }
        result = self.collector.transform(raw)
        assert result["chamber"] == "senate"
        assert result["party"] == "Republican"

    def test_transform_missing_bioguide_returns_none(self):
        raw = {"name": "Nobody", "bioguideId": ""}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_name_from_comma_format(self):
        raw = {
            "bioguideId": "X000001",
            "name": "Smith, John A.",
            "currentMember": True,
            "terms": [],
        }
        result = self.collector.transform(raw)
        assert result["first_name"] == "John A."
        assert result["last_name"] == "Smith"


class TestCongressBillTransform:
    def setup_method(self):
        self.collector = CongressBillCollector(congress=118)

    def test_transform_basic_bill(self):
        raw = {
            "congress": 118,
            "type": "HR",
            "number": 1234,
            "title": "A bill to do something important",
            "introducedDate": "2023-03-15",
            "latestAction": {
                "actionDate": "2023-04-01",
                "text": "Referred to Committee",
            },
            "sponsors": [
                {"bioguideId": "P000197", "fullName": "Nancy Pelosi"}
            ],
            "policyArea": {"name": "Finance and Financial Sector"},
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["bill_id"] == "hr1234-118"
        assert result["congress_number"] == 118
        assert result["bill_type"] == "hr"
        assert result["bill_number"] == 1234
        assert result["introduced_date"] == date(2023, 3, 15)
        assert result["sponsor_bioguide_id"] == "P000197"
        assert result["policy_area"] == "Finance and Financial Sector"
        assert result["latest_action_text"] == "Referred to Committee"

    def test_transform_bill_no_sponsor(self):
        raw = {
            "congress": 118,
            "type": "S",
            "number": 5678,
            "title": "Senate bill",
            "introducedDate": "2023-06-01",
            "latestAction": {"text": "Introduced"},
            "sponsors": [],
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["bill_id"] == "s5678-118"
        assert result["sponsor_bioguide_id"] is None

    def test_transform_missing_fields_returns_none(self):
        raw = {"congress": None, "type": "", "number": None}
        result = self.collector.transform(raw)
        assert result is None


class TestCongressCommitteeTransform:
    def setup_method(self):
        self.collector = CongressCommitteeCollector()

    def test_transform_house_committee(self):
        raw = {
            "systemCode": "HSBA00",
            "name": "Financial Services",
            "chamber": "House",
            "isCurrent": True,
            "url": "https://financialservices.house.gov",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["system_code"] == "HSBA00"
        assert result["chamber"] == "house"
        assert result["is_current"] is True

    def test_transform_subcommittee(self):
        raw = {
            "systemCode": "HSBA01",
            "name": "Capital Markets Subcommittee",
            "chamber": "House",
            "parent": {"systemCode": "HSBA00"},
        }
        result = self.collector.transform(raw)
        assert result["parent_code"] == "HSBA00"

    def test_transform_empty_code_returns_none(self):
        raw = {"systemCode": "", "name": "Test"}
        result = self.collector.transform(raw)
        assert result is None


class TestCongressHearingTransform:
    def setup_method(self):
        self.collector = CongressHearingCollector(congress=118)

    def test_transform_basic_hearing(self):
        raw = {
            "title": "Oversight of Financial Regulators",
            "date": "2023-09-20",
            "chamber": "House",
            "congress": 118,
            "committee": {"systemCode": "HSBA00"},
            "url": "https://example.com/hearing",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["title"] == "Oversight of Financial Regulators"
        assert result["hearing_date"] == date(2023, 9, 20)
        assert result["committee_code"] == "HSBA00"
        assert result["chamber"] == "house"

    def test_transform_with_associated_bills(self):
        raw = {
            "title": "Discussion of HR 1234",
            "date": "2023-10-01",
            "chamber": "Senate",
            "congress": 118,
            "committee": {"systemCode": "SSBK00"},
            "associatedBills": [
                {"congress": 118, "type": "HR", "number": 1234}
            ],
        }
        result = self.collector.transform(raw)
        assert len(result["related_bills"]) == 1
        assert result["related_bills"][0]["number"] == 1234

    def test_transform_no_title_returns_none(self):
        raw = {"title": "", "date": "2023-01-01"}
        result = self.collector.transform(raw)
        assert result is None

    def test_transform_api_v3_format(self):
        """Test transform with actual Congress.gov API v3 response format."""
        raw = {
            "chamber": "Senate",
            "congress": 119,
            "title": "BUSINESS MEETING",
            "committees": [
                {
                    "name": "Senate Public Works Committee",
                    "systemCode": "ssev00",
                    "url": "https://api.congress.gov/v3/committee/senate/ssev00",
                }
            ],
            "dates": [{"date": "2025-02-05"}],
            "formats": [
                {"type": "PDF", "url": "https://congress.gov/119/chrg/test.pdf"}
            ],
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["title"] == "BUSINESS MEETING"
        assert result["hearing_date"] == date(2025, 2, 5)
        assert result["committee_code"] == "ssev00"
        assert result["chamber"] == "senate"
        assert result["url"] == "https://congress.gov/119/chrg/test.pdf"


class TestCurrentCongress:
    def test_current_congress_is_reasonable(self):
        congress = _current_congress()
        # 119th Congress started Jan 2025
        assert congress >= 119
        assert congress < 130  # sanity upper bound
