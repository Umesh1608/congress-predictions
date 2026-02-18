from __future__ import annotations

from datetime import date, datetime
from pydantic import BaseModel


class BillResponse(BaseModel):
    id: int
    bill_id: str
    congress_number: int
    bill_type: str
    bill_number: int
    title: str
    short_title: str | None = None
    introduced_date: date | None = None
    sponsor_bioguide_id: str | None = None
    sponsor_name: str | None = None
    status: str | None = None
    policy_area: str | None = None
    latest_action_date: date | None = None
    latest_action_text: str | None = None
    committees: list | None = None
    subjects: list | None = None

    model_config = {"from_attributes": True}


class CommitteeResponse(BaseModel):
    system_code: str
    name: str
    chamber: str
    parent_code: str | None = None
    url: str | None = None
    is_current: bool = True

    model_config = {"from_attributes": True}


class HearingResponse(BaseModel):
    id: int
    committee_code: str | None = None
    title: str
    hearing_date: date | None = None
    chamber: str | None = None
    congress_number: int | None = None
    url: str | None = None
    related_bills: list | None = None

    model_config = {"from_attributes": True}


class LegislativeContextResponse(BaseModel):
    trade_id: int
    member_bioguide_id: str | None = None
    transaction_date: date
    disclosure_date: date | None = None
    disclosure_lag_days: int | None = None
    nearby_hearings: list[dict] = []
    nearby_bills: list[dict] = []
    nearby_votes: list[dict] = []
    committee_sector_alignment: bool = False
    aligned_committees: list[str] = []
    min_hearing_distance_days: int | None = None
    min_bill_distance_days: int | None = None
    timing_suspicion_score: float = 0.0


class CommitteeTradeResponse(BaseModel):
    trade_id: int
    member_name: str
    member_bioguide_id: str | None = None
    ticker: str | None = None
    asset_name: str
    transaction_type: str
    transaction_date: str
    disclosure_date: str | None = None
    amount_range_low: float | None = None
    amount_range_high: float | None = None
    committee_role: str | None = None
    committee_name: str | None = None
