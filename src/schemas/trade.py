from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pydantic import BaseModel


class TradeResponse(BaseModel):
    id: int
    member_name: str
    member_bioguide_id: str | None = None
    filer_type: str
    ticker: str | None = None
    asset_name: str
    asset_type: str | None = None
    transaction_type: str
    transaction_date: date
    disclosure_date: date | None = None
    amount_range_low: Decimal | None = None
    amount_range_high: Decimal | None = None
    chamber: str | None = None
    source: str
    filing_url: str | None = None
    disclosure_lag_days: int | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class TradeListParams(BaseModel):
    member_name: str | None = None
    ticker: str | None = None
    chamber: str | None = None
    transaction_type: str | None = None
    filer_type: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    limit: int = 50
    offset: int = 0


class TradeStatsResponse(BaseModel):
    total_trades: int
    total_members: int
    most_traded_tickers: list[dict]
    most_active_members: list[dict]
    recent_trades_count_7d: int
