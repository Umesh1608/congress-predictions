"""Pydantic schemas for signals and alerts API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class SignalResponse(BaseModel):
    id: int
    signal_type: str
    member_bioguide_id: str | None = None
    ticker: str | None = None
    direction: str
    strength: float
    confidence: float
    is_active: bool = True
    expires_at: datetime | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class SignalDetailResponse(SignalResponse):
    evidence: dict = {}


class SignalStatsResponse(BaseModel):
    total_active: int = 0
    by_type: dict = {}
    avg_strength: float = 0.0
    by_direction: dict = {}


class AlertConfigCreate(BaseModel):
    name: str
    signal_types: list[str] = []
    min_strength: float = 0.5
    tickers: list[str] | None = None
    members: list[str] | None = None
    webhook_url: str | None = None


class AlertConfigResponse(BaseModel):
    id: int
    name: str
    signal_types: list | None = None
    min_strength: float
    tickers: list | None = None
    members: list | None = None
    webhook_url: str | None = None
    is_active: bool = True
    created_at: datetime | None = None

    model_config = {"from_attributes": True}
