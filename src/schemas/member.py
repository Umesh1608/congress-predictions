from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel


class MemberResponse(BaseModel):
    bioguide_id: str
    full_name: str
    first_name: str | None = None
    last_name: str | None = None
    chamber: str
    state: str | None = None
    district: str | None = None
    party: str | None = None
    in_office: bool = True
    trade_count: int | None = None

    model_config = {"from_attributes": True}


class MemberDetailResponse(MemberResponse):
    first_elected: int | None = None
    social_accounts: dict | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
