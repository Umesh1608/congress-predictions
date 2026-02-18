"""Pydantic schemas for network graph API responses."""

from __future__ import annotations

from pydantic import BaseModel


class GraphNodeResponse(BaseModel):
    id: int | None = None
    labels: list[str] = []
    properties: dict = {}


class GraphRelationshipResponse(BaseModel):
    type: str
    startNode: int | None = None
    endNode: int | None = None
    properties: dict = {}


class MemberNetworkResponse(BaseModel):
    nodes: list[dict] = []
    relationships: list[dict] = []


class PathResponse(BaseModel):
    nodes: list[dict] = []
    relationships: list[dict] = []
    length: int = 0


class PathsResponse(BaseModel):
    member_bioguide_id: str
    ticker: str
    paths: list[PathResponse] = []
    total_paths: int = 0


class SuspiciousTriangleResponse(BaseModel):
    member_bioguide_id: str
    member_name: str | None = None
    ticker: str
    company_name: str | None = None
    lobbying_firm: str | None = None
    trade_type: str | None = None
    trade_date: str | None = None
    amount_low: float | None = None
    amount_high: float | None = None


class CommitteeCompanyOverlapResponse(BaseModel):
    member_bioguide_id: str
    member_name: str | None = None
    committee_code: str
    committee_name: str | None = None
    ticker: str
    company_name: str | None = None
    lobbying_firm: str | None = None
    related_bill: str | None = None
    trade_type: str | None = None
    trade_date: str | None = None


class TradingConnectionsResponse(BaseModel):
    bioguide_id: str
    full_name: str | None = None
    traded_companies: list[dict] = []
    lobbying_connections: list[dict] = []
    donation_connections: list[dict] = []
    companies_traded: int = 0
    lobbying_link_count: int = 0
    donation_link_count: int = 0


class GraphStatsResponse(BaseModel):
    members: int = 0
    companies: int = 0
    committees: int = 0
    bills: int = 0
    firms: int = 0
    trades: int = 0
    lobbied_for: int = 0
    donated_to: int = 0
