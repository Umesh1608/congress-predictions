"""Pydantic response models for media content and sentiment endpoints."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel


class SentimentResponse(BaseModel):
    model_name: str
    sentiment_label: str | None = None
    sentiment_score: float | None = None
    confidence: float | None = None
    entities: list[dict[str, Any]] = []
    sectors: list[dict[str, Any]] = []
    tickers_extracted: list[str] = []


class MediaContentResponse(BaseModel):
    id: int
    source_type: str
    title: str | None = None
    summary: str | None = None
    url: str | None = None
    author: str | None = None
    published_date: date | None = None
    tickers_mentioned: list[str] = []
    member_bioguide_ids: list[str] = []
    sentiment: SentimentResponse | None = None
    created_at: datetime | None = None


class MediaContentDetailResponse(MediaContentResponse):
    content: str | None = None
    raw_metadata: dict[str, Any] = {}
    sentiment_analyses: list[SentimentResponse] = []


class MediaStatsResponse(BaseModel):
    total_content: int = 0
    by_source_type: dict[str, int] = {}
    recent_sentiment: dict[str, Any] = {}
    content_last_7_days: int = 0


class MemberSentimentPoint(BaseModel):
    date: date
    avg_score: float
    count: int
    sources: list[str] = []


class MemberSentimentTimelineResponse(BaseModel):
    bioguide_id: str
    member_name: str | None = None
    timeline: list[MemberSentimentPoint] = []
