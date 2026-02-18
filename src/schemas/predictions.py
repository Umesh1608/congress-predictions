"""Pydantic schemas for ML predictions API."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    id: int
    trade_id: int
    prediction_type: str
    predicted_value: float | None = None
    predicted_label: str | None = None
    confidence: float | None = None
    actual_return_5d: float | None = None
    actual_return_21d: float | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class PredictionDetailResponse(PredictionResponse):
    model_artifact_id: int | None = None
    feature_vector: dict | None = None


class ModelPerformanceResponse(BaseModel):
    model_name: str
    model_version: str
    is_active: bool
    metrics: dict = {}
    feature_columns: list[str] = []
    trained_at: datetime | None = None


class LeaderboardEntry(BaseModel):
    member_name: str
    bioguide_id: str | None = None
    total_trades: int = 0
    predicted_profitable: int = 0
    actual_profitable: int = 0
    accuracy: float = 0.0
    avg_confidence: float = 0.0


class PredictionStatsResponse(BaseModel):
    total_predictions: int = 0
    predictions_by_type: dict = {}
    avg_confidence: float = 0.0
    accuracy_5d: float | None = None
