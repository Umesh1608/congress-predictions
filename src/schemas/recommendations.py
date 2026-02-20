"""Pydantic schemas for trade screener recommendations."""

from __future__ import annotations

from pydantic import BaseModel


class RecommendationEvidence(BaseModel):
    member_names: list[str] = []
    signal_count: int = 0
    signal_types: list[str] = []
    strongest_type: str | None = None


class RiskContext(BaseModel):
    volatility_21d: float | None = None
    rsi_14: float | None = None
    price_change_5d: float | None = None


class TickerRecommendation(BaseModel):
    ticker: str
    action: str = "BUY"
    composite_score: float
    allocation_pct: float
    suggested_amount: float
    suggested_shares: int
    current_price: float | None = None
    # Score components
    avg_signal_strength: float = 0.0
    ml_confidence: float = 0.0
    best_member_win_rate: float = 0.0
    freshness_score: float = 0.0
    corroboration_score: float = 0.0
    # Nested
    evidence: RecommendationEvidence = RecommendationEvidence()
    risk: RiskContext = RiskContext()


class ScreenerResponse(BaseModel):
    portfolio_size: float
    num_recommendations: int
    total_investable: float
    cash_remainder: float
    recommendations: list[TickerRecommendation]
    disclaimer: str = (
        "NOT financial advice. This is an automated screening tool for research "
        "purposes only. Past congressional trading patterns do not guarantee "
        "future results. Always do your own due diligence before investing."
    )
    generated_at: str
