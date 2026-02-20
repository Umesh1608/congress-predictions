"""Trade screener recommendations API endpoint."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.ml.screener import TradeScreener
from src.schemas.recommendations import (
    RecommendationEvidence,
    RiskContext,
    ScreenerResponse,
    TickerRecommendation,
)

router = APIRouter(tags=["recommendations"])


@router.get("/recommendations", response_model=ScreenerResponse)
async def get_recommendations(
    portfolio_size: float = Query(default=3000.0, ge=500, le=100000),
    top_n: int = Query(default=5, ge=1, le=10),
    min_signal_strength: float = Query(default=0.3, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    screener = TradeScreener(db)
    recs = await screener.get_recommendations(
        portfolio_size=portfolio_size,
        top_n=top_n,
        min_signal_strength=min_signal_strength,
    )

    recommendations = [
        TickerRecommendation(
            ticker=r["ticker"],
            action=r["action"],
            composite_score=r["composite_score"],
            allocation_pct=round(r.get("allocation_pct", 0), 4),
            suggested_amount=r.get("suggested_amount", 0),
            suggested_shares=r.get("suggested_shares", 0),
            current_price=r.get("current_price"),
            avg_signal_strength=r.get("avg_signal_strength", 0),
            ml_confidence=r.get("ml_confidence", 0),
            best_member_win_rate=r.get("best_member_win_rate", 0),
            freshness_score=r.get("freshness_score", 0),
            corroboration_score=r.get("corroboration_score", 0),
            evidence=RecommendationEvidence(**r.get("evidence", {})),
            risk=RiskContext(**r.get("risk", {})),
        )
        for r in recs
    ]

    total_investable = sum(r.suggested_amount for r in recommendations)
    cash_remainder = portfolio_size - total_investable

    return ScreenerResponse(
        portfolio_size=portfolio_size,
        num_recommendations=len(recommendations),
        total_investable=round(total_investable, 2),
        cash_remainder=round(cash_remainder, 2),
        recommendations=recommendations,
        generated_at=datetime.now(timezone.utc).isoformat(),
    ).model_dump()
