"""ML predictions API endpoints."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.models.ml import MLModelArtifact, TradePrediction
from src.models.trade import TradeDisclosure
from src.schemas.predictions import (
    LeaderboardEntry,
    ModelPerformanceResponse,
    PredictionDetailResponse,
    PredictionResponse,
    PredictionStatsResponse,
)

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("", response_model=list[PredictionResponse])
async def list_predictions(
    member_bioguide_id: str | None = None,
    ticker: str | None = None,
    prediction_type: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    query = (
        select(TradePrediction)
        .join(TradeDisclosure, TradePrediction.trade_id == TradeDisclosure.id)
        .order_by(TradePrediction.created_at.desc())
    )

    if member_bioguide_id:
        query = query.where(TradeDisclosure.member_bioguide_id == member_bioguide_id)
    if ticker:
        query = query.where(TradeDisclosure.ticker == ticker)
    if prediction_type:
        query = query.where(TradePrediction.prediction_type == prediction_type)
    if date_from:
        query = query.where(TradeDisclosure.transaction_date >= date_from)
    if date_to:
        query = query.where(TradeDisclosure.transaction_date <= date_to)

    query = query.offset(offset).limit(limit)
    result = await db.execute(query)

    return [
        {
            "id": p.id,
            "trade_id": p.trade_id,
            "prediction_type": p.prediction_type,
            "predicted_value": float(p.predicted_value) if p.predicted_value is not None else None,
            "predicted_label": p.predicted_label,
            "confidence": float(p.confidence) if p.confidence is not None else None,
            "actual_return_5d": float(p.actual_return_5d) if p.actual_return_5d is not None else None,
            "actual_return_21d": float(p.actual_return_21d) if p.actual_return_21d is not None else None,
            "created_at": p.created_at,
        }
        for p in result.scalars().all()
    ]


@router.get("/model-performance", response_model=list[ModelPerformanceResponse])
async def model_performance(db: AsyncSession = Depends(get_db)) -> list[dict]:
    result = await db.execute(
        select(MLModelArtifact)
        .where(MLModelArtifact.is_active.is_(True))
        .order_by(MLModelArtifact.model_name)
    )

    return [
        {
            "model_name": a.model_name,
            "model_version": a.model_version,
            "is_active": a.is_active,
            "metrics": a.metrics or {},
            "feature_columns": a.feature_columns or [],
            "trained_at": a.trained_at,
        }
        for a in result.scalars().all()
    ]


@router.get("/stats", response_model=PredictionStatsResponse)
async def prediction_stats(db: AsyncSession = Depends(get_db)) -> dict:
    # Total predictions
    total_result = await db.execute(select(func.count(TradePrediction.id)))
    total = total_result.scalar() or 0

    # By type
    type_result = await db.execute(
        select(TradePrediction.prediction_type, func.count(TradePrediction.id))
        .group_by(TradePrediction.prediction_type)
    )
    by_type = {row[0]: row[1] for row in type_result.all()}

    # Average confidence
    conf_result = await db.execute(
        select(func.avg(TradePrediction.confidence))
        .where(TradePrediction.confidence.isnot(None))
    )
    avg_conf = float(conf_result.scalar() or 0)

    # 5d accuracy (for profitability predictions with actual returns)
    accuracy_result = await db.execute(
        select(
            func.count(TradePrediction.id),
            func.sum(
                case(
                    (
                        and_(
                            TradePrediction.predicted_label == "profitable",
                            TradePrediction.actual_return_5d > 0,
                        ),
                        1,
                    ),
                    (
                        and_(
                            TradePrediction.predicted_label == "unprofitable",
                            TradePrediction.actual_return_5d <= 0,
                        ),
                        1,
                    ),
                    else_=0,
                )
            ),
        )
        .where(
            and_(
                TradePrediction.prediction_type == "profitability",
                TradePrediction.actual_return_5d.isnot(None),
            )
        )
    )
    acc_row = accuracy_result.one()
    accuracy_5d = None
    if acc_row[0] and acc_row[0] > 0:
        accuracy_5d = float((acc_row[1] or 0) / acc_row[0])

    return {
        "total_predictions": total,
        "predictions_by_type": by_type,
        "avg_confidence": avg_conf,
        "accuracy_5d": accuracy_5d,
    }


@router.get("/leaderboard", response_model=list[LeaderboardEntry])
async def prediction_leaderboard(
    limit: int = Query(default=20, le=50),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Members ranked by how well their trades match ML predictions."""
    result = await db.execute(
        select(
            TradeDisclosure.member_name,
            TradeDisclosure.member_bioguide_id,
            func.count(TradePrediction.id).label("total"),
            func.sum(
                case(
                    (TradePrediction.predicted_label == "profitable", 1),
                    else_=0,
                )
            ).label("predicted_profitable"),
            func.sum(
                case(
                    (TradePrediction.actual_return_5d > 0, 1),
                    else_=0,
                )
            ).label("actual_profitable"),
            func.avg(TradePrediction.confidence).label("avg_conf"),
        )
        .join(TradePrediction, TradePrediction.trade_id == TradeDisclosure.id)
        .where(TradePrediction.prediction_type == "profitability")
        .group_by(TradeDisclosure.member_name, TradeDisclosure.member_bioguide_id)
        .having(func.count(TradePrediction.id) >= 5)
        .order_by(func.count(TradePrediction.id).desc())
        .limit(limit)
    )

    leaderboard = []
    for row in result.all():
        total = row.total or 0
        actual = row.actual_profitable or 0
        accuracy = actual / total if total > 0 else 0

        leaderboard.append({
            "member_name": row.member_name,
            "bioguide_id": row.member_bioguide_id,
            "total_trades": total,
            "predicted_profitable": row.predicted_profitable or 0,
            "actual_profitable": actual,
            "accuracy": round(accuracy, 4),
            "avg_confidence": round(float(row.avg_conf or 0), 4),
        })

    return leaderboard


@router.get("/{trade_id}", response_model=list[PredictionDetailResponse])
async def get_trade_predictions(
    trade_id: int,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    # Verify trade exists
    trade = await db.get(TradeDisclosure, trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    result = await db.execute(
        select(TradePrediction)
        .where(TradePrediction.trade_id == trade_id)
        .order_by(TradePrediction.prediction_type)
    )

    return [
        {
            "id": p.id,
            "trade_id": p.trade_id,
            "model_artifact_id": p.model_artifact_id,
            "prediction_type": p.prediction_type,
            "predicted_value": float(p.predicted_value) if p.predicted_value is not None else None,
            "predicted_label": p.predicted_label,
            "confidence": float(p.confidence) if p.confidence is not None else None,
            "feature_vector": p.feature_vector,
            "actual_return_5d": float(p.actual_return_5d) if p.actual_return_5d is not None else None,
            "actual_return_21d": float(p.actual_return_21d) if p.actual_return_21d is not None else None,
            "created_at": p.created_at,
        }
        for p in result.scalars().all()
    ]
