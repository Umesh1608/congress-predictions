"""ML model artifact and trade prediction models."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.postgres import Base


class MLModelArtifact(Base):
    """Tracks trained ML model versions and their performance metrics."""

    __tablename__ = "ml_model_artifact"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), index=True)
    model_version: Mapped[str] = mapped_column(String(50))
    artifact_path: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    feature_columns: Mapped[dict | None] = mapped_column(JSONB, default=list)
    training_config: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("model_name", "model_version", name="uq_model_name_version"),
    )


class TradePrediction(Base):
    """ML predictions for individual trades."""

    __tablename__ = "trade_prediction"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    trade_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("trade_disclosure.id"), index=True
    )
    model_artifact_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("ml_model_artifact.id")
    )
    prediction_type: Mapped[str] = mapped_column(String(50))
    predicted_value: Mapped[float | None] = mapped_column(Numeric(8, 4))
    predicted_label: Mapped[str | None] = mapped_column(String(50))
    confidence: Mapped[float | None] = mapped_column(Numeric(5, 4))
    feature_vector: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    actual_return_5d: Mapped[float | None] = mapped_column(Numeric(8, 4))
    actual_return_21d: Mapped[float | None] = mapped_column(Numeric(8, 4))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "trade_id", "model_artifact_id", "prediction_type",
            name="uq_trade_prediction_unique",
        ),
    )
