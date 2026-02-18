"""Signal and alert configuration models."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.postgres import Base


class Signal(Base):
    """Generated trading signals from ML predictions and data fusion."""

    __tablename__ = "signal"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    signal_type: Mapped[str] = mapped_column(String(50), index=True)
    member_bioguide_id: Mapped[str | None] = mapped_column(
        String(10), ForeignKey("congress_member.bioguide_id"), index=True
    )
    ticker: Mapped[str | None] = mapped_column(String(20), index=True)
    direction: Mapped[str] = mapped_column(String(10))  # bullish / bearish / neutral
    strength: Mapped[float] = mapped_column(Numeric(5, 4), default=0)
    confidence: Mapped[float] = mapped_column(Numeric(5, 4), default=0)
    evidence: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_signal_type_active_created", "signal_type", "is_active", "created_at"),
    )


class AlertConfig(Base):
    """User-defined alert configurations for signal notifications."""

    __tablename__ = "alert_config"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200))
    signal_types: Mapped[dict | None] = mapped_column(JSONB, default=list)
    min_strength: Mapped[float] = mapped_column(Numeric(5, 4), default=0.5)
    tickers: Mapped[dict | None] = mapped_column(JSONB)  # nullable, filter to specific tickers
    members: Mapped[dict | None] = mapped_column(JSONB)  # nullable, filter to specific members
    webhook_url: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
