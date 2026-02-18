"""Media content and NLP analysis models.

Stores all media types (hearing transcripts, YouTube videos, news articles,
press releases, tweets) in a unified table with source_type discriminator.
Sentiment analysis results are stored separately and linked by FK.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.postgres import Base


class MediaContent(Base):
    """Unified table for all media content types."""

    __tablename__ = "media_content"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(String(50), index=True)
    source_id: Mapped[str] = mapped_column(String(500))
    title: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(String(300))
    published_date: Mapped[date | None] = mapped_column(Date, index=True)
    member_bioguide_ids: Mapped[dict | None] = mapped_column(JSONB, default=list)
    tickers_mentioned: Mapped[dict | None] = mapped_column(JSONB, default=list)
    raw_metadata: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    sentiment_analyses: Mapped[list[SentimentAnalysis]] = relationship(
        back_populates="media_content", cascade="all, delete-orphan"
    )
    member_mentions: Mapped[list[MemberMediaMention]] = relationship(
        back_populates="media_content", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("source_type", "source_id", name="uq_media_content_source"),
        Index("ix_media_source_type_date", "source_type", "published_date"),
    )


class SentimentAnalysis(Base):
    """NLP analysis results linked to media content."""

    __tablename__ = "sentiment_analysis"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    media_content_id: Mapped[int] = mapped_column(
        ForeignKey("media_content.id", ondelete="CASCADE"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(100))
    sentiment_label: Mapped[str | None] = mapped_column(String(20))
    sentiment_score: Mapped[float | None] = mapped_column(Numeric(5, 4))
    confidence: Mapped[float | None] = mapped_column(Numeric(5, 4))
    entities: Mapped[dict | None] = mapped_column(JSONB, default=list)
    sectors: Mapped[dict | None] = mapped_column(JSONB, default=list)
    tickers_extracted: Mapped[dict | None] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    media_content: Mapped[MediaContent] = relationship(back_populates="sentiment_analyses")

    __table_args__ = (
        UniqueConstraint(
            "media_content_id", "model_name", name="uq_sentiment_content_model"
        ),
    )


class MemberMediaMention(Base):
    """Join table linking congress members to media content where they appear."""

    __tablename__ = "member_media_mention"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    member_bioguide_id: Mapped[str] = mapped_column(
        String(10), ForeignKey("congress_member.bioguide_id"), index=True
    )
    media_content_id: Mapped[int] = mapped_column(
        ForeignKey("media_content.id", ondelete="CASCADE"), index=True
    )
    mention_type: Mapped[str | None] = mapped_column(String(50))
    context_snippet: Mapped[str | None] = mapped_column(Text)

    media_content: Mapped[MediaContent] = relationship(back_populates="member_mentions")
