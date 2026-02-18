from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Date, DateTime, ForeignKey, Index, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.postgres import Base


class TradeDisclosure(Base):
    __tablename__ = "trade_disclosure"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    member_bioguide_id: Mapped[str | None] = mapped_column(
        ForeignKey("congress_member.bioguide_id"), index=True
    )
    member_name: Mapped[str] = mapped_column(String(200))
    filer_type: Mapped[str] = mapped_column(
        String(50), default="member"
    )  # member / spouse / dependent / joint
    ticker: Mapped[str | None] = mapped_column(String(20), index=True)
    asset_name: Mapped[str] = mapped_column(String(500))
    asset_type: Mapped[str | None] = mapped_column(String(100))
    transaction_type: Mapped[str] = mapped_column(String(50))  # purchase / sale / exchange
    transaction_date: Mapped[date] = mapped_column(Date, index=True)
    disclosure_date: Mapped[date | None] = mapped_column(Date, index=True)
    amount_range_low: Mapped[Decimal | None] = mapped_column(Numeric(15, 2))
    amount_range_high: Mapped[Decimal | None] = mapped_column(Numeric(15, 2))
    chamber: Mapped[str | None] = mapped_column(String(10))  # house / senate
    source: Mapped[str] = mapped_column(String(50))  # house_watcher / senate_watcher / fmp
    filing_url: Mapped[str | None] = mapped_column(Text)
    raw_data: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    member: Mapped[CongressMember | None] = relationship(
        back_populates="trades", foreign_keys=[member_bioguide_id]
    )

    __table_args__ = (
        Index(
            "ix_trade_dedup",
            "member_name",
            "ticker",
            "transaction_date",
            "transaction_type",
            "amount_range_low",
            unique=True,
        ),
    )
