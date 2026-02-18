"""Campaign finance models from FEC bulk data."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Integer, Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.postgres import Base


class CampaignCommittee(Base):
    """A political campaign committee registered with the FEC."""

    __tablename__ = "campaign_committee"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    fec_committee_id: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(300))
    committee_type: Mapped[str | None] = mapped_column(String(10))  # H/S/P/etc.
    party: Mapped[str | None] = mapped_column(String(10))
    state: Mapped[str | None] = mapped_column(String(2))

    # Link to the candidate (congress member)
    candidate_fec_id: Mapped[str | None] = mapped_column(String(20), index=True)
    member_bioguide_id: Mapped[str | None] = mapped_column(
        String(10), index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    contributions: Mapped[list[CampaignContribution]] = relationship(
        back_populates="committee"
    )


class CampaignContribution(Base):
    """An individual campaign contribution from FEC records.

    We focus on contributions from employees of companies whose stock
    is traded by congress members — this forms a key network edge.
    """

    __tablename__ = "campaign_contribution"
    __table_args__ = (
        UniqueConstraint(
            "fec_transaction_id",
            name="uq_campaign_contribution_txn",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    fec_transaction_id: Mapped[str | None] = mapped_column(String(50), index=True)
    committee_id: Mapped[int | None] = mapped_column(
        ForeignKey("campaign_committee.id"), index=True
    )

    # Contributor info
    contributor_name: Mapped[str] = mapped_column(String(300))
    contributor_employer: Mapped[str | None] = mapped_column(String(300), index=True)
    contributor_occupation: Mapped[str | None] = mapped_column(String(300))
    contributor_city: Mapped[str | None] = mapped_column(String(100))
    contributor_state: Mapped[str | None] = mapped_column(String(2))
    contributor_zip: Mapped[str | None] = mapped_column(String(10))

    # Transaction info
    amount: Mapped[float] = mapped_column(Numeric(12, 2))
    contribution_date: Mapped[date | None] = mapped_column(Date, index=True)
    transaction_type: Mapped[str | None] = mapped_column(String(10))  # 15/15E/etc.

    # Entity resolution: matched employer to ticker
    matched_ticker: Mapped[str | None] = mapped_column(String(20), index=True)
    match_confidence: Mapped[float | None] = mapped_column(Numeric(5, 4))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    committee: Mapped[CampaignCommittee | None] = relationship(
        back_populates="contributions"
    )
