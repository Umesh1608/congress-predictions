from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.postgres import Base


class Bill(Base):
    __tablename__ = "bill"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    bill_id: Mapped[str] = mapped_column(String(30), unique=True, index=True)  # e.g. "hr1234-118"
    congress_number: Mapped[int] = mapped_column(Integer, index=True)
    bill_type: Mapped[str] = mapped_column(String(10))  # hr, s, hjres, sjres, hconres, sconres
    bill_number: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(Text)
    short_title: Mapped[str | None] = mapped_column(String(500))
    introduced_date: Mapped[date | None] = mapped_column(Date, index=True)
    sponsor_bioguide_id: Mapped[str | None] = mapped_column(
        ForeignKey("congress_member.bioguide_id"), index=True
    )
    sponsor_name: Mapped[str | None] = mapped_column(String(200))
    status: Mapped[str | None] = mapped_column(String(100))  # latest action summary
    subjects: Mapped[dict | None] = mapped_column(JSONB, default=list)  # policy area + legislative subjects
    committees: Mapped[dict | None] = mapped_column(JSONB, default=list)  # committee codes assigned
    actions: Mapped[dict | None] = mapped_column(JSONB, default=list)  # legislative action history
    latest_action_date: Mapped[date | None] = mapped_column(Date)
    latest_action_text: Mapped[str | None] = mapped_column(Text)
    policy_area: Mapped[str | None] = mapped_column(String(200))  # top-level policy area
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    cosponsors: Mapped[list[BillCosponsor]] = relationship(back_populates="bill")

    __table_args__ = (
        Index("ix_bill_congress_type", "congress_number", "bill_type"),
    )


class BillCosponsor(Base):
    __tablename__ = "bill_cosponsor"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    bill_id: Mapped[int] = mapped_column(ForeignKey("bill.id"), index=True)
    member_bioguide_id: Mapped[str] = mapped_column(
        ForeignKey("congress_member.bioguide_id"), index=True
    )
    date_added: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    bill: Mapped[Bill] = relationship(back_populates="cosponsors")

    __table_args__ = (
        Index("ix_cosponsor_dedup", "bill_id", "member_bioguide_id", unique=True),
    )


class Committee(Base):
    __tablename__ = "committee"

    system_code: Mapped[str] = mapped_column(String(10), primary_key=True)  # e.g. HSAG00
    name: Mapped[str] = mapped_column(String(300))
    chamber: Mapped[str] = mapped_column(String(10))  # house / senate / joint
    parent_code: Mapped[str | None] = mapped_column(String(10))  # for subcommittees
    url: Mapped[str | None] = mapped_column(Text)
    is_current: Mapped[bool] = mapped_column(default=True)
    jurisdiction: Mapped[str | None] = mapped_column(Text)  # free-text jurisdiction description
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    hearings: Mapped[list[CommitteeHearing]] = relationship(back_populates="committee")


class CommitteeHearing(Base):
    __tablename__ = "committee_hearing"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    committee_code: Mapped[str | None] = mapped_column(
        ForeignKey("committee.system_code"), index=True
    )
    title: Mapped[str] = mapped_column(Text)
    hearing_date: Mapped[date | None] = mapped_column(Date, index=True)
    chamber: Mapped[str | None] = mapped_column(String(10))
    congress_number: Mapped[int | None] = mapped_column(Integer)
    url: Mapped[str | None] = mapped_column(Text)
    transcript_url: Mapped[str | None] = mapped_column(Text)
    related_bills: Mapped[dict | None] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    committee: Mapped[Committee | None] = relationship(back_populates="hearings")

    __table_args__ = (
        Index("ix_hearing_dedup", "committee_code", "title", "hearing_date", unique=True),
    )


class VoteRecord(Base):
    __tablename__ = "vote_record"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    member_bioguide_id: Mapped[str] = mapped_column(
        ForeignKey("congress_member.bioguide_id"), index=True
    )
    bill_id: Mapped[int | None] = mapped_column(ForeignKey("bill.id"), index=True)
    vote_date: Mapped[date] = mapped_column(Date, index=True)
    congress_number: Mapped[int] = mapped_column(Integer)
    roll_call_number: Mapped[int | None] = mapped_column(Integer)
    chamber: Mapped[str] = mapped_column(String(10))
    position: Mapped[str] = mapped_column(String(20))  # yea / nay / abstain / not_voting / present
    question: Mapped[str | None] = mapped_column(Text)  # what was being voted on
    result: Mapped[str | None] = mapped_column(String(50))  # passed / failed / agreed to
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index(
            "ix_vote_dedup",
            "member_bioguide_id", "vote_date", "roll_call_number", "chamber",
            unique=True,
        ),
    )
