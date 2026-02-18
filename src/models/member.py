from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.postgres import Base


class CongressMember(Base):
    __tablename__ = "congress_member"

    bioguide_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    full_name: Mapped[str] = mapped_column(String(200))
    first_name: Mapped[str | None] = mapped_column(String(100))
    last_name: Mapped[str | None] = mapped_column(String(100))
    chamber: Mapped[str] = mapped_column(String(10))  # house / senate
    state: Mapped[str | None] = mapped_column(String(2))
    district: Mapped[str | None] = mapped_column(String(10))
    party: Mapped[str | None] = mapped_column(String(50))
    in_office: Mapped[bool] = mapped_column(Boolean, default=True)
    first_elected: Mapped[int | None] = mapped_column()
    social_accounts: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    # DW-NOMINATE ideology scores from Voteview
    nominate_dim1: Mapped[float | None] = mapped_column(Float)  # economic liberal-conservative
    nominate_dim2: Mapped[float | None] = mapped_column(Float)  # social conservative-liberal
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    trades: Mapped[list[TradeDisclosure]] = relationship(
        back_populates="member", foreign_keys="TradeDisclosure.member_bioguide_id"
    )
    family_members: Mapped[list[MemberFamily]] = relationship(back_populates="member")
    staff: Mapped[list[MemberStaff]] = relationship(back_populates="member")
    committee_assignments: Mapped[list[CommitteeAssignment]] = relationship(
        back_populates="member"
    )


class MemberFamily(Base):
    __tablename__ = "member_family"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    member_bioguide_id: Mapped[str] = mapped_column(
        ForeignKey("congress_member.bioguide_id"), index=True
    )
    name: Mapped[str] = mapped_column(String(200))
    relationship_type: Mapped[str] = mapped_column(String(50))  # spouse / dependent
    known_employers: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    member: Mapped[CongressMember] = relationship(back_populates="family_members")


class MemberStaff(Base):
    __tablename__ = "member_staff"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    member_bioguide_id: Mapped[str] = mapped_column(
        ForeignKey("congress_member.bioguide_id"), index=True
    )
    name: Mapped[str] = mapped_column(String(200))
    title: Mapped[str | None] = mapped_column(String(200))
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    subsequent_employer: Mapped[str | None] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    member: Mapped[CongressMember] = relationship(back_populates="staff")


class CommitteeAssignment(Base):
    __tablename__ = "committee_assignment"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    member_bioguide_id: Mapped[str] = mapped_column(
        ForeignKey("congress_member.bioguide_id"), index=True
    )
    committee_code: Mapped[str] = mapped_column(String(20))
    committee_name: Mapped[str] = mapped_column(String(300))
    role: Mapped[str | None] = mapped_column(String(50))  # chair / ranking / member
    congress_number: Mapped[int | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    member: Mapped[CongressMember] = relationship(back_populates="committee_assignments")
