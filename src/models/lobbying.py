"""Lobbying data models from Senate Lobbying Disclosure Act (LDA) filings."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.postgres import Base


class LobbyingFiling(Base):
    """A lobbying disclosure filing (LD-1 registration or LD-2 activity report)."""

    __tablename__ = "lobbying_filing"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filing_uuid: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    filing_type: Mapped[str] = mapped_column(String(50))  # registration / report
    filing_year: Mapped[int] = mapped_column(Integer)
    filing_period: Mapped[str | None] = mapped_column(String(20))  # Q1/Q2/H1/H2/etc.
    filing_date: Mapped[date | None] = mapped_column(Date)
    amount: Mapped[float | None] = mapped_column(Numeric(15, 2))

    # Registrant (the lobbying firm)
    registrant_id: Mapped[int | None] = mapped_column(ForeignKey("lobbying_registrant.id"))
    registrant_name: Mapped[str | None] = mapped_column(String(300))

    # Client (the company/org being represented)
    client_id: Mapped[int | None] = mapped_column(ForeignKey("lobbying_client.id"))
    client_name: Mapped[str | None] = mapped_column(String(300))

    # Issues lobbied on
    specific_issues: Mapped[dict | None] = mapped_column(JSONB)
    general_issue_codes: Mapped[list | None] = mapped_column(JSONB)

    # Government entities contacted
    government_entities: Mapped[list | None] = mapped_column(JSONB)

    # Bills lobbied on (list of bill IDs)
    lobbied_bills: Mapped[list | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    registrant: Mapped[LobbyingRegistrant | None] = relationship(back_populates="filings")
    client: Mapped[LobbyingClient | None] = relationship(back_populates="filings")
    lobbyists: Mapped[list[LobbyingLobbyist]] = relationship(back_populates="filing")


class LobbyingRegistrant(Base):
    """A lobbying firm or self-filing organization."""

    __tablename__ = "lobbying_registrant"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    senate_id: Mapped[str | None] = mapped_column(String(50), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(300), index=True)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    filings: Mapped[list[LobbyingFiling]] = relationship(back_populates="registrant")


class LobbyingClient(Base):
    """An organization that retains a lobbying firm (or self-lobbies)."""

    __tablename__ = "lobbying_client"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(300), index=True)
    normalized_name: Mapped[str | None] = mapped_column(String(300), index=True)
    description: Mapped[str | None] = mapped_column(Text)
    country: Mapped[str | None] = mapped_column(String(100))
    state: Mapped[str | None] = mapped_column(String(2))

    # Entity resolution: matched ticker and company info
    matched_ticker: Mapped[str | None] = mapped_column(String(20), index=True)
    match_confidence: Mapped[float | None] = mapped_column(Numeric(5, 4))
    match_method: Mapped[str | None] = mapped_column(String(50))  # fuzzy / manual / edgar

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    filings: Mapped[list[LobbyingFiling]] = relationship(back_populates="client")


class LobbyingLobbyist(Base):
    """An individual lobbyist named on a filing."""

    __tablename__ = "lobbying_lobbyist"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filing_id: Mapped[int] = mapped_column(ForeignKey("lobbying_filing.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    covered_position: Mapped[str | None] = mapped_column(Text)  # former government position
    is_former_congress: Mapped[bool | None] = mapped_column(default=False)
    is_former_executive: Mapped[bool | None] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    filing: Mapped[LobbyingFiling] = relationship(back_populates="lobbyists")
