"""Database loading utilities for ingested data."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.trade import TradeDisclosure
from src.models.financial import StockDaily
from src.models.member import CongressMember
from src.models.legislation import Bill, BillCosponsor, Committee, CommitteeHearing, VoteRecord
from src.models.lobbying import LobbyingFiling, LobbyingRegistrant, LobbyingClient, LobbyingLobbyist
from src.models.campaign_finance import CampaignCommittee, CampaignContribution
from src.models.media import MediaContent, SentimentAnalysis, MemberMediaMention

logger = logging.getLogger(__name__)


async def upsert_trades(session: AsyncSession, records: list[dict[str, Any]]) -> int:
    """Upsert trade disclosure records. Returns count of new records inserted."""
    if not records:
        return 0

    inserted = 0
    for record in records:
        stmt = pg_insert(TradeDisclosure.__table__).values(**record)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=[
                "member_name", "ticker", "transaction_date",
                "transaction_type", "amount_range_low",
            ]
        )
        result = await session.execute(stmt)
        inserted += result.rowcount

    await session.commit()
    logger.info("Upserted trades: %d new out of %d total", inserted, len(records))
    return inserted


async def upsert_stock_daily(session: AsyncSession, records: list[dict[str, Any]]) -> int:
    """Upsert stock daily records."""
    if not records:
        return 0

    inserted = 0
    for record in records:
        stmt = pg_insert(StockDaily.__table__).values(**record)
        stmt = stmt.on_conflict_do_nothing(index_elements=["ticker", "date"])
        result = await session.execute(stmt)
        inserted += result.rowcount

    await session.commit()
    logger.info("Upserted stock data: %d new out of %d total", inserted, len(records))
    return inserted


async def get_unique_tickers(session: AsyncSession) -> list[str]:
    """Get all unique tickers from trade disclosures."""
    result = await session.execute(
        select(TradeDisclosure.ticker)
        .where(TradeDisclosure.ticker.is_not(None))
        .distinct()
    )
    return [row[0] for row in result.all()]


async def get_existing_filing_urls(session: AsyncSession, source: str) -> set[str]:
    """Get all filing_url values for a given source. Used for incremental collection."""
    result = await session.execute(
        select(TradeDisclosure.filing_url)
        .where(TradeDisclosure.source == source)
        .where(TradeDisclosure.filing_url.is_not(None))
        .distinct()
    )
    return {row[0] for row in result.all()}


# ---------------------------------------------------------------------------
# Phase 2: Legislation loaders
# ---------------------------------------------------------------------------


async def upsert_members(session: AsyncSession, records: list[dict[str, Any]]) -> int:
    """Upsert congress member records from Congress.gov or Voteview."""
    if not records:
        return 0

    inserted = 0
    for record in records:
        raw_data = record.pop("raw_data", None)
        stmt = pg_insert(CongressMember.__table__).values(**record)
        stmt = stmt.on_conflict_do_update(
            index_elements=["bioguide_id"],
            set_={
                k: stmt.excluded[k]
                for k in record
                if k != "bioguide_id"
            },
        )
        result = await session.execute(stmt)
        inserted += result.rowcount

    await session.commit()
    logger.info("Upserted members: %d processed out of %d total", inserted, len(records))
    return inserted


async def update_member_ideology(
    session: AsyncSession, records: list[dict[str, Any]]
) -> int:
    """Update nominate_dim1/dim2 on CongressMember from Voteview data."""
    if not records:
        return 0

    updated = 0
    for record in records:
        bioguide_id = record.get("bioguide_id")
        if not bioguide_id:
            continue

        stmt = pg_insert(CongressMember.__table__).values(
            bioguide_id=bioguide_id,
            full_name=record.get("full_name", ""),
            chamber=record.get("chamber", "house"),
            party=record.get("party"),
            state=record.get("state"),
            nominate_dim1=record.get("nominate_dim1"),
            nominate_dim2=record.get("nominate_dim2"),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["bioguide_id"],
            set_={
                "nominate_dim1": stmt.excluded.nominate_dim1,
                "nominate_dim2": stmt.excluded.nominate_dim2,
            },
        )
        result = await session.execute(stmt)
        updated += result.rowcount

    await session.commit()
    logger.info("Updated ideology scores for %d members", updated)
    return updated


async def upsert_bills(session: AsyncSession, records: list[dict[str, Any]]) -> int:
    """Upsert bill records."""
    if not records:
        return 0

    inserted = 0
    for record in records:
        record.pop("raw_data", None)
        stmt = pg_insert(Bill.__table__).values(**record)
        stmt = stmt.on_conflict_do_update(
            index_elements=["bill_id"],
            set_={
                "status": stmt.excluded.status,
                "latest_action_date": stmt.excluded.latest_action_date,
                "latest_action_text": stmt.excluded.latest_action_text,
                "subjects": stmt.excluded.subjects,
                "committees": stmt.excluded.committees,
                "actions": stmt.excluded.actions,
            },
        )
        result = await session.execute(stmt)
        inserted += result.rowcount

    await session.commit()
    logger.info("Upserted bills: %d processed out of %d total", inserted, len(records))
    return inserted


async def upsert_committees(session: AsyncSession, records: list[dict[str, Any]]) -> int:
    """Upsert committee records."""
    if not records:
        return 0

    inserted = 0
    for record in records:
        stmt = pg_insert(Committee.__table__).values(**record)
        stmt = stmt.on_conflict_do_update(
            index_elements=["system_code"],
            set_={
                "name": stmt.excluded.name,
                "chamber": stmt.excluded.chamber,
                "is_current": stmt.excluded.is_current,
            },
        )
        result = await session.execute(stmt)
        inserted += result.rowcount

    await session.commit()
    logger.info("Upserted committees: %d processed out of %d total", inserted, len(records))
    return inserted


async def upsert_hearings(session: AsyncSession, records: list[dict[str, Any]]) -> int:
    """Upsert committee hearing records."""
    if not records:
        return 0

    inserted = 0
    for record in records:
        stmt = pg_insert(CommitteeHearing.__table__).values(**record)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["committee_code", "title", "hearing_date"]
        )
        result = await session.execute(stmt)
        inserted += result.rowcount

    await session.commit()
    logger.info("Upserted hearings: %d new out of %d total", inserted, len(records))
    return inserted


# ---------------------------------------------------------------------------
# Phase 3: Network (lobbying + campaign finance) loaders
# ---------------------------------------------------------------------------


async def upsert_lobbying_filings(
    session: AsyncSession, records: list[dict[str, Any]]
) -> int:
    """Upsert lobbying filing records with nested registrant, client, and lobbyists.

    Each record is expected to have:
    - filing_uuid, filing_type, filing_year, etc. (filing-level fields)
    - registrant: {senate_id, name, description}
    - client: {name, description, country, state}
    - lobbyists: [{name, covered_position, is_former_congress, is_former_executive}]
    """
    if not records:
        return 0

    inserted = 0
    for record in records:
        # 1. Upsert registrant
        registrant_data = record.pop("registrant", {})
        registrant_id = None
        if registrant_data and registrant_data.get("senate_id"):
            reg_stmt = pg_insert(LobbyingRegistrant.__table__).values(
                senate_id=registrant_data["senate_id"],
                name=registrant_data.get("name", ""),
                description=registrant_data.get("description"),
            )
            reg_stmt = reg_stmt.on_conflict_do_update(
                index_elements=["senate_id"],
                set_={"name": reg_stmt.excluded.name},
            )
            await session.execute(reg_stmt)

            # Get the registrant ID
            reg_result = await session.execute(
                select(LobbyingRegistrant.id).where(
                    LobbyingRegistrant.senate_id == registrant_data["senate_id"]
                )
            )
            registrant_id = reg_result.scalar_one_or_none()

        # 2. Upsert client
        client_data = record.pop("client", {})
        client_id = None
        if client_data and client_data.get("name"):
            # Check if client exists by name
            existing = await session.execute(
                select(LobbyingClient.id).where(
                    LobbyingClient.name == client_data["name"]
                )
            )
            client_id = existing.scalar_one_or_none()

            if not client_id:
                client_stmt = pg_insert(LobbyingClient.__table__).values(
                    name=client_data["name"],
                    description=client_data.get("description"),
                    country=client_data.get("country"),
                    state=client_data.get("state"),
                )
                client_stmt = client_stmt.on_conflict_do_nothing()
                await session.execute(client_stmt)

                new_client = await session.execute(
                    select(LobbyingClient.id).where(
                        LobbyingClient.name == client_data["name"]
                    )
                )
                client_id = new_client.scalar_one_or_none()

        # 3. Upsert the filing itself
        lobbyists_data = record.pop("lobbyists", [])
        filing_values = {
            k: v for k, v in record.items()
            if k in {
                "filing_uuid", "filing_type", "filing_year", "filing_period",
                "filing_date", "amount", "registrant_name", "client_name",
                "specific_issues", "general_issue_codes", "government_entities",
                "lobbied_bills",
            }
        }
        filing_values["registrant_id"] = registrant_id
        filing_values["client_id"] = client_id
        filing_values["registrant_name"] = registrant_data.get("name")
        filing_values["client_name"] = client_data.get("name")

        stmt = pg_insert(LobbyingFiling.__table__).values(**filing_values)
        stmt = stmt.on_conflict_do_nothing(index_elements=["filing_uuid"])
        result = await session.execute(stmt)
        inserted += result.rowcount

        # 4. Get filing ID and insert lobbyists
        if lobbyists_data:
            filing_result = await session.execute(
                select(LobbyingFiling.id).where(
                    LobbyingFiling.filing_uuid == record.get("filing_uuid")
                )
            )
            filing_id = filing_result.scalar_one_or_none()

            if filing_id:
                for lob in lobbyists_data:
                    lob_stmt = pg_insert(LobbyingLobbyist.__table__).values(
                        filing_id=filing_id,
                        name=lob.get("name", "Unknown"),
                        covered_position=lob.get("covered_position"),
                        is_former_congress=lob.get("is_former_congress", False),
                        is_former_executive=lob.get("is_former_executive", False),
                    )
                    lob_stmt = lob_stmt.on_conflict_do_nothing()
                    await session.execute(lob_stmt)

    await session.commit()
    logger.info("Upserted lobbying filings: %d new out of %d total", inserted, len(records))
    return inserted


async def upsert_campaign_committees(
    session: AsyncSession, records: list[dict[str, Any]]
) -> int:
    """Upsert FEC campaign committee records."""
    if not records:
        return 0

    inserted = 0
    for record in records:
        stmt = pg_insert(CampaignCommittee.__table__).values(**record)
        stmt = stmt.on_conflict_do_update(
            index_elements=["fec_committee_id"],
            set_={
                "name": stmt.excluded.name,
                "committee_type": stmt.excluded.committee_type,
                "party": stmt.excluded.party,
                "candidate_fec_id": stmt.excluded.candidate_fec_id,
            },
        )
        result = await session.execute(stmt)
        inserted += result.rowcount

    await session.commit()
    logger.info("Upserted campaign committees: %d processed", inserted)
    return inserted


async def upsert_campaign_contributions(
    session: AsyncSession,
    records: list[dict[str, Any]],
    committee_fec_id_map: dict[str, int] | None = None,
) -> int:
    """Upsert FEC contribution records.

    committee_fec_id_map maps FEC committee IDs to our internal committee PKs.
    """
    if not records:
        return 0

    # Build committee map if not provided
    if committee_fec_id_map is None:
        result = await session.execute(
            select(CampaignCommittee.fec_committee_id, CampaignCommittee.id)
        )
        committee_fec_id_map = {row[0]: row[1] for row in result.all()}

    inserted = 0
    for record in records:
        fec_committee_id = record.pop("fec_committee_id", None)
        if fec_committee_id:
            record["committee_id"] = committee_fec_id_map.get(fec_committee_id)

        fec_txn_id = record.get("fec_transaction_id")
        if not fec_txn_id:
            continue

        stmt = pg_insert(CampaignContribution.__table__).values(**record)
        stmt = stmt.on_conflict_do_nothing(constraint="uq_campaign_contribution_txn")
        result = await session.execute(stmt)
        inserted += result.rowcount

    await session.commit()
    logger.info("Upserted contributions: %d new out of %d total", inserted, len(records))
    return inserted


# ---------------------------------------------------------------------------
# Phase 4: Media content loaders
# ---------------------------------------------------------------------------


async def upsert_media_content(
    session: AsyncSession, records: list[dict[str, Any]]
) -> int:
    """Upsert media content records. Dedup on (source_type, source_id)."""
    if not records:
        return 0

    inserted = 0
    for record in records:
        stmt = pg_insert(MediaContent.__table__).values(**record)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_media_content_source",
            set_={
                "title": stmt.excluded.title,
                "content": stmt.excluded.content,
                "summary": stmt.excluded.summary,
                "url": stmt.excluded.url,
                "author": stmt.excluded.author,
                "published_date": stmt.excluded.published_date,
                "member_bioguide_ids": stmt.excluded.member_bioguide_ids,
                "tickers_mentioned": stmt.excluded.tickers_mentioned,
                "raw_metadata": stmt.excluded.raw_metadata,
            },
        )
        result = await session.execute(stmt)
        inserted += result.rowcount

    await session.commit()
    logger.info("Upserted media content: %d processed out of %d total", inserted, len(records))
    return inserted


async def upsert_sentiment_analyses(
    session: AsyncSession, records: list[dict[str, Any]]
) -> int:
    """Upsert sentiment analysis results. Dedup on (media_content_id, model_name)."""
    if not records:
        return 0

    inserted = 0
    for record in records:
        stmt = pg_insert(SentimentAnalysis.__table__).values(**record)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_sentiment_content_model",
            set_={
                "sentiment_label": stmt.excluded.sentiment_label,
                "sentiment_score": stmt.excluded.sentiment_score,
                "confidence": stmt.excluded.confidence,
                "entities": stmt.excluded.entities,
                "sectors": stmt.excluded.sectors,
                "tickers_extracted": stmt.excluded.tickers_extracted,
            },
        )
        result = await session.execute(stmt)
        inserted += result.rowcount

    await session.commit()
    logger.info("Upserted sentiment analyses: %d processed out of %d total", inserted, len(records))
    return inserted


async def upsert_member_media_mentions(
    session: AsyncSession, records: list[dict[str, Any]]
) -> int:
    """Insert member-media mention links. Skips duplicates."""
    if not records:
        return 0

    inserted = 0
    for record in records:
        stmt = pg_insert(MemberMediaMention.__table__).values(**record)
        stmt = stmt.on_conflict_do_nothing()
        result = await session.execute(stmt)
        inserted += result.rowcount

    await session.commit()
    logger.info("Upserted member media mentions: %d new out of %d total", inserted, len(records))
    return inserted
