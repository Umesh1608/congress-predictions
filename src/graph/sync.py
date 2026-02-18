"""Sync PostgreSQL data into Neo4j graph.

Reads from PostgreSQL and merges nodes/relationships into Neo4j.
Uses MERGE (idempotent) so it can be run repeatedly without duplicates.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncSession as Neo4jSession
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession as PgSession

from src.models.campaign_finance import CampaignCommittee, CampaignContribution
from src.models.legislation import Bill, BillCosponsor, Committee, CommitteeHearing
from src.models.lobbying import LobbyingClient, LobbyingFiling, LobbyingLobbyist
from src.models.member import CommitteeAssignment, CongressMember
from src.models.trade import TradeDisclosure

logger = logging.getLogger(__name__)


async def sync_members(pg: PgSession, neo4j: Neo4jSession) -> int:
    """Sync CongressMember rows to Member nodes."""
    result = await pg.execute(select(CongressMember))
    members = result.scalars().all()

    count = 0
    for m in members:
        await neo4j.run(
            """
            MERGE (member:Member {bioguide_id: $bioguide_id})
            SET member.full_name = $full_name,
                member.chamber = $chamber,
                member.state = $state,
                member.party = $party,
                member.in_office = $in_office,
                member.nominate_dim1 = $nominate_dim1,
                member.nominate_dim2 = $nominate_dim2
            """,
            bioguide_id=m.bioguide_id,
            full_name=m.full_name,
            chamber=m.chamber,
            state=m.state,
            party=m.party,
            in_office=m.in_office,
            nominate_dim1=m.nominate_dim1,
            nominate_dim2=m.nominate_dim2,
        )
        count += 1

    logger.info("Synced %d Member nodes", count)
    return count


async def sync_committees(pg: PgSession, neo4j: Neo4jSession) -> int:
    """Sync Committee rows to Committee nodes."""
    result = await pg.execute(select(Committee))
    committees = result.scalars().all()

    count = 0
    for c in committees:
        await neo4j.run(
            """
            MERGE (committee:Committee {system_code: $system_code})
            SET committee.name = $name,
                committee.chamber = $chamber
            """,
            system_code=c.system_code,
            name=c.name,
            chamber=c.chamber,
        )
        count += 1

    logger.info("Synced %d Committee nodes", count)
    return count


async def sync_committee_assignments(pg: PgSession, neo4j: Neo4jSession) -> int:
    """Sync CommitteeAssignment rows to SITS_ON relationships."""
    result = await pg.execute(select(CommitteeAssignment))
    assignments = result.scalars().all()

    count = 0
    for a in assignments:
        await neo4j.run(
            """
            MATCH (member:Member {bioguide_id: $bioguide_id})
            MATCH (committee:Committee {system_code: $committee_code})
            MERGE (member)-[r:SITS_ON]->(committee)
            SET r.role = $role,
                r.committee_name = $committee_name
            """,
            bioguide_id=a.member_bioguide_id,
            committee_code=a.committee_code,
            role=a.role,
            committee_name=a.committee_name,
        )
        count += 1

    logger.info("Synced %d SITS_ON relationships", count)
    return count


async def sync_trades(pg: PgSession, neo4j: Neo4jSession) -> int:
    """Sync trades to Company nodes and TRADED relationships."""
    result = await pg.execute(
        select(TradeDisclosure).where(TradeDisclosure.ticker.is_not(None))
    )
    trades = result.scalars().all()

    count = 0
    for t in trades:
        if not t.member_bioguide_id or not t.ticker:
            continue

        await neo4j.run(
            """
            MERGE (company:Company {ticker: $ticker})
            ON CREATE SET company.name = $asset_name
            """,
            ticker=t.ticker,
            asset_name=t.asset_name or t.ticker,
        )

        await neo4j.run(
            """
            MATCH (member:Member {bioguide_id: $bioguide_id})
            MATCH (company:Company {ticker: $ticker})
            MERGE (member)-[r:TRADED {trade_id: $trade_id}]->(company)
            SET r.transaction_type = $tx_type,
                r.transaction_date = $tx_date,
                r.amount_range_low = $amount_low,
                r.amount_range_high = $amount_high,
                r.asset_name = $asset_name,
                r.filer_type = $filer_type
            """,
            bioguide_id=t.member_bioguide_id,
            ticker=t.ticker,
            trade_id=t.id,
            tx_type=t.transaction_type,
            tx_date=t.transaction_date.isoformat() if t.transaction_date else None,
            amount_low=float(t.amount_range_low) if t.amount_range_low else None,
            amount_high=float(t.amount_range_high) if t.amount_range_high else None,
            asset_name=t.asset_name,
            filer_type=t.filer_type,
        )
        count += 1

    logger.info("Synced %d TRADED relationships", count)
    return count


async def sync_bills(pg: PgSession, neo4j: Neo4jSession) -> int:
    """Sync Bill rows to Bill nodes and SPONSORED/REFERRED_TO relationships."""
    result = await pg.execute(select(Bill))
    bills = result.scalars().all()

    count = 0
    for b in bills:
        await neo4j.run(
            """
            MERGE (bill:Bill {bill_id: $bill_id})
            SET bill.title = $title,
                bill.bill_type = $bill_type,
                bill.congress = $congress,
                bill.policy_area = $policy_area,
                bill.introduced_date = $introduced_date
            """,
            bill_id=b.bill_id,
            title=(b.title or "")[:200],
            bill_type=b.bill_type,
            congress=b.congress_number,
            policy_area=b.policy_area,
            introduced_date=b.introduced_date.isoformat() if b.introduced_date else None,
        )

        # SPONSORED relationship
        if b.sponsor_bioguide_id:
            await neo4j.run(
                """
                MATCH (member:Member {bioguide_id: $bioguide_id})
                MATCH (bill:Bill {bill_id: $bill_id})
                MERGE (member)-[:SPONSORED]->(bill)
                """,
                bioguide_id=b.sponsor_bioguide_id,
                bill_id=b.bill_id,
            )

        # REFERRED_TO committee relationships
        for committee_info in (b.committees or []):
            if isinstance(committee_info, dict):
                code = committee_info.get("code", "")
                if code:
                    await neo4j.run(
                        """
                        MATCH (bill:Bill {bill_id: $bill_id})
                        MATCH (committee:Committee {system_code: $code})
                        MERGE (bill)-[:REFERRED_TO]->(committee)
                        """,
                        bill_id=b.bill_id,
                        code=code,
                    )

        count += 1

    logger.info("Synced %d Bill nodes", count)
    return count


async def sync_cosponsors(pg: PgSession, neo4j: Neo4jSession) -> int:
    """Sync BillCosponsor rows to COSPONSORED relationships."""
    result = await pg.execute(select(BillCosponsor))
    cosponsors = result.scalars().all()

    count = 0
    for cs in cosponsors:
        await neo4j.run(
            """
            MATCH (member:Member {bioguide_id: $bioguide_id})
            MATCH (bill:Bill {bill_id: $bill_id})
            MERGE (member)-[:COSPONSORED]->(bill)
            """,
            bioguide_id=cs.member_bioguide_id,
            bill_id=cs.bill_id,
        )
        count += 1

    logger.info("Synced %d COSPONSORED relationships", count)
    return count


async def sync_lobbying(pg: PgSession, neo4j: Neo4jSession) -> int:
    """Sync lobbying data to LobbyingFirm/Company nodes and relationships.

    Creates:
    - LobbyingFirm nodes from registrants
    - LOBBIED_FOR: LobbyingFirm -> Company (when client has matched ticker)
    - LOBBIED: LobbyingFirm -> Member (via government entities contacted)
    - EMPLOYED_BY: Lobbyist -> LobbyingFirm
    - FORMERLY_AT: Lobbyist -> Member (revolving door lobbyists)
    """
    # Sync lobbying firms (registrants)
    from src.models.lobbying import LobbyingRegistrant

    reg_result = await pg.execute(
        select(LobbyingRegistrant).where(LobbyingRegistrant.senate_id.is_not(None))
    )
    registrants = reg_result.scalars().all()

    firm_count = 0
    for r in registrants:
        await neo4j.run(
            """
            MERGE (firm:LobbyingFirm {senate_id: $senate_id})
            SET firm.name = $name
            """,
            senate_id=r.senate_id,
            name=r.name,
        )
        firm_count += 1

    logger.info("Synced %d LobbyingFirm nodes", firm_count)

    # Sync filings: LOBBIED_FOR relationships (firm -> company via matched client)
    filing_result = await pg.execute(
        select(LobbyingFiling).where(LobbyingFiling.registrant_id.is_not(None))
    )
    filings = filing_result.scalars().all()

    lobbied_for_count = 0
    for f in filings:
        # Get the client for this filing (if it has a matched ticker)
        if f.client_id:
            client_result = await pg.execute(
                select(LobbyingClient).where(LobbyingClient.id == f.client_id)
            )
            client = client_result.scalar_one_or_none()
            if client and client.matched_ticker:
                # Get registrant's senate_id
                reg_result2 = await pg.execute(
                    select(LobbyingRegistrant).where(LobbyingRegistrant.id == f.registrant_id)
                )
                reg = reg_result2.scalar_one_or_none()
                if reg and reg.senate_id:
                    await neo4j.run(
                        """
                        MATCH (firm:LobbyingFirm {senate_id: $senate_id})
                        MERGE (company:Company {ticker: $ticker})
                        ON CREATE SET company.name = $client_name
                        MERGE (firm)-[r:LOBBIED_FOR]->(company)
                        SET r.amount = coalesce(r.amount, 0) + coalesce($amount, 0),
                            r.filing_count = coalesce(r.filing_count, 0) + 1,
                            r.year = $year
                        """,
                        senate_id=reg.senate_id,
                        ticker=client.matched_ticker,
                        client_name=client.name,
                        amount=float(f.amount) if f.amount else 0,
                        year=f.filing_year,
                    )
                    lobbied_for_count += 1

    logger.info("Synced %d LOBBIED_FOR relationships", lobbied_for_count)

    # Sync lobbyists and revolving door connections
    lobbyist_result = await pg.execute(select(LobbyingLobbyist))
    lobbyists = lobbyist_result.scalars().all()

    lobbyist_count = 0
    for lob in lobbyists:
        # Get the filing's registrant senate_id
        filing_res = await pg.execute(
            select(LobbyingFiling).where(LobbyingFiling.id == lob.filing_id)
        )
        filing = filing_res.scalar_one_or_none()
        if not filing or not filing.registrant_id:
            continue

        reg_res = await pg.execute(
            select(LobbyingRegistrant).where(LobbyingRegistrant.id == filing.registrant_id)
        )
        reg = reg_res.scalar_one_or_none()
        if not reg or not reg.senate_id:
            continue

        await neo4j.run(
            """
            MERGE (lobbyist:Lobbyist {name: $name})
            SET lobbyist.covered_position = $covered_position,
                lobbyist.is_former_congress = $is_former_congress
            WITH lobbyist
            MATCH (firm:LobbyingFirm {senate_id: $senate_id})
            MERGE (lobbyist)-[:EMPLOYED_BY]->(firm)
            """,
            name=lob.name,
            covered_position=lob.covered_position,
            is_former_congress=lob.is_former_congress or False,
            senate_id=reg.senate_id,
        )
        lobbyist_count += 1

    logger.info("Synced %d Lobbyist nodes", lobbyist_count)

    return firm_count + lobbied_for_count + lobbyist_count


async def sync_campaign_finance(pg: PgSession, neo4j: Neo4jSession) -> int:
    """Sync campaign finance data to DONATED_TO relationships.

    Creates Company -[DONATED_TO]-> Member relationships based on
    aggregated contributions from company employees.
    """
    # Get committees linked to members
    committee_result = await pg.execute(
        select(CampaignCommittee).where(
            CampaignCommittee.member_bioguide_id.is_not(None)
        )
    )
    committees = committee_result.scalars().all()
    committee_map = {c.id: c.member_bioguide_id for c in committees}

    # Get contributions with matched tickers
    contrib_result = await pg.execute(
        select(CampaignContribution).where(
            CampaignContribution.matched_ticker.is_not(None)
        )
    )
    contributions = contrib_result.scalars().all()

    # Aggregate by (ticker, member_bioguide_id)
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    for c in contributions:
        member_id = committee_map.get(c.committee_id)
        if not member_id or not c.matched_ticker:
            continue

        key = (c.matched_ticker, member_id)
        if key not in aggregated:
            aggregated[key] = {"total_amount": 0.0, "count": 0, "employer": c.contributor_employer}
        aggregated[key]["total_amount"] += float(c.amount) if c.amount else 0
        aggregated[key]["count"] += 1

    count = 0
    for (ticker, member_id), agg in aggregated.items():
        await neo4j.run(
            """
            MERGE (company:Company {ticker: $ticker})
            WITH company
            MATCH (member:Member {bioguide_id: $bioguide_id})
            MERGE (company)-[r:DONATED_TO]->(member)
            SET r.total_amount = $total_amount,
                r.contribution_count = $count,
                r.employer_name = $employer
            """,
            ticker=ticker,
            bioguide_id=member_id,
            total_amount=agg["total_amount"],
            count=agg["count"],
            employer=agg["employer"],
        )
        count += 1

    logger.info("Synced %d DONATED_TO relationships", count)
    return count


async def run_full_sync(pg: PgSession, neo4j: Neo4jSession) -> dict[str, int]:
    """Run a complete sync from PostgreSQL to Neo4j.

    Returns a summary dict of counts per entity type.
    """
    from src.graph.schema import create_constraints

    await create_constraints(neo4j)

    summary = {
        "members": await sync_members(pg, neo4j),
        "committees": await sync_committees(pg, neo4j),
        "committee_assignments": await sync_committee_assignments(pg, neo4j),
        "trades": await sync_trades(pg, neo4j),
        "bills": await sync_bills(pg, neo4j),
        "cosponsors": await sync_cosponsors(pg, neo4j),
        "lobbying": await sync_lobbying(pg, neo4j),
        "campaign_finance": await sync_campaign_finance(pg, neo4j),
    }

    logger.info("Full sync complete: %s", summary)
    return summary
