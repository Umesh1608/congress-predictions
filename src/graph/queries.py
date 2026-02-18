"""Cypher query templates for Neo4j graph analysis.

These queries power the network analysis features that differentiate this system
from existing platforms. They find hidden connections between congress members
and companies through lobbying, donations, committee assignments, and trades.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncSession

logger = logging.getLogger(__name__)


async def get_member_network(
    session: AsyncSession,
    bioguide_id: str,
    max_depth: int = 2,
) -> dict[str, Any]:
    """Get the full network around a congress member.

    Returns all nodes and relationships within max_depth hops,
    including: committees, trades, lobbying connections, donations.
    """
    result = await session.run(
        """
        MATCH (member:Member {bioguide_id: $bioguide_id})
        CALL apoc.path.subgraphAll(member, {maxLevel: $max_depth})
        YIELD nodes, relationships
        WITH nodes, relationships
        UNWIND nodes AS node
        WITH collect(DISTINCT {
            id: id(node),
            labels: labels(node),
            properties: properties(node)
        }) AS nodeList, relationships
        UNWIND relationships AS rel
        RETURN nodeList AS nodes,
               collect(DISTINCT {
                   id: id(rel),
                   type: type(rel),
                   startNode: id(startNode(rel)),
                   endNode: id(endNode(rel)),
                   properties: properties(rel)
               }) AS relationships
        """,
        bioguide_id=bioguide_id,
        max_depth=max_depth,
    )

    record = await result.single()
    if not record:
        # Fallback: simple query without APOC
        return await _get_member_network_simple(session, bioguide_id)

    return {
        "nodes": record["nodes"],
        "relationships": record["relationships"],
    }


async def _get_member_network_simple(
    session: AsyncSession,
    bioguide_id: str,
) -> dict[str, Any]:
    """Fallback network query without APOC dependency."""
    result = await session.run(
        """
        MATCH (member:Member {bioguide_id: $bioguide_id})
        OPTIONAL MATCH (member)-[r1]->(n1)
        OPTIONAL MATCH (n1)-[r2]->(n2)
        WITH member,
             collect(DISTINCT {
                 id: id(n1), labels: labels(n1), properties: properties(n1)
             }) AS neighbors,
             collect(DISTINCT {
                 id: id(n2), labels: labels(n2), properties: properties(n2)
             }) AS second_neighbors,
             collect(DISTINCT {
                 type: type(r1), startNode: id(startNode(r1)),
                 endNode: id(endNode(r1)), properties: properties(r1)
             }) AS rels1,
             collect(DISTINCT {
                 type: type(r2), startNode: id(startNode(r2)),
                 endNode: id(endNode(r2)), properties: properties(r2)
             }) AS rels2
        RETURN [{id: id(member), labels: labels(member), properties: properties(member)}]
               + neighbors + second_neighbors AS nodes,
               rels1 + rels2 AS relationships
        """,
        bioguide_id=bioguide_id,
    )

    record = await result.single()
    if not record:
        return {"nodes": [], "relationships": []}

    return {
        "nodes": record["nodes"],
        "relationships": record["relationships"],
    }


async def find_paths_between(
    session: AsyncSession,
    bioguide_id: str,
    ticker: str,
    max_depth: int = 4,
) -> list[dict[str, Any]]:
    """Find all connection paths between a member and a company/ticker.

    This is the key differentiator — shows HOW a member is connected to
    a company beyond just trading it. Paths might include:
    - Direct trade (TRADED)
    - Committee oversight (SITS_ON -> Committee <- REFERRED_TO <- Bill about company)
    - Lobbying (LobbyingFirm -> LOBBIED_FOR -> Company, same firm -> LOBBIED -> Member)
    - Donations (Company -> DONATED_TO -> Member)
    """
    result = await session.run(
        """
        MATCH (member:Member {bioguide_id: $bioguide_id})
        MATCH (company:Company {ticker: $ticker})
        MATCH path = allShortestPaths((member)-[*..{max_depth}]-(company))
        WITH path, length(path) AS pathLength
        ORDER BY pathLength
        LIMIT 10
        RETURN [node IN nodes(path) | {
            id: id(node),
            labels: labels(node),
            properties: properties(node)
        }] AS path_nodes,
        [rel IN relationships(path) | {
            type: type(rel),
            startNode: id(startNode(rel)),
            endNode: id(endNode(rel)),
            properties: properties(rel)
        }] AS path_relationships,
        pathLength
        """,
        bioguide_id=bioguide_id,
        ticker=ticker,
        max_depth=max_depth,
    )

    paths = []
    async for record in result:
        paths.append({
            "nodes": record["path_nodes"],
            "relationships": record["path_relationships"],
            "length": record["pathLength"],
        })

    return paths


async def get_suspicious_triangles(
    session: AsyncSession,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Find suspicious triangles: member traded company + lobbying firm lobbied
    that company + same firm lobbied the member.

    This pattern suggests potential influence:
    LobbyingFirm -[LOBBIED_FOR]-> Company <-[TRADED]- Member <-[LOBBIED]- LobbyingFirm
    """
    result = await session.run(
        """
        MATCH (firm:LobbyingFirm)-[:LOBBIED_FOR]->(company:Company)
              <-[trade:TRADED]-(member:Member)
        WHERE EXISTS {
            MATCH (firm)-[:LOBBIED]->(member)
        }
        OR EXISTS {
            MATCH (firm)-[:EMPLOYED_BY]-(lobbyist:Lobbyist)-[:FORMERLY_AT]->(member)
        }
        RETURN member.bioguide_id AS member_bioguide_id,
               member.full_name AS member_name,
               company.ticker AS ticker,
               company.name AS company_name,
               firm.name AS lobbying_firm,
               trade.transaction_type AS trade_type,
               trade.transaction_date AS trade_date,
               trade.amount_range_low AS amount_low,
               trade.amount_range_high AS amount_high
        ORDER BY trade.transaction_date DESC
        LIMIT $limit
        """,
        limit=limit,
    )

    triangles = []
    async for record in result:
        triangles.append(dict(record))

    return triangles


async def committee_company_overlap(
    session: AsyncSession,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Find companies that both lobby committees AND are traded by committee members.

    Pattern: Company is lobbied on bills referred to Committee,
    and a Member sitting on that Committee traded that Company.
    """
    result = await session.run(
        """
        MATCH (member:Member)-[:SITS_ON]->(committee:Committee)
              <-[:REFERRED_TO]-(bill:Bill)
        MATCH (member)-[trade:TRADED]->(company:Company)
        MATCH (firm:LobbyingFirm)-[:LOBBIED_FOR]->(company)
        RETURN DISTINCT
               member.bioguide_id AS member_bioguide_id,
               member.full_name AS member_name,
               committee.system_code AS committee_code,
               committee.name AS committee_name,
               company.ticker AS ticker,
               company.name AS company_name,
               firm.name AS lobbying_firm,
               bill.bill_id AS related_bill,
               trade.transaction_type AS trade_type,
               trade.transaction_date AS trade_date
        ORDER BY trade.transaction_date DESC
        LIMIT $limit
        """,
        limit=limit,
    )

    overlaps = []
    async for record in result:
        overlaps.append(dict(record))

    return overlaps


async def get_member_trading_connections(
    session: AsyncSession,
    bioguide_id: str,
) -> dict[str, Any]:
    """Get a summary of a member's trading-related connections.

    Returns counts and details of:
    - Companies traded
    - Lobbying firms connected via traded companies
    - Campaign donors from traded companies
    - Committee overlap with traded companies
    """
    result = await session.run(
        """
        MATCH (member:Member {bioguide_id: $bioguide_id})

        // Companies traded
        OPTIONAL MATCH (member)-[trade:TRADED]->(company:Company)
        WITH member, collect(DISTINCT {
            ticker: company.ticker,
            name: company.name,
            trade_count: count(trade)
        }) AS traded_companies

        // Lobbying connections to traded companies
        OPTIONAL MATCH (member)-[:TRADED]->(company:Company)
                       <-[:LOBBIED_FOR]-(firm:LobbyingFirm)
        WITH member, traded_companies,
             collect(DISTINCT {
                 firm_name: firm.name,
                 company_ticker: company.ticker
             }) AS lobbying_connections

        // Donation connections from traded companies
        OPTIONAL MATCH (company:Company)-[d:DONATED_TO]->(member)
        WHERE (member)-[:TRADED]->(company)
        WITH member, traded_companies, lobbying_connections,
             collect(DISTINCT {
                 ticker: company.ticker,
                 total_amount: d.total_amount,
                 count: d.contribution_count
             }) AS donation_connections

        RETURN member.bioguide_id AS bioguide_id,
               member.full_name AS full_name,
               traded_companies,
               lobbying_connections,
               donation_connections,
               size(traded_companies) AS companies_traded,
               size(lobbying_connections) AS lobbying_link_count,
               size(donation_connections) AS donation_link_count
        """,
        bioguide_id=bioguide_id,
    )

    record = await result.single()
    if not record:
        return {}

    return dict(record)


async def get_graph_stats(session: AsyncSession) -> dict[str, int]:
    """Get basic graph statistics for the health/status endpoint."""
    result = await session.run(
        """
        OPTIONAL MATCH (m:Member) WITH count(m) AS members
        OPTIONAL MATCH (c:Company) WITH members, count(c) AS companies
        OPTIONAL MATCH (com:Committee) WITH members, companies, count(com) AS committees
        OPTIONAL MATCH (b:Bill) WITH members, companies, committees, count(b) AS bills
        OPTIONAL MATCH (f:LobbyingFirm) WITH members, companies, committees, bills, count(f) AS firms
        OPTIONAL MATCH ()-[t:TRADED]->() WITH members, companies, committees, bills, firms, count(t) AS trades
        OPTIONAL MATCH ()-[l:LOBBIED_FOR]->() WITH members, companies, committees, bills, firms, trades, count(l) AS lobbied_for
        OPTIONAL MATCH ()-[d:DONATED_TO]->() WITH members, companies, committees, bills, firms, trades, lobbied_for, count(d) AS donated_to
        RETURN members, companies, committees, bills, firms, trades, lobbied_for, donated_to
        """
    )

    record = await result.single()
    if not record:
        return {}

    return dict(record)
