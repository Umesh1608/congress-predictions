"""Network graph API endpoints.

Provides access to the Neo4j relationship graph connecting congress members
to companies through trading, lobbying, campaign finance, and committee paths.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from src.db.neo4j import get_neo4j_session, verify_connectivity
from src.graph.queries import (
    committee_company_overlap,
    find_paths_between,
    get_graph_stats,
    get_member_network,
    get_member_trading_connections,
    get_suspicious_triangles,
)
from src.schemas.network import (
    CommitteeCompanyOverlapResponse,
    GraphStatsResponse,
    MemberNetworkResponse,
    PathResponse,
    PathsResponse,
    SuspiciousTriangleResponse,
    TradingConnectionsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["network"])


async def _check_neo4j() -> None:
    """Verify Neo4j is available, raise 503 if not."""
    if not await verify_connectivity():
        raise HTTPException(
            status_code=503,
            detail="Neo4j graph database is not available",
        )


@router.get("/network/stats", response_model=GraphStatsResponse)
async def graph_stats():
    """Get graph database statistics."""
    await _check_neo4j()
    async with get_neo4j_session() as session:
        stats = await get_graph_stats(session)
        return GraphStatsResponse(**stats)


@router.get("/network/member/{bioguide_id}", response_model=MemberNetworkResponse)
async def member_network(
    bioguide_id: str,
    max_depth: int = Query(2, ge=1, le=3, description="Max hops from member"),
):
    """Get the relationship network around a congress member.

    Returns all nodes and relationships within max_depth hops, including
    committees, traded companies, lobbying firms, and campaign donors.
    """
    await _check_neo4j()
    async with get_neo4j_session() as session:
        network = await get_member_network(session, bioguide_id, max_depth)
        if not network.get("nodes"):
            raise HTTPException(status_code=404, detail="Member not found in graph")
        return MemberNetworkResponse(**network)


@router.get("/network/member/{bioguide_id}/paths-to/{ticker}", response_model=PathsResponse)
async def member_paths_to_ticker(
    bioguide_id: str,
    ticker: str,
    max_depth: int = Query(4, ge=2, le=6, description="Max path length"),
):
    """Find all connection paths between a member and a company/ticker.

    Shows HOW a member is connected to a company beyond direct trades:
    - Committee oversight paths
    - Lobbying connections
    - Campaign donation links
    - Bill sponsorship paths
    """
    await _check_neo4j()
    async with get_neo4j_session() as session:
        paths = await find_paths_between(
            session, bioguide_id, ticker.upper(), max_depth
        )
        return PathsResponse(
            member_bioguide_id=bioguide_id,
            ticker=ticker.upper(),
            paths=[PathResponse(**p) for p in paths],
            total_paths=len(paths),
        )


@router.get("/network/member/{bioguide_id}/connections", response_model=TradingConnectionsResponse)
async def member_trading_connections(bioguide_id: str):
    """Get a summary of a member's trading-related connections.

    Returns counts and details of companies traded, connected lobbying firms,
    and campaign donors from traded companies.
    """
    await _check_neo4j()
    async with get_neo4j_session() as session:
        connections = await get_member_trading_connections(session, bioguide_id)
        if not connections:
            raise HTTPException(status_code=404, detail="Member not found in graph")
        return TradingConnectionsResponse(**connections)


@router.get(
    "/network/suspicious-triangles",
    response_model=list[SuspiciousTriangleResponse],
)
async def suspicious_triangles(
    limit: int = Query(50, le=200),
):
    """Find suspicious triangles: member traded a company that lobbied them.

    Pattern: A lobbying firm lobbied FOR a company, that same company was
    TRADED by a member, AND the lobbying firm LOBBIED that member.
    This is a key signal for potential insider influence.
    """
    await _check_neo4j()
    async with get_neo4j_session() as session:
        triangles = await get_suspicious_triangles(session, limit)
        return [SuspiciousTriangleResponse(**t) for t in triangles]


@router.get(
    "/network/committee-company-overlap",
    response_model=list[CommitteeCompanyOverlapResponse],
)
async def committee_overlap(
    limit: int = Query(50, le=200),
):
    """Find companies lobbied on bills in committees where members traded that company.

    Pattern: Member sits on Committee, Bill referred to Committee,
    Lobbying firm lobbied for Company on that Bill, Member traded that Company.
    """
    await _check_neo4j()
    async with get_neo4j_session() as session:
        overlaps = await committee_company_overlap(session, limit)
        return [CommitteeCompanyOverlapResponse(**o) for o in overlaps]
