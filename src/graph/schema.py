"""Neo4j graph schema: node labels, relationship types, and constraint definitions.

Node types:
    - Member          — congress member (bioguide_id)
    - Company         — company/ticker (ticker)
    - Committee       — congressional committee (system_code)
    - Bill            — legislation (bill_id)
    - LobbyingFirm    — lobbying registrant (senate_id)
    - Lobbyist        — individual lobbyist (name)

Relationship types:
    - TRADED          — Member -[TRADED {type, date, amount}]-> Company
    - SITS_ON         — Member -[SITS_ON {role}]-> Committee
    - SPONSORED       — Member -[SPONSORED]-> Bill
    - COSPONSORED     — Member -[COSPONSORED]-> Bill
    - REFERRED_TO     — Bill -[REFERRED_TO]-> Committee
    - LOBBIED_FOR     — LobbyingFirm -[LOBBIED_FOR {amount, year}]-> Company
    - LOBBIED         — LobbyingFirm -[LOBBIED {issues}]-> Member
    - EMPLOYED_BY     — Lobbyist -[EMPLOYED_BY]-> LobbyingFirm
    - FORMERLY_AT     — Lobbyist -[FORMERLY_AT {position}]-> Member  (revolving door)
    - DONATED_TO      — Company -[DONATED_TO {total_amount, count}]-> Member
    - HEARING_ON      — Committee -[HEARING_ON {date, title}]-> Bill
"""

from __future__ import annotations

import logging

from neo4j import AsyncSession

logger = logging.getLogger(__name__)

# Constraints ensure uniqueness and create indexes
CONSTRAINTS = [
    "CREATE CONSTRAINT member_bioguide IF NOT EXISTS FOR (m:Member) REQUIRE m.bioguide_id IS UNIQUE",
    "CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
    "CREATE CONSTRAINT committee_code IF NOT EXISTS FOR (c:Committee) REQUIRE c.system_code IS UNIQUE",
    "CREATE CONSTRAINT bill_id IF NOT EXISTS FOR (b:Bill) REQUIRE b.bill_id IS UNIQUE",
    "CREATE CONSTRAINT lobbying_firm_id IF NOT EXISTS FOR (f:LobbyingFirm) REQUIRE f.senate_id IS UNIQUE",
]

# Additional indexes for common lookups
INDEXES = [
    "CREATE INDEX member_name IF NOT EXISTS FOR (m:Member) ON (m.full_name)",
    "CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)",
    "CREATE INDEX lobbyist_name IF NOT EXISTS FOR (l:Lobbyist) ON (l.name)",
]


async def create_constraints(session: AsyncSession) -> None:
    """Create all uniqueness constraints and indexes in Neo4j."""
    for constraint in CONSTRAINTS:
        try:
            await session.run(constraint)
        except Exception as e:
            logger.debug("Constraint may already exist: %s", e)

    for index in INDEXES:
        try:
            await session.run(index)
        except Exception as e:
            logger.debug("Index may already exist: %s", e)

    logger.info("Neo4j schema constraints and indexes ensured")
