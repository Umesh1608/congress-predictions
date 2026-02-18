"""Neo4j async driver wrapper.

Provides a singleton AsyncDriver and a session context manager
for use throughout the application.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from src.config import settings

logger = logging.getLogger(__name__)

_driver: AsyncDriver | None = None


def get_driver() -> AsyncDriver:
    """Get or create the singleton Neo4j async driver."""
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=25,
        )
        logger.info("Created Neo4j driver for %s", settings.neo4j_uri)
    return _driver


async def close_driver() -> None:
    """Close the Neo4j driver (call on app shutdown)."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
        logger.info("Closed Neo4j driver")


@asynccontextmanager
async def get_neo4j_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for a Neo4j session."""
    driver = get_driver()
    async with driver.session() as session:
        yield session


async def verify_connectivity() -> bool:
    """Check that Neo4j is reachable."""
    try:
        driver = get_driver()
        await driver.verify_connectivity()
        return True
    except Exception as e:
        logger.warning("Neo4j connectivity check failed: %s", e)
        return False
