"""Tests for health check endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.v1.health import _check_postgres, _check_redis, _check_neo4j


class TestCheckPostgres:
    @pytest.mark.asyncio
    async def test_ok(self):
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result

        result = await _check_postgres(mock_session)
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_error(self):
        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("connection refused")

        result = await _check_postgres(mock_session)
        assert result["status"] == "error"
        assert "connection refused" in result["detail"]


class TestCheckRedis:
    @pytest.mark.asyncio
    async def test_ok(self):
        mock_redis_client = MagicMock()
        mock_redis_client.ping.return_value = True
        mock_redis_module = MagicMock()
        mock_redis_module.from_url.return_value = mock_redis_client

        with patch.dict("sys.modules", {"redis": mock_redis_module}):
            result = await _check_redis()
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_error(self):
        mock_redis_module = MagicMock()
        mock_redis_module.from_url.side_effect = Exception("refused")

        with patch.dict("sys.modules", {"redis": mock_redis_module}):
            result = await _check_redis()
        assert result["status"] == "error"


class TestCheckNeo4j:
    @pytest.mark.asyncio
    async def test_ok(self):
        with patch("src.db.neo4j.verify_connectivity", new_callable=AsyncMock, return_value=True):
            result = await _check_neo4j()
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_failed(self):
        with patch("src.db.neo4j.verify_connectivity", new_callable=AsyncMock, return_value=False):
            result = await _check_neo4j()
        assert result["status"] == "error"


class TestLegalDisclaimer:
    def test_health_legal_import(self):
        """Verify the legal endpoint function exists and is importable."""
        from src.api.v1.health import legal_disclaimer
        assert legal_disclaimer is not None
