"""Tests for GNN embedding pipeline with mock Neo4j data."""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
import torch


class TestExtractGraph:
    @pytest.mark.asyncio
    async def test_extract_empty_graph(self):
        """Extract from a graph with no nodes should return empty structures."""
        from src.ml.gnn_embeddings import extract_graph_from_neo4j

        class MockEmptyResult:
            def __aiter__(self):
                return self
            async def __anext__(self):
                raise StopAsyncIteration

        mock_session = AsyncMock()
        mock_session.run.return_value = MockEmptyResult()

        graph = await extract_graph_from_neo4j(mock_session)
        assert "node_id_maps" in graph
        assert "node_features" in graph
        assert "edge_indices" in graph

        # All node types should exist but be empty
        for nt in ["Member", "Company", "Committee", "Bill", "LobbyingFirm", "Lobbyist"]:
            assert nt in graph["node_id_maps"]
            assert len(graph["node_id_maps"][nt]) == 0

    @pytest.mark.asyncio
    async def test_extract_with_members(self):
        """Extract graph with a few member nodes."""
        from src.ml.gnn_embeddings import extract_graph_from_neo4j

        member_records = [
            {"id": "A000001", "chamber": "house", "party": "Democratic",
             "in_office": True, "dim1": 0.5, "dim2": -0.3},
            {"id": "B000002", "chamber": "senate", "party": "Republican",
             "in_office": False, "dim1": -0.2, "dim2": 0.1},
        ]

        call_count = 0

        class MockAsyncIterResult:
            """Mock async iterable for Neo4j query results."""
            def __init__(self, records):
                self._records = iter(records)
            def __aiter__(self):
                return self
            async def __anext__(self):
                try:
                    return next(self._records)
                except StopIteration:
                    raise StopAsyncIteration

        async def mock_run(query, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # Members query
                return MockAsyncIterResult(member_records)
            return MockAsyncIterResult([])

        mock_session = AsyncMock()
        mock_session.run = mock_run

        graph = await extract_graph_from_neo4j(mock_session)

        assert len(graph["node_id_maps"]["Member"]) == 2
        assert "A000001" in graph["node_id_maps"]["Member"]
        assert "B000002" in graph["node_id_maps"]["Member"]
        assert graph["node_features"]["Member"].shape == (2, 8)


class TestTrainGNN:
    def test_train_with_insufficient_edges(self):
        """Training with very few TRADED edges should still return embeddings."""
        from src.ml.gnn_embeddings import train_gnn

        graph_data = {
            "node_id_maps": {
                "Member": {"M1": 0, "M2": 1},
                "Company": {"AAPL": 0, "MSFT": 1},
                "Committee": {},
                "Bill": {},
                "LobbyingFirm": {},
                "Lobbyist": {},
            },
            "node_features": {
                "Member": torch.randn(2, 8),
                "Company": torch.randn(2, 1),
                "Committee": torch.zeros(0, 1),
                "Bill": torch.zeros(0, 1),
                "LobbyingFirm": torch.zeros(0, 1),
                "Lobbyist": torch.zeros(0, 1),
            },
            "edge_indices": {
                ("Member", "TRADED", "Company"): torch.tensor([[0], [0]]),
            },
        }

        embeddings = train_gnn(graph_data, embed_dim=16, epochs=5, device="cpu")

        assert "Member" in embeddings
        assert "Company" in embeddings
        assert "M1" in embeddings["Member"]
        assert embeddings["Member"]["M1"].shape == (16,)

    def test_train_with_sufficient_edges(self):
        """Training with enough edges should converge."""
        from src.ml.gnn_embeddings import train_gnn

        n_members = 20
        n_companies = 10
        n_edges = 50

        graph_data = {
            "node_id_maps": {
                "Member": {f"M{i}": i for i in range(n_members)},
                "Company": {f"T{i}": i for i in range(n_companies)},
                "Committee": {},
                "Bill": {},
                "LobbyingFirm": {},
                "Lobbyist": {},
            },
            "node_features": {
                "Member": torch.randn(n_members, 8),
                "Company": torch.randn(n_companies, 1),
                "Committee": torch.zeros(0, 1),
                "Bill": torch.zeros(0, 1),
                "LobbyingFirm": torch.zeros(0, 1),
                "Lobbyist": torch.zeros(0, 1),
            },
            "edge_indices": {
                ("Member", "TRADED", "Company"): torch.tensor([
                    np.random.randint(0, n_members, n_edges).tolist(),
                    np.random.randint(0, n_companies, n_edges).tolist(),
                ], dtype=torch.long),
            },
        }

        embeddings = train_gnn(
            graph_data, embed_dim=32, hidden_dim=32, epochs=20, device="cpu"
        )

        assert len(embeddings["Member"]) == n_members
        assert len(embeddings["Company"]) == n_companies
        assert embeddings["Member"]["M0"].shape == (32,)

    def test_embedding_dim_respected(self):
        """Embeddings should have the requested dimension."""
        from src.ml.gnn_embeddings import train_gnn

        graph_data = {
            "node_id_maps": {
                "Member": {"M1": 0},
                "Company": {"AAPL": 0},
                "Committee": {},
                "Bill": {},
                "LobbyingFirm": {},
                "Lobbyist": {},
            },
            "node_features": {
                "Member": torch.randn(1, 8),
                "Company": torch.randn(1, 1),
                "Committee": torch.zeros(0, 1),
                "Bill": torch.zeros(0, 1),
                "LobbyingFirm": torch.zeros(0, 1),
                "Lobbyist": torch.zeros(0, 1),
            },
            "edge_indices": {},
        }

        for dim in [16, 32, 64]:
            embeddings = train_gnn(graph_data, embed_dim=dim, epochs=1, device="cpu")
            assert embeddings["Member"]["M1"].shape == (dim,)


class TestBuildTradeEmbeddingColumns:
    def test_adds_correct_columns(self):
        """Should add gnn_member_* and gnn_company_* columns."""
        from src.ml.gnn_embeddings import build_trade_embedding_columns

        df = pd.DataFrame({
            "member_bioguide_id": ["M1", "M2", "M1"],
            "ticker": ["AAPL", "MSFT", "AAPL"],
            "amount": [100, 200, 300],
        })

        embed_dim = 4
        embeddings = {
            "Member": {
                "M1": np.array([1.0, 2.0, 3.0, 4.0]),
                "M2": np.array([5.0, 6.0, 7.0, 8.0]),
            },
            "Company": {
                "AAPL": np.array([0.1, 0.2, 0.3, 0.4]),
                "MSFT": np.array([0.5, 0.6, 0.7, 0.8]),
            },
        }

        result, new_cols = build_trade_embedding_columns(df, embeddings, embed_dim)

        assert len(new_cols) == 2 * embed_dim
        assert "gnn_member_0" in result.columns
        assert "gnn_company_3" in result.columns
        assert result["gnn_member_0"].iloc[0] == 1.0  # M1
        assert result["gnn_member_0"].iloc[1] == 5.0  # M2
        assert result["gnn_company_0"].iloc[0] == 0.1  # AAPL

    def test_unknown_node_gets_zeros(self):
        """Unknown bioguide_id or ticker should get zero embeddings."""
        from src.ml.gnn_embeddings import build_trade_embedding_columns

        df = pd.DataFrame({
            "member_bioguide_id": ["UNKNOWN"],
            "ticker": ["UNKNOWN_TICKER"],
            "amount": [100],
        })

        embeddings = {"Member": {}, "Company": {}}
        result, new_cols = build_trade_embedding_columns(df, embeddings, embed_dim=4)

        for col in new_cols:
            assert result[col].iloc[0] == 0.0

    def test_preserves_original_columns(self):
        """Original DataFrame columns should be unchanged."""
        from src.ml.gnn_embeddings import build_trade_embedding_columns

        df = pd.DataFrame({
            "member_bioguide_id": ["M1"],
            "ticker": ["AAPL"],
            "feature1": [42.0],
        })

        embeddings = {
            "Member": {"M1": np.zeros(4)},
            "Company": {"AAPL": np.zeros(4)},
        }
        result, _ = build_trade_embedding_columns(df, embeddings, embed_dim=4)

        assert result["feature1"].iloc[0] == 42.0
        assert "member_bioguide_id" in result.columns


class TestSaveLoadEmbeddings:
    def test_round_trip(self):
        """Save and load embeddings should produce identical results."""
        from src.ml.gnn_embeddings import load_embeddings, save_embeddings

        embeddings = {
            "Member": {"M1": np.array([1.0, 2.0, 3.0])},
            "Company": {"AAPL": np.array([0.1, 0.2, 0.3])},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "embeddings.pkl")
            save_embeddings(embeddings, path)
            loaded = load_embeddings(path)

            assert loaded is not None
            np.testing.assert_array_equal(
                loaded["Member"]["M1"], embeddings["Member"]["M1"]
            )
            np.testing.assert_array_equal(
                loaded["Company"]["AAPL"], embeddings["Company"]["AAPL"]
            )

    def test_load_nonexistent_returns_none(self):
        from src.ml.gnn_embeddings import load_embeddings

        result = load_embeddings("/nonexistent/path/embeddings.pkl")
        assert result is None


class TestHeteroGraphSAGE:
    def test_forward_pass(self):
        """Test that the model produces embeddings of the correct shape."""
        from src.ml.gnn_embeddings import HeteroGraphSAGE

        node_dims = {"Member": 8, "Company": 1}
        edge_types = [("Member", "TRADED", "Company")]

        model = HeteroGraphSAGE(
            node_dims=node_dims,
            hidden_dim=16,
            embed_dim=32,
            edge_types=edge_types,
        )

        x_dict = {
            "Member": torch.randn(5, 8),
            "Company": torch.randn(3, 1),
        }
        ei_dict = {
            ("Member", "TRADED", "Company"): torch.tensor([[0, 1, 2], [0, 1, 2]]),
        }

        out = model(x_dict, ei_dict)
        assert out["Member"].shape == (5, 32)
        assert out["Company"].shape == (3, 32)

    def test_empty_edges(self):
        """Model should handle empty edge tensors gracefully."""
        from src.ml.gnn_embeddings import HeteroGraphSAGE

        node_dims = {"Member": 8, "Company": 1}
        model = HeteroGraphSAGE(node_dims=node_dims, hidden_dim=16, embed_dim=32)

        x_dict = {
            "Member": torch.randn(3, 8),
            "Company": torch.randn(2, 1),
        }
        ei_dict = {}

        out = model(x_dict, ei_dict)
        assert out["Member"].shape == (3, 32)
        assert out["Company"].shape == (2, 32)
