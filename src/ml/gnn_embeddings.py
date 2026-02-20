"""GNN embedding pipeline: extract graph from Neo4j, train GraphSAGE, produce embeddings.

Extracts the heterogeneous graph from Neo4j, trains a self-supervised GraphSAGE
model via link prediction on TRADED edges, and produces per-node embeddings that
can be appended to tabular features for FT-Transformer training.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)

EMBEDDING_CACHE_PATH = "data/gnn_embeddings.pkl"

# --- Node type definitions ---
NODE_TYPES = ["Member", "Company", "Committee", "Bill", "LobbyingFirm", "Lobbyist"]
EDGE_TYPES = [
    ("Member", "TRADED", "Company"),
    ("Member", "SITS_ON", "Committee"),
    ("Member", "SPONSORED", "Bill"),
    ("Member", "COSPONSORED", "Bill"),
    ("Bill", "REFERRED_TO", "Committee"),
    ("LobbyingFirm", "LOBBIED_FOR", "Company"),
    ("LobbyingFirm", "LOBBIED", "Member"),
    ("Lobbyist", "EMPLOYED_BY", "LobbyingFirm"),
    ("Lobbyist", "FORMERLY_AT", "Member"),
    ("Company", "DONATED_TO", "Member"),
    ("Committee", "HEARING_ON", "Bill"),
]

# Party encoding: Democrat=0, Republican=1, Independent=2
PARTY_MAP = {"Democratic": 0, "Republican": 1, "Independent": 2}
CHAMBER_MAP = {"house": 0, "senate": 1}


async def extract_graph_from_neo4j(neo4j_session) -> dict[str, Any]:
    """Extract all nodes and edges from Neo4j into PyTorch Geometric-compatible format.

    Returns a dict with:
        - node_id_maps: {node_type: {key: int_id}}
        - node_features: {node_type: Tensor}
        - edge_indices: {(src_type, rel_type, dst_type): Tensor[2, num_edges]}
    """
    node_id_maps: dict[str, dict[str, int]] = {}
    node_features: dict[str, Tensor] = {}

    # --- Extract Member nodes ---
    result = await neo4j_session.run(
        "MATCH (m:Member) RETURN m.bioguide_id AS id, m.chamber AS chamber, "
        "m.party AS party, m.in_office AS in_office, "
        "m.nominate_dim1 AS dim1, m.nominate_dim2 AS dim2"
    )
    members = []
    member_map: dict[str, int] = {}
    async for record in result:
        idx = len(member_map)
        member_map[record["id"]] = idx
        chamber_enc = [0.0, 0.0]
        ch = CHAMBER_MAP.get(record["chamber"] or "", -1)
        if ch >= 0:
            chamber_enc[ch] = 1.0
        party_enc = [0.0, 0.0, 0.0]
        p = PARTY_MAP.get(record["party"] or "", -1)
        if p >= 0:
            party_enc[p] = 1.0
        in_office = 1.0 if record["in_office"] else 0.0
        dim1 = float(record["dim1"] or 0.0)
        dim2 = float(record["dim2"] or 0.0)
        members.append(chamber_enc + party_enc + [in_office, dim1, dim2])
    node_id_maps["Member"] = member_map
    node_features["Member"] = torch.tensor(members, dtype=torch.float32) if members else torch.zeros(0, 8)

    # --- Extract Company nodes (degree-based features computed after edges) ---
    result = await neo4j_session.run(
        "MATCH (c:Company) RETURN c.ticker AS id"
    )
    company_map: dict[str, int] = {}
    async for record in result:
        company_map[record["id"]] = len(company_map)
    node_id_maps["Company"] = company_map

    # --- Extract Committee nodes ---
    result = await neo4j_session.run(
        "MATCH (c:Committee) RETURN c.system_code AS id"
    )
    committee_map: dict[str, int] = {}
    async for record in result:
        committee_map[record["id"]] = len(committee_map)
    node_id_maps["Committee"] = committee_map

    # --- Extract Bill nodes ---
    result = await neo4j_session.run(
        "MATCH (b:Bill) RETURN b.bill_id AS id"
    )
    bill_map: dict[str, int] = {}
    async for record in result:
        bill_map[record["id"]] = len(bill_map)
    node_id_maps["Bill"] = bill_map

    # --- Extract LobbyingFirm nodes ---
    result = await neo4j_session.run(
        "MATCH (f:LobbyingFirm) RETURN f.senate_id AS id"
    )
    firm_map: dict[str, int] = {}
    async for record in result:
        firm_map[record["id"]] = len(firm_map)
    node_id_maps["LobbyingFirm"] = firm_map

    # --- Extract Lobbyist nodes ---
    result = await neo4j_session.run(
        "MATCH (l:Lobbyist) RETURN l.name AS id"
    )
    lobbyist_map: dict[str, int] = {}
    async for record in result:
        lobbyist_map[record["id"]] = len(lobbyist_map)
    node_id_maps["Lobbyist"] = lobbyist_map

    # --- Extract edges ---
    edge_indices: dict[tuple[str, str, str], Tensor] = {}

    edge_queries = [
        (("Member", "TRADED", "Company"),
         "MATCH (a:Member)-[:TRADED]->(b:Company) RETURN DISTINCT a.bioguide_id AS src, b.ticker AS dst"),
        (("Member", "SITS_ON", "Committee"),
         "MATCH (a:Member)-[:SITS_ON]->(b:Committee) RETURN DISTINCT a.bioguide_id AS src, b.system_code AS dst"),
        (("Member", "SPONSORED", "Bill"),
         "MATCH (a:Member)-[:SPONSORED]->(b:Bill) RETURN DISTINCT a.bioguide_id AS src, b.bill_id AS dst"),
        (("Member", "COSPONSORED", "Bill"),
         "MATCH (a:Member)-[:COSPONSORED]->(b:Bill) RETURN DISTINCT a.bioguide_id AS src, b.bill_id AS dst"),
        (("Bill", "REFERRED_TO", "Committee"),
         "MATCH (a:Bill)-[:REFERRED_TO]->(b:Committee) RETURN DISTINCT a.bill_id AS src, b.system_code AS dst"),
        (("LobbyingFirm", "LOBBIED_FOR", "Company"),
         "MATCH (a:LobbyingFirm)-[:LOBBIED_FOR]->(b:Company) RETURN DISTINCT a.senate_id AS src, b.ticker AS dst"),
        (("LobbyingFirm", "LOBBIED", "Member"),
         "MATCH (a:LobbyingFirm)-[:LOBBIED]->(b:Member) RETURN DISTINCT a.senate_id AS src, b.bioguide_id AS dst"),
        (("Lobbyist", "EMPLOYED_BY", "LobbyingFirm"),
         "MATCH (a:Lobbyist)-[:EMPLOYED_BY]->(b:LobbyingFirm) RETURN DISTINCT a.name AS src, b.senate_id AS dst"),
        (("Lobbyist", "FORMERLY_AT", "Member"),
         "MATCH (a:Lobbyist)-[:FORMERLY_AT]->(b:Member) RETURN DISTINCT a.name AS src, b.bioguide_id AS dst"),
        (("Company", "DONATED_TO", "Member"),
         "MATCH (a:Company)-[:DONATED_TO]->(b:Member) RETURN DISTINCT a.ticker AS src, b.bioguide_id AS dst"),
        (("Committee", "HEARING_ON", "Bill"),
         "MATCH (a:Committee)-[:HEARING_ON]->(b:Bill) RETURN DISTINCT a.system_code AS src, b.bill_id AS dst"),
    ]

    for edge_type, query in edge_queries:
        src_type, _, dst_type = edge_type
        src_map = node_id_maps[src_type]
        dst_map = node_id_maps[dst_type]

        result = await neo4j_session.run(query)
        src_ids = []
        dst_ids = []
        async for record in result:
            s = src_map.get(record["src"])
            d = dst_map.get(record["dst"])
            if s is not None and d is not None:
                src_ids.append(s)
                dst_ids.append(d)

        if src_ids:
            edge_indices[edge_type] = torch.tensor([src_ids, dst_ids], dtype=torch.long)
        else:
            edge_indices[edge_type] = torch.zeros(2, 0, dtype=torch.long)

    # --- Compute degree-based features for non-Member nodes ---
    def _degree_features(node_type: str, n_nodes: int) -> Tensor:
        """Compute in-degree per relationship type as features."""
        degrees = torch.zeros(max(n_nodes, 1), dtype=torch.float32)
        for (src_t, _, dst_t), ei in edge_indices.items():
            if dst_t == node_type and ei.shape[1] > 0:
                for idx in ei[1].tolist():
                    if idx < n_nodes:
                        degrees[idx] += 1
        for (src_t, _, dst_t), ei in edge_indices.items():
            if src_t == node_type and ei.shape[1] > 0:
                for idx in ei[0].tolist():
                    if idx < n_nodes:
                        degrees[idx] += 1
        return degrees.unsqueeze(1)  # [n_nodes, 1]

    node_features["Company"] = _degree_features("Company", len(company_map))
    node_features["Committee"] = _degree_features("Committee", len(committee_map))
    node_features["Bill"] = _degree_features("Bill", len(bill_map))
    node_features["LobbyingFirm"] = _degree_features("LobbyingFirm", len(firm_map))
    node_features["Lobbyist"] = _degree_features("Lobbyist", len(lobbyist_map))

    logger.info(
        "Extracted graph: %s",
        {nt: len(m) for nt, m in node_id_maps.items()},
    )
    logger.info(
        "Edge counts: %s",
        {f"{s}-{r}->{d}": ei.shape[1] for (s, r, d), ei in edge_indices.items()},
    )

    return {
        "node_id_maps": node_id_maps,
        "node_features": node_features,
        "edge_indices": edge_indices,
    }


# ---------------------------------------------------------------------------
# Heterogeneous GraphSAGE model
# ---------------------------------------------------------------------------


class HeteroGraphSAGELayer(nn.Module):
    """One layer of heterogeneous message passing.

    For each edge type (src_type, rel, dst_type), aggregates source node
    features into destination nodes, then combines with a linear transform.
    """

    def __init__(
        self,
        in_dims: dict[str, int],
        out_dim: int,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        # Projection for each node type's own features
        self.self_lins = nn.ModuleDict({
            nt: nn.Linear(d, out_dim) for nt, d in in_dims.items()
        })
        # Projection for each edge type's aggregated messages
        self.neigh_lins = nn.ModuleDict()

    def add_edge_type(self, src_type: str, rel: str, dst_type: str, src_dim: int) -> None:
        """Register a linear transform for a specific edge type."""
        key = f"{src_type}__{rel}__{dst_type}"
        self.neigh_lins[key] = nn.Linear(src_dim, self.out_dim)

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_indices: dict[tuple[str, str, str], Tensor],
    ) -> dict[str, Tensor]:
        """Forward pass: aggregate neighbors, combine with self features."""
        # Start with self-projection
        out: dict[str, Tensor] = {}
        for nt, x in x_dict.items():
            if nt in self.self_lins and x.shape[0] > 0:
                out[nt] = self.self_lins[nt](x)
            else:
                out[nt] = torch.zeros(x.shape[0], self.out_dim, device=x.device)

        # Add neighbor aggregations
        for (src_t, rel, dst_t), ei in edge_indices.items():
            key = f"{src_t}__{rel}__{dst_t}"
            if key not in self.neigh_lins or ei.shape[1] == 0:
                continue

            src_feats = x_dict[src_t]
            src_idx, dst_idx = ei[0], ei[1]

            # Mean aggregation
            n_dst = x_dict[dst_t].shape[0]
            agg = torch.zeros(n_dst, src_feats.shape[1], device=src_feats.device)
            count = torch.zeros(n_dst, 1, device=src_feats.device)
            agg.index_add_(0, dst_idx, src_feats[src_idx])
            count.index_add_(0, dst_idx, torch.ones(len(dst_idx), 1, device=src_feats.device))
            count = count.clamp(min=1)
            agg = agg / count

            neigh_proj = self.neigh_lins[key](agg)
            if dst_t in out:
                out[dst_t] = out[dst_t] + neigh_proj
            else:
                out[dst_t] = neigh_proj

        # Activation
        return {nt: F.relu(h) for nt, h in out.items()}


class HeteroGraphSAGE(nn.Module):
    """Two-layer heterogeneous GraphSAGE for link prediction."""

    def __init__(
        self,
        node_dims: dict[str, int],
        hidden_dim: int = 64,
        embed_dim: int = 64,
        edge_types: list[tuple[str, str, str]] | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Layer 1: input dims -> hidden_dim
        self.layer1 = HeteroGraphSAGELayer(node_dims, hidden_dim)
        if edge_types:
            for src_t, rel, dst_t in edge_types:
                if src_t in node_dims:
                    self.layer1.add_edge_type(src_t, rel, dst_t, node_dims[src_t])

        # Layer 2: hidden_dim -> embed_dim
        hidden_dims = {nt: hidden_dim for nt in node_dims}
        self.layer2 = HeteroGraphSAGELayer(hidden_dims, embed_dim)
        if edge_types:
            for src_t, rel, dst_t in edge_types:
                if src_t in node_dims:
                    self.layer2.add_edge_type(src_t, rel, dst_t, hidden_dim)

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_indices: dict[tuple[str, str, str], Tensor],
    ) -> dict[str, Tensor]:
        """Two-layer forward pass producing node embeddings."""
        h = self.layer1(x_dict, edge_indices)
        h = self.layer2(h, edge_indices)
        return h


def train_gnn(
    graph_data: dict[str, Any],
    embed_dim: int = 64,
    hidden_dim: int = 64,
    epochs: int = 100,
    lr: float = 0.01,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Train GNN via self-supervised link prediction on TRADED edges.

    Returns dict mapping node_type -> {key: embedding_array}.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    node_features = graph_data["node_features"]
    edge_indices = graph_data["edge_indices"]
    node_id_maps = graph_data["node_id_maps"]

    # Get node feature dimensions
    node_dims: dict[str, int] = {}
    for nt, feat in node_features.items():
        if feat.shape[0] > 0:
            node_dims[nt] = feat.shape[1]
        else:
            node_dims[nt] = 1  # placeholder for empty node types

    # Filter edge types that have both source and destination nodes
    active_edge_types = []
    for et, ei in edge_indices.items():
        src_t, _, dst_t = et
        if src_t in node_dims and dst_t in node_dims and ei.shape[1] > 0:
            active_edge_types.append(et)

    # Build model
    model = HeteroGraphSAGE(
        node_dims=node_dims,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        edge_types=active_edge_types,
    ).to(device)

    # Move features and edges to device
    x_dict = {nt: feat.to(device) for nt, feat in node_features.items()}
    ei_dict = {et: ei.to(device) for et, ei in edge_indices.items()}

    # Link prediction on TRADED edges
    traded_key = ("Member", "TRADED", "Company")
    traded_edges = edge_indices.get(traded_key)

    if traded_edges is None or traded_edges.shape[1] < 10:
        logger.warning("Not enough TRADED edges for GNN training (%d). Using random init.",
                        0 if traded_edges is None else traded_edges.shape[1])
        # Return random embeddings
        return _extract_embeddings(model, x_dict, ei_dict, node_id_maps, device)

    # Split TRADED edges 80/20
    n_edges = traded_edges.shape[1]
    perm = torch.randperm(n_edges)
    n_train = int(n_edges * 0.8)
    train_edges = traded_edges[:, perm[:n_train]]
    val_edges = traded_edges[:, perm[n_train:]]

    # Use train edges for message passing (replace full TRADED edges)
    train_ei = dict(ei_dict)
    train_ei[traded_key] = train_edges.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_members = len(node_id_maps["Member"])
    n_companies = len(node_id_maps["Company"])

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        embeddings = model(x_dict, train_ei)
        member_emb = embeddings.get("Member")
        company_emb = embeddings.get("Company")

        if member_emb is None or company_emb is None:
            break

        # Positive edges (from training split)
        pos_src = train_edges[0].to(device)
        pos_dst = train_edges[1].to(device)
        pos_scores = (member_emb[pos_src] * company_emb[pos_dst]).sum(dim=1)

        # Negative sampling: random member-company pairs
        neg_src = torch.randint(0, n_members, (len(pos_src),), device=device)
        neg_dst = torch.randint(0, n_companies, (len(pos_src),), device=device)
        neg_scores = (member_emb[neg_src] * company_emb[neg_dst]).sum(dim=1)

        # BCE loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_emb = model(x_dict, train_ei)
                v_member = val_emb.get("Member")
                v_company = val_emb.get("Company")
                if v_member is not None and v_company is not None:
                    v_pos_src = val_edges[0].to(device)
                    v_pos_dst = val_edges[1].to(device)
                    v_pos = (v_member[v_pos_src] * v_company[v_pos_dst]).sum(dim=1)
                    v_neg_src = torch.randint(0, n_members, (val_edges.shape[1],), device=device)
                    v_neg_dst = torch.randint(0, n_companies, (val_edges.shape[1],), device=device)
                    v_neg = (v_member[v_neg_src] * v_company[v_neg_dst]).sum(dim=1)
                    val_loss = (
                        F.binary_cross_entropy_with_logits(v_pos, torch.ones_like(v_pos))
                        + F.binary_cross_entropy_with_logits(v_neg, torch.zeros_like(v_neg))
                    ).item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    logger.info(
                        "GNN epoch %d/%d: train_loss=%.4f, val_loss=%.4f",
                        epoch + 1, epochs, loss.item(), val_loss,
                    )

                    if patience_counter >= 5:
                        logger.info("Early stopping at epoch %d", epoch + 1)
                        break

    return _extract_embeddings(model, x_dict, ei_dict, node_id_maps, device)


def _extract_embeddings(
    model: HeteroGraphSAGE,
    x_dict: dict[str, Tensor],
    ei_dict: dict[tuple[str, str, str], Tensor],
    node_id_maps: dict[str, dict[str, int]],
    device: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Extract embeddings from trained model."""
    model.eval()
    with torch.no_grad():
        embeddings = model(x_dict, ei_dict)

    result: dict[str, dict[str, np.ndarray]] = {}
    for node_type, id_map in node_id_maps.items():
        emb = embeddings.get(node_type)
        if emb is None:
            continue
        emb_np = emb.cpu().numpy()
        type_map: dict[str, np.ndarray] = {}
        for key, idx in id_map.items():
            if idx < emb_np.shape[0]:
                type_map[key] = emb_np[idx]
        result[node_type] = type_map

    return result


def build_trade_embedding_columns(
    df: pd.DataFrame,
    embeddings: dict[str, dict[str, np.ndarray]],
    embed_dim: int = 64,
) -> tuple[pd.DataFrame, list[str]]:
    """Append GNN embeddings to a trade DataFrame.

    For each trade, looks up member and company (ticker) embeddings.
    Returns (augmented_df, list_of_new_column_names).
    """
    member_embs = embeddings.get("Member", {})
    company_embs = embeddings.get("Company", {})

    member_cols = [f"gnn_member_{i}" for i in range(embed_dim)]
    company_cols = [f"gnn_company_{i}" for i in range(embed_dim)]
    all_cols = member_cols + company_cols

    zero_member = np.zeros(embed_dim)
    zero_company = np.zeros(embed_dim)

    # Determine which columns hold the keys
    member_key_col = "member_bioguide_id" if "member_bioguide_id" in df.columns else None
    ticker_col = "ticker" if "ticker" in df.columns else None

    rows = []
    for _, row in df.iterrows():
        m_id = row[member_key_col] if member_key_col else None
        t_id = row[ticker_col] if ticker_col else None

        m_emb = member_embs.get(m_id, zero_member) if m_id else zero_member
        c_emb = company_embs.get(t_id, zero_company) if t_id else zero_company
        rows.append(np.concatenate([m_emb, c_emb]))

    emb_df = pd.DataFrame(rows, columns=all_cols, index=df.index)
    result = pd.concat([df, emb_df], axis=1)
    return result, all_cols


def save_embeddings(
    embeddings: dict[str, dict[str, np.ndarray]],
    path: str = EMBEDDING_CACHE_PATH,
) -> None:
    """Save embeddings to disk for reuse during inference."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info("Saved GNN embeddings to %s", path)


def load_embeddings(path: str = EMBEDDING_CACHE_PATH) -> dict[str, dict[str, np.ndarray]] | None:
    """Load cached embeddings from disk."""
    if not Path(path).exists():
        logger.warning("No cached GNN embeddings at %s", path)
        return None
    with open(path, "rb") as f:
        embeddings = pickle.load(f)  # noqa: S301
    logger.info("Loaded GNN embeddings from %s", path)
    return embeddings
