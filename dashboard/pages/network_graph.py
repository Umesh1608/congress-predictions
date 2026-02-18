"""Network Graph page — interactive pyvis graph visualization."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from dashboard.api_client import CongressAPI


def render() -> None:
    st.title("Network Graph")

    api = CongressAPI()

    # Controls
    col1, col2 = st.columns(2)

    with col1:
        try:
            members = api.get_members(limit=100)
            member_options = {
                f"{m['full_name']} ({m.get('state', '?')})": m["bioguide_id"]
                for m in members
            }
            selected = st.selectbox("Select Member", list(member_options.keys()))
        except Exception:
            st.error("Failed to load members.")
            return

    with col2:
        max_depth = st.slider("Network Depth", min_value=1, max_value=3, value=2)

    if not selected:
        return

    bioguide_id = member_options[selected]

    # Fetch network data
    try:
        network = api.get_member_network(bioguide_id, max_depth=max_depth)
    except Exception as e:
        st.error(f"Failed to load network: {e}")
        st.info("Make sure Neo4j is running and the graph has been synced.")
        return

    nodes = network.get("nodes", [])
    relationships = network.get("relationships", [])

    if not nodes:
        st.info("No network data found for this member.")
        return

    st.subheader(f"Network: {len(nodes)} nodes, {len(relationships)} relationships")

    # Build pyvis graph
    try:
        from pyvis.network import Network

        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        net.force_atlas_2based()

        # Color map for node types
        colors = {
            "Member": "#3498db",
            "Company": "#2ecc71",
            "Committee": "#f39c12",
            "Bill": "#9b59b6",
            "LobbyingFirm": "#e74c3c",
            "Lobbyist": "#1abc9c",
        }

        for node in nodes:
            label = node.get("label", "Unknown")
            node_type = node.get("type", "Unknown")
            color = colors.get(node_type, "#95a5a6")
            title = f"{node_type}: {label}"
            for key, value in node.get("properties", {}).items():
                title += f"\n{key}: {value}"
            net.add_node(
                node.get("id", label),
                label=label[:30],
                color=color,
                title=title,
                size=20 if node_type == "Member" else 15,
            )

        for rel in relationships:
            net.add_edge(
                rel.get("source"),
                rel.get("target"),
                title=rel.get("type", ""),
                label=rel.get("type", "")[:15],
            )

        html = net.generate_html()
        components.html(html, height=620, scrolling=True)

    except ImportError:
        st.warning("pyvis is not installed. Install with: pip install -e '.[dashboard]'")
        st.json({"nodes": len(nodes), "relationships": len(relationships)})

    # Suspicious triangles
    st.markdown("---")
    st.subheader("Suspicious Triangles")
    st.caption("Members who traded stocks of companies that lobbied them")

    try:
        # Use paths-to for a specific ticker if available
        ticker_input = st.text_input("Check paths to ticker", placeholder="e.g., NVDA")
        if ticker_input:
            paths = api.get_member_paths(bioguide_id, ticker_input.upper())
            if paths.get("paths"):
                for path in paths["paths"][:5]:
                    st.write(f"Path length: {path.get('length', '?')}")
                    for step in path.get("steps", []):
                        st.write(f"  → {step}")
            else:
                st.info(f"No paths found between this member and {ticker_input.upper()}")
    except Exception:
        st.info("Path search not available.")
