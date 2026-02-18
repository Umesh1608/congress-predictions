"""Signals page — active signals and alert configuration."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import CongressAPI
from dashboard.charts import signal_strength_chart


def render() -> None:
    st.title("Trading Signals")

    api = CongressAPI()

    # Signal stats
    try:
        stats = api.get_signal_stats()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Signals", stats.get("total_active", 0))
        with col2:
            st.metric("Avg Strength", f"{stats.get('avg_strength', 0):.2f}")
        with col3:
            by_dir = stats.get("by_direction", {})
            bullish = by_dir.get("bullish", 0)
            bearish = by_dir.get("bearish", 0)
            st.metric("Bullish / Bearish", f"{bullish} / {bearish}")
    except Exception:
        st.info("No signal statistics available.")

    st.markdown("---")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        signal_type = st.selectbox(
            "Signal Type",
            ["All", "trade_follow", "anomaly_alert", "sentiment_divergence",
             "insider_cluster", "network_signal"],
        )
    with col2:
        ticker = st.text_input("Ticker Filter", placeholder="e.g., NVDA")
    with col3:
        min_strength = st.slider("Min Strength", 0.0, 1.0, 0.0, 0.05)

    params: dict = {"limit": 50}
    if signal_type != "All":
        params["signal_type"] = signal_type
    if ticker:
        params["ticker"] = ticker.upper()
    if min_strength > 0:
        params["min_strength"] = min_strength

    try:
        signals = api.get_signals(**params)
    except Exception as e:
        st.error(f"Failed to load signals: {e}")
        return

    if signals:
        # Chart
        fig = signal_strength_chart(signals)
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.subheader(f"Active Signals ({len(signals)})")
        for signal in signals:
            with st.expander(
                f"{'🟢' if signal['direction'] == 'bullish' else '🔴' if signal['direction'] == 'bearish' else '🟡'} "
                f"{signal.get('signal_type', '')} — "
                f"{signal.get('ticker', 'N/A')} "
                f"(strength: {signal.get('strength', 0):.2f})"
            ):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.write(f"**Direction:** {signal.get('direction', '')}")
                    st.write(f"**Confidence:** {signal.get('confidence', 0):.2f}")
                with col_b:
                    st.write(f"**Member:** {signal.get('member_bioguide_id', 'N/A')}")
                    st.write(f"**Expires:** {signal.get('expires_at', 'N/A')}")
                with col_c:
                    st.write(f"**Created:** {signal.get('created_at', '')}")

                # Show evidence
                try:
                    detail = api.get_signal(signal["id"])
                    evidence = detail.get("evidence", {})
                    if evidence:
                        st.json(evidence)
                except Exception:
                    pass
    else:
        st.info("No signals match the current filters.")

    # Alert configuration
    st.markdown("---")
    st.subheader("Alert Configurations")

    try:
        configs = api.get_signals()  # TODO: separate endpoint
    except Exception:
        configs = []

    with st.expander("Create New Alert Config"):
        name = st.text_input("Alert Name", placeholder="My NVDA alert")
        alert_types = st.multiselect(
            "Signal Types",
            ["trade_follow", "anomaly_alert", "sentiment_divergence",
             "insider_cluster", "network_signal"],
        )
        alert_min = st.slider("Min Strength for Alert", 0.0, 1.0, 0.5, 0.05, key="alert_min")
        alert_tickers = st.text_input("Tickers (comma-separated)", placeholder="NVDA,AAPL")
        webhook = st.text_input("Webhook URL (optional)", placeholder="https://hooks.slack.com/...")

        if st.button("Create Alert"):
            st.info(
                f"Alert '{name}' would be created via POST /alerts/configs. "
                "Connect to a running API to create alerts."
            )
