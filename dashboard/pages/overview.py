"""Overview page — system dashboard with KPIs and recent activity."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import CongressAPI
from dashboard.charts import signal_strength_chart, top_tickers_chart


def render() -> None:
    st.title("System Overview")

    api = CongressAPI()

    try:
        trade_stats = api.get_trade_stats()
        signal_stats = api.get_signal_stats()
        prediction_stats = api.get_prediction_stats()
        media_stats = api.get_media_stats()
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
        st.info("Make sure the FastAPI server is running at the configured API_BASE_URL.")
        return

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_trades = trade_stats.get("total_trades", 0)
        st.metric("Total Trades", f"{total_trades:,}")

    with col2:
        active_signals = signal_stats.get("total_active", 0)
        st.metric("Active Signals", active_signals)

    with col3:
        accuracy = prediction_stats.get("accuracy_5d")
        st.metric("5d Accuracy", f"{accuracy:.1%}" if accuracy else "N/A")

    with col4:
        media_count = media_stats.get("total_content", 0)
        st.metric("Media Items", f"{media_count:,}")

    st.markdown("---")

    # Two column layout
    left, right = st.columns(2)

    with left:
        st.subheader("Top Traded Tickers")
        fig = top_tickers_chart(trade_stats)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Active Signals")
        try:
            signals = api.get_signals(limit=20)
            if signals:
                fig = signal_strength_chart(signals)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No active signals")
        except Exception:
            st.info("No signal data available")

    # Recent trades table
    st.subheader("Recent Trades")
    try:
        trades = api.get_trades(limit=20)
        if trades:
            st.dataframe(
                [
                    {
                        "Date": t.get("transaction_date", ""),
                        "Member": t.get("member_name", ""),
                        "Ticker": t.get("ticker", ""),
                        "Type": t.get("transaction_type", ""),
                        "Amount Low": f"${t.get('amount_range_low', 0):,.0f}"
                        if t.get("amount_range_low")
                        else "N/A",
                    }
                    for t in trades
                ],
                use_container_width=True,
            )
        else:
            st.info("No trade data available")
    except Exception:
        st.info("No trade data available")

    # Data freshness
    st.subheader("Data Sources")
    source_counts = media_stats.get("by_source_type", {})
    if source_counts:
        cols = st.columns(len(source_counts))
        for i, (source, count) in enumerate(source_counts.items()):
            with cols[i]:
                st.metric(source, count)
