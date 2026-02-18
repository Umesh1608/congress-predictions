"""Member Explorer page — drill-down into individual congress members."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import CongressAPI
from dashboard.charts import sentiment_timeline_chart, trade_timeline_chart


def render() -> None:
    st.title("Member Explorer")

    api = CongressAPI()

    try:
        members = api.get_members(limit=200)
    except Exception as e:
        st.error(f"Failed to load members: {e}")
        return

    if not members:
        st.info("No members available.")
        return

    # Member selector
    member_options = {
        f"{m['full_name']} ({m.get('party', '?')}-{m.get('state', '?')})": m["bioguide_id"]
        for m in members
    }
    selected_label = st.selectbox("Select Member", list(member_options.keys()))

    if not selected_label:
        return

    bioguide_id = member_options[selected_label]

    try:
        member = api.get_member(bioguide_id)
    except Exception:
        st.error("Failed to load member details")
        return

    # Member info cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Chamber", member.get("chamber", "").title())
    with col2:
        st.metric("Party", member.get("party", "N/A"))
    with col3:
        st.metric("State", member.get("state", "N/A"))
    with col4:
        st.metric("In Office", "Yes" if member.get("in_office") else "No")

    st.markdown("---")

    # Trades
    st.subheader("Trade History")
    try:
        trades = api.get_trades(member_bioguide_id=bioguide_id, limit=100)
        if trades:
            fig = trade_timeline_chart(trades, title=f"Trades — {member.get('full_name', '')}")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                [
                    {
                        "Date": t.get("transaction_date", ""),
                        "Ticker": t.get("ticker", ""),
                        "Type": t.get("transaction_type", ""),
                        "Amount": f"${t.get('amount_range_low', 0):,.0f}"
                        if t.get("amount_range_low")
                        else "N/A",
                        "Filer": t.get("filer_type", ""),
                    }
                    for t in trades
                ],
                use_container_width=True,
            )
        else:
            st.info("No trades found for this member.")
    except Exception:
        st.info("No trade data available.")

    # Sentiment timeline
    st.subheader("Sentiment Timeline")
    try:
        timeline = api.get_member_sentiment_timeline(bioguide_id)
        points = timeline.get("timeline", [])
        if points:
            fig = sentiment_timeline_chart(points, member.get("full_name", ""))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sentiment data available for this member.")
    except Exception:
        st.info("No sentiment data available.")

    # Predictions
    st.subheader("ML Predictions")
    try:
        predictions = api.get_predictions(member_bioguide_id=bioguide_id, limit=20)
        if predictions:
            st.dataframe(
                [
                    {
                        "Trade ID": p.get("trade_id"),
                        "Type": p.get("prediction_type", ""),
                        "Prediction": p.get("predicted_label", ""),
                        "Confidence": f"{p.get('confidence', 0):.2f}",
                        "Actual 5d": f"{p.get('actual_return_5d', 0):.2%}"
                        if p.get("actual_return_5d") is not None
                        else "Pending",
                    }
                    for p in predictions
                ],
                use_container_width=True,
            )
        else:
            st.info("No predictions yet.")
    except Exception:
        st.info("No prediction data available.")
