"""Trade Feed page — filterable live trade table with predictions."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from dashboard.api_client import CongressAPI


def render() -> None:
    st.title("Trade Feed")

    api = CongressAPI()

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ticker = st.text_input("Ticker", placeholder="e.g., NVDA")

    with col2:
        tx_type = st.selectbox(
            "Transaction Type",
            ["All", "purchase", "sale", "sale_full", "sale_partial", "exchange"],
        )

    with col3:
        date_from = st.date_input("From Date", value=date.today() - timedelta(days=90))

    with col4:
        date_to = st.date_input("To Date", value=date.today())

    # Build params
    params: dict = {"limit": 100}
    if ticker:
        params["ticker"] = ticker.upper()
    if tx_type != "All":
        params["transaction_type"] = tx_type
    if date_from:
        params["date_from"] = date_from.isoformat()
    if date_to:
        params["date_to"] = date_to.isoformat()

    try:
        trades = api.get_trades(**params)
    except Exception as e:
        st.error(f"Failed to load trades: {e}")
        return

    if not trades:
        st.info("No trades match the current filters.")
        return

    st.subheader(f"Showing {len(trades)} trades")

    st.dataframe(
        [
            {
                "Date": t.get("transaction_date", ""),
                "Disclosure": t.get("disclosure_date", ""),
                "Member": t.get("member_name", ""),
                "Chamber": t.get("chamber", ""),
                "Ticker": t.get("ticker", ""),
                "Asset": t.get("asset_name", "")[:50],
                "Type": t.get("transaction_type", ""),
                "Filer": t.get("filer_type", ""),
                "Amount Low": f"${t.get('amount_range_low', 0):,.0f}"
                if t.get("amount_range_low")
                else "N/A",
                "Amount High": f"${t.get('amount_range_high', 0):,.0f}"
                if t.get("amount_range_high")
                else "N/A",
                "Source": t.get("source", ""),
            }
            for t in trades
        ],
        use_container_width=True,
        height=600,
    )

    # Legislative context for selected trade
    st.markdown("---")
    st.subheader("Legislative Context")

    trade_ids = [t.get("id") for t in trades if t.get("id")]
    if trade_ids:
        selected_id = st.selectbox("Select trade for context", trade_ids[:20])
        if selected_id:
            try:
                context = api.get_trade_context(selected_id)
                if context:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            "Suspicion Score",
                            f"{context.get('timing_suspicion_score', 0):.2f}",
                        )
                        st.metric(
                            "Disclosure Lag",
                            f"{context.get('disclosure_lag_days', 'N/A')} days",
                        )
                    with col_b:
                        hearings = context.get("nearby_hearings", [])
                        bills = context.get("nearby_bills", [])
                        st.metric("Nearby Hearings", len(hearings))
                        st.metric("Nearby Bills", len(bills))

                    if hearings:
                        st.write("**Nearby Hearings:**")
                        for h in hearings[:5]:
                            st.write(
                                f"- {h.get('title', 'N/A')} "
                                f"({h.get('distance_days', '?')} days)"
                            )
            except Exception:
                st.info("No legislative context available for this trade.")
