"""Screener page — 'What should I buy?' recommendation engine."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import CongressAPI


def render() -> None:
    st.title("Trade Screener")

    # Disclaimer banner
    st.warning(
        "**NOT financial advice.** This is an automated screening tool for research "
        "purposes only. Past congressional trading patterns do not guarantee future "
        "results. Always do your own due diligence before investing."
    )

    api = CongressAPI()

    # Sidebar controls
    st.sidebar.markdown("### Screener Settings")
    portfolio_size = st.sidebar.number_input(
        "Portfolio Size ($)", min_value=500, max_value=100000, value=3000, step=500
    )
    top_n = st.sidebar.slider("Number of Positions", 1, 10, 5)
    min_strength = st.sidebar.slider(
        "Min Signal Strength", 0.0, 1.0, 0.3, 0.05, key="screener_min_strength"
    )

    # Fetch recommendations
    try:
        data = api.get_recommendations(
            portfolio_size=portfolio_size,
            top_n=top_n,
            min_signal_strength=min_strength,
        )
    except Exception as e:
        st.error(f"Failed to load recommendations: {e}")
        return

    recs = data.get("recommendations", [])

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Recommendations", data.get("num_recommendations", 0))
    with col2:
        st.metric("Total Invested", f"${data.get('total_investable', 0):,.2f}")
    with col3:
        st.metric("Cash Remainder", f"${data.get('cash_remainder', 0):,.2f}")
    with col4:
        if recs:
            avg_score = sum(r["composite_score"] for r in recs) / len(recs)
            st.metric("Avg Score", f"{avg_score:.3f}")
        else:
            st.metric("Avg Score", "N/A")

    if not recs:
        st.info(
            "No bullish recommendations available. This may mean there are no active "
            "bullish signals — run the signal generation pipeline first."
        )
        return

    st.markdown("---")

    # Composite score bar chart
    import plotly.graph_objects as go

    tickers = [r["ticker"] for r in recs]
    scores = [r["composite_score"] for r in recs]

    fig_bar = go.Figure(
        data=[go.Bar(x=tickers, y=scores, marker_color="#2196F3")]
    )
    fig_bar.update_layout(
        title="Composite Scores",
        yaxis_title="Score",
        xaxis_title="Ticker",
        height=300,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Allocation pie chart
    labels = [r["ticker"] for r in recs]
    values = [r["allocation_pct"] * 100 for r in recs]

    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig_pie.update_layout(title="Portfolio Allocation", height=300)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # Recommendation cards
    st.subheader("Recommendations")
    for i, r in enumerate(recs):
        expanded = i < 3
        label = (
            f"{r['ticker']} — "
            f"Score: {r['composite_score']:.3f} | "
            f"${r.get('suggested_amount', 0):,.2f} | "
            f"{r.get('suggested_shares', 0)} shares"
        )
        with st.expander(label, expanded=expanded):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write(f"**Action:** {r.get('action', 'BUY')}")
                price = r.get("current_price")
                st.write(f"**Price:** ${price:,.2f}" if price else "**Price:** N/A")
                st.write(f"**Shares:** {r.get('suggested_shares', 0)}")
                st.write(f"**Amount:** ${r.get('suggested_amount', 0):,.2f}")
            with c2:
                st.write(f"**Allocation:** {r.get('allocation_pct', 0) * 100:.1f}%")
                st.write(f"**Signal Strength:** {r.get('avg_signal_strength', 0):.3f}")
                st.write(f"**ML Confidence:** {r.get('ml_confidence', 0):.3f}")
                st.write(f"**Win Rate:** {r.get('best_member_win_rate', 0):.3f}")
            with c3:
                risk = r.get("risk", {})
                vol = risk.get("volatility_21d")
                st.write(f"**Volatility:** {vol:.2%}" if vol else "**Volatility:** N/A")
                rsi = risk.get("rsi_14")
                st.write(f"**RSI(14):** {rsi:.1f}" if rsi else "**RSI(14):** N/A")
                pc5 = risk.get("price_change_5d")
                st.write(f"**5d Change:** {pc5:.2%}" if pc5 else "**5d Change:** N/A")
                st.write(f"**Freshness:** {r.get('freshness_score', 0):.3f}")

            # Evidence
            evidence = r.get("evidence", {})
            members = evidence.get("member_names", [])
            sig_types = evidence.get("signal_types", [])
            if members:
                st.write(f"**Who's Buying:** {', '.join(members)}")
            if sig_types:
                st.write(f"**Signal Types:** {', '.join(sig_types)}")

    # Footer disclaimer
    st.markdown("---")
    st.caption(data.get("disclaimer", "NOT financial advice."))
