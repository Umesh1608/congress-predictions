"""Congress Predictions Dashboard — Streamlit multi-page application."""

import streamlit as st

st.set_page_config(
    page_title="Congress Predictions",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Navigation
pages = {
    "Overview": "dashboard.pages.overview",
    "Member Explorer": "dashboard.pages.member_explorer",
    "Trade Feed": "dashboard.pages.trade_feed",
    "Network Graph": "dashboard.pages.network_graph",
    "Signals": "dashboard.pages.signals",
    "Backtesting": "dashboard.pages.backtesting",
}

st.sidebar.title("Congress Predictions")
st.sidebar.markdown("---")

selection = st.sidebar.radio("Navigation", list(pages.keys()))

st.sidebar.markdown("---")
st.sidebar.caption("Congressional trade tracking and prediction system")
st.sidebar.caption("NOT financial advice. For research only.")

# Load selected page
if selection == "Overview":
    from dashboard.pages.overview import render
    render()
elif selection == "Member Explorer":
    from dashboard.pages.member_explorer import render
    render()
elif selection == "Trade Feed":
    from dashboard.pages.trade_feed import render
    render()
elif selection == "Network Graph":
    from dashboard.pages.network_graph import render
    render()
elif selection == "Signals":
    from dashboard.pages.signals import render
    render()
elif selection == "Backtesting":
    from dashboard.pages.backtesting import render
    render()
