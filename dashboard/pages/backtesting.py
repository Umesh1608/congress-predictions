"""Backtesting page — model performance and historical accuracy."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import CongressAPI
from dashboard.charts import model_performance_chart


def render() -> None:
    st.title("Model Performance & Backtesting")

    api = CongressAPI()

    # Model performance
    st.subheader("Active Models")
    try:
        models = api.get_model_performance()
        if models:
            fig = model_performance_chart(models)
            st.plotly_chart(fig, use_container_width=True)

            for model in models:
                with st.expander(f"{model.get('model_name', '')} — v{model.get('model_version', '')}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Trained:** {model.get('trained_at', 'N/A')}")
                        st.write(f"**Active:** {model.get('is_active', False)}")
                        st.write(f"**Features:** {len(model.get('feature_columns', []))}")
                    with col2:
                        metrics = model.get("metrics", {})
                        for key, value in metrics.items():
                            st.write(f"**{key}:** {value:.4f}" if isinstance(value, float) else f"**{key}:** {value}")
        else:
            st.info("No models have been trained yet.")
    except Exception as e:
        st.error(f"Failed to load model performance: {e}")

    st.markdown("---")

    # Prediction stats
    st.subheader("Prediction Statistics")
    try:
        pred_stats = api.get_prediction_stats()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", f"{pred_stats.get('total_predictions', 0):,}")
        with col2:
            st.metric("Avg Confidence", f"{pred_stats.get('avg_confidence', 0):.2f}")
        with col3:
            acc = pred_stats.get("accuracy_5d")
            st.metric("5-Day Accuracy", f"{acc:.1%}" if acc is not None else "N/A")

        by_type = pred_stats.get("predictions_by_type", {})
        if by_type:
            st.write("**Predictions by Type:**")
            for ptype, count in by_type.items():
                st.write(f"- {ptype}: {count:,}")
    except Exception:
        st.info("No prediction statistics available.")

    st.markdown("---")

    # Leaderboard
    st.subheader("Member Leaderboard")
    st.caption("Members ranked by trade prediction accuracy (min 5 trades)")
    try:
        leaderboard = api.get_leaderboard(limit=20)
        if leaderboard:
            st.dataframe(
                [
                    {
                        "Member": entry.get("member_name", ""),
                        "Total Trades": entry.get("total_trades", 0),
                        "Predicted Profitable": entry.get("predicted_profitable", 0),
                        "Actually Profitable": entry.get("actual_profitable", 0),
                        "Accuracy": f"{entry.get('accuracy', 0):.1%}",
                        "Avg Confidence": f"{entry.get('avg_confidence', 0):.2f}",
                    }
                    for entry in leaderboard
                ],
                use_container_width=True,
            )
        else:
            st.info("Not enough data for leaderboard (need 5+ trades per member).")
    except Exception:
        st.info("No leaderboard data available.")
