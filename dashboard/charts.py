"""Reusable Plotly chart functions for the Streamlit dashboard."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def trade_timeline_chart(trades: list[dict], title: str = "Trade Timeline") -> go.Figure:
    """Create a scatter plot of trades over time."""
    fig = go.Figure()

    purchases = [t for t in trades if t.get("transaction_type") == "purchase"]
    sales = [t for t in trades if t.get("transaction_type") != "purchase"]

    if purchases:
        fig.add_trace(go.Scatter(
            x=[t.get("transaction_date") for t in purchases],
            y=[t.get("amount_range_low", 0) for t in purchases],
            mode="markers",
            name="Purchases",
            marker=dict(color="green", size=8),
            text=[f"{t.get('ticker', 'N/A')} - {t.get('member_name', '')}" for t in purchases],
            hovertemplate="%{text}<br>Date: %{x}<br>Amount: $%{y:,.0f}<extra></extra>",
        ))

    if sales:
        fig.add_trace(go.Scatter(
            x=[t.get("transaction_date") for t in sales],
            y=[t.get("amount_range_low", 0) for t in sales],
            mode="markers",
            name="Sales",
            marker=dict(color="red", size=8),
            text=[f"{t.get('ticker', 'N/A')} - {t.get('member_name', '')}" for t in sales],
            hovertemplate="%{text}<br>Date: %{x}<br>Amount: $%{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        yaxis_type="log",
        template="plotly_white",
        height=400,
    )
    return fig


def sentiment_timeline_chart(
    timeline: list[dict], member_name: str = ""
) -> go.Figure:
    """Create a line chart of sentiment over time."""
    fig = go.Figure()

    dates = [point.get("date") for point in timeline]
    scores = [point.get("avg_score", 0) for point in timeline]

    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode="lines+markers",
        name="Avg Sentiment",
        line=dict(color="blue", width=2),
        marker=dict(size=5),
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=f"Sentiment Timeline{f' — {member_name}' if member_name else ''}",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        yaxis_range=[-1, 1],
        template="plotly_white",
        height=350,
    )
    return fig


def signal_strength_chart(signals: list[dict]) -> go.Figure:
    """Create a bar chart of signal strengths by type."""
    from collections import defaultdict

    type_strengths: dict[str, list[float]] = defaultdict(list)
    for s in signals:
        type_strengths[s.get("signal_type", "unknown")].append(s.get("strength", 0))

    types = list(type_strengths.keys())
    avg_strengths = [
        sum(v) / len(v) if v else 0 for v in type_strengths.values()
    ]
    counts = [len(v) for v in type_strengths.values()]

    fig = go.Figure(data=[
        go.Bar(
            x=types,
            y=avg_strengths,
            text=[f"n={c}" for c in counts],
            textposition="auto",
            marker_color=["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6"][: len(types)],
        )
    ])

    fig.update_layout(
        title="Average Signal Strength by Type",
        xaxis_title="Signal Type",
        yaxis_title="Avg Strength",
        yaxis_range=[0, 1],
        template="plotly_white",
        height=350,
    )
    return fig


def model_performance_chart(models: list[dict]) -> go.Figure:
    """Create a grouped bar chart of model metrics."""
    fig = go.Figure()

    model_names = [m.get("model_name", "") for m in models]
    metrics_to_show = ["accuracy", "precision", "recall", "f1", "val_accuracy", "val_f1"]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]

    for i, metric in enumerate(metrics_to_show):
        values = [m.get("metrics", {}).get(metric, 0) for m in models]
        if any(v > 0 for v in values):
            fig.add_trace(go.Bar(
                x=model_names,
                y=values,
                name=metric,
                marker_color=colors[i % len(colors)],
            ))

    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Model",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        barmode="group",
        template="plotly_white",
        height=400,
    )
    return fig


def top_tickers_chart(stats: dict) -> go.Figure:
    """Create a horizontal bar chart of top traded tickers."""
    top_tickers = stats.get("top_tickers", [])

    tickers = [t.get("ticker", "") for t in top_tickers[:15]]
    counts = [t.get("count", 0) for t in top_tickers[:15]]

    fig = go.Figure(data=[
        go.Bar(
            y=tickers,
            x=counts,
            orientation="h",
            marker_color="#3498db",
        )
    ])

    fig.update_layout(
        title="Top 15 Traded Tickers",
        xaxis_title="Trade Count",
        template="plotly_white",
        height=400,
    )
    return fig
