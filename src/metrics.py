"""Prometheus metrics for the Congress Predictions API."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---------- Counters ----------

trades_ingested_total = Counter(
    "trades_ingested_total",
    "Total number of trade disclosures ingested",
    ["source"],
)

predictions_generated_total = Counter(
    "predictions_generated_total",
    "Total ML predictions generated",
    ["prediction_type"],
)

signals_generated_total = Counter(
    "signals_generated_total",
    "Total signals generated",
    ["signal_type"],
)

# ---------- Histograms ----------

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Time to generate a single trade prediction",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

feature_computation_seconds = Histogram(
    "feature_computation_seconds",
    "Time to compute feature vector for a trade",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ---------- Gauges ----------

active_signals_count = Gauge(
    "active_signals_count",
    "Current number of active signals",
    ["signal_type"],
)

model_accuracy_current = Gauge(
    "model_accuracy_current",
    "Current accuracy of active ML models",
    ["model_name"],
)
