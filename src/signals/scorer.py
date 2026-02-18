"""Composite signal scoring with lag penalties and corroboration bonuses."""

from __future__ import annotations

from typing import Any


def score_signal(signal_data: dict[str, Any]) -> dict[str, float]:
    """Compute composite strength and confidence scores for a signal.

    Adjustments:
    - Freshness bonus: +0.1 if disclosure lag < 7 days
    - Corroboration bonus: +0.15 per additional evidence source type
    - Lag penalty: multiply by max(0.3, 1 - lag_days/45)
    - Cap at 1.0

    Args:
        signal_data: Dict with signal_type, confidence, evidence, and
            optionally disclosure_lag_days.

    Returns:
        Dict with adjusted 'strength' and 'confidence'.
    """
    base_confidence = float(signal_data.get("confidence", 0.5))
    strength = base_confidence

    evidence = signal_data.get("evidence", {})
    signal_type = signal_data.get("signal_type", "")

    # Freshness bonus
    lag_days = signal_data.get("disclosure_lag_days")
    if lag_days is not None and lag_days < 7:
        strength += 0.1

    # Lag penalty
    if lag_days is not None and lag_days > 0:
        lag_multiplier = max(0.3, 1.0 - lag_days / 45.0)
        strength *= lag_multiplier

    # Corroboration bonus: count distinct evidence sources
    evidence_types = set()
    if "trade_id" in evidence:
        evidence_types.add("trade")
    if "prediction_id" in evidence:
        evidence_types.add("prediction")
    if "anomaly_score" in evidence:
        evidence_types.add("anomaly")
    if "avg_sentiment_30d" in evidence:
        evidence_types.add("sentiment")
    if "cluster_size" in evidence:
        evidence_types.add("cluster")
    if "lobbying_amount" in evidence:
        evidence_types.add("network")

    corroboration_bonus = max(0, (len(evidence_types) - 1)) * 0.15
    strength += corroboration_bonus

    # Signal type-specific adjustments
    if signal_type == "insider_cluster":
        cluster_size = evidence.get("cluster_size", 0)
        if cluster_size >= 5:
            strength += 0.2
        elif cluster_size >= 3:
            strength += 0.1

    # Cap at 1.0
    strength = min(strength, 1.0)
    confidence = min(base_confidence, 1.0)

    return {
        "strength": round(strength, 4),
        "confidence": round(confidence, 4),
    }
