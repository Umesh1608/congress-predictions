"""Isolation Forest for detecting unusual trading patterns."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.ml.models.base import BasePredictor

logger = logging.getLogger(__name__)


class AnomalyDetector(BasePredictor):
    """Isolation Forest for detecting anomalous trading patterns.

    Uses only trade + timing features (not market features) to avoid
    future data leakage. Anomaly scores range from -1 (most anomalous)
    to 1 (most normal).
    """

    model_name = "anomaly_detector"

    # Features to exclude (market features would leak future info)
    EXCLUDED_FEATURES = {
        "price_change_5d",
        "price_change_21d",
        "volatility_21d",
        "volume_ratio_5d",
        "rsi_14",
    }

    def __init__(self, **kwargs: Any) -> None:
        self.params = {
            "n_estimators": kwargs.get("n_estimators", 200),
            "contamination": kwargs.get("contamination", 0.05),
            "random_state": kwargs.get("random_state", 42),
            "n_jobs": kwargs.get("n_jobs", -1),
        }
        self.model = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train Isolation Forest (unsupervised — y_train is ignored)."""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError(
                "scikit-learn is required for AnomalyDetector. "
                "Install with: pip install -e '.[ml]'"
            )

        self.model = IsolationForest(**self.params)
        self.model.fit(X_train)

        # Compute basic stats on training data
        scores = self.model.decision_function(X_train)
        labels = self.model.predict(X_train)
        anomaly_count = int((labels == -1).sum())
        total = len(labels)

        metrics = {
            "anomaly_rate": anomaly_count / total if total > 0 else 0,
            "anomaly_count": float(anomaly_count),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
        }

        logger.info("AnomalyDetector trained: %s", metrics)
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly labels: -1 for anomalous, 1 for normal."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (decision function).

        More negative = more anomalous. Rescaled to [0, 1] where
        1 = most anomalous.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        scores = self.model.decision_function(X)
        # Rescale: more negative → higher anomaly score
        # decision_function returns values where negative = anomalous
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return np.zeros_like(scores)
        normalized = (max_score - scores) / (max_score - min_score)
        return normalized

    @classmethod
    def filter_features(
        cls, feature_dict: dict[str, float]
    ) -> dict[str, float]:
        """Remove market features that would leak future info."""
        return {k: v for k, v in feature_dict.items() if k not in cls.EXCLUDED_FEATURES}
