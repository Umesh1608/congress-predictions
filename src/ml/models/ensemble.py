"""Stacking ensemble combining all model outputs into a unified score."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.ml.models.base import BasePredictor

logger = logging.getLogger(__name__)


class EnsembleModel(BasePredictor):
    """Logistic Regression meta-learner stacking model outputs.

    Inputs:
    - trade_predictor probability
    - return_predictor predicted return
    - anomaly_detector anomaly score
    - timing_suspicion_score (direct feature)
    - avg_sentiment_7d (direct feature)

    Outputs: unified trade_signal_score [0, 1]
    """

    model_name = "ensemble"

    META_FEATURES = [
        "trade_predictor_proba",
        "return_predictor_score",
        "anomaly_score",
        "timing_suspicion_score",
        "avg_sentiment_7d",
    ]

    def __init__(self, **kwargs: Any) -> None:
        self.params = {
            "C": kwargs.get("C", 1.0),
            "random_state": kwargs.get("random_state", 42),
            "max_iter": kwargs.get("max_iter", 1000),
        }
        self.model = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train Logistic Regression meta-learner."""
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            raise ImportError(
                "scikit-learn is required for EnsembleModel. "
                "Install with: pip install -e '.[ml]'"
            )

        self.model = LogisticRegression(**self.params)
        self.model.fit(X_train, y_train)

        from src.ml.evaluation import evaluate_classifier

        train_preds = self.model.predict(X_train)
        train_proba = self.model.predict_proba(X_train)[:, 1]
        metrics = evaluate_classifier(y_train, train_preds, train_proba)

        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            val_proba = self.model.predict_proba(X_val)[:, 1]
            val_metrics = evaluate_classifier(y_val, val_preds, val_proba)
            metrics = {f"val_{k}": v for k, v in val_metrics.items()}

        logger.info("EnsembleModel trained: %s", metrics)
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of profitable signal (class 1)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba

    @classmethod
    def build_meta_features(
        cls,
        trade_proba: float,
        return_score: float,
        anomaly_score: float,
        timing_suspicion: float = 0.0,
        avg_sentiment: float = 0.0,
    ) -> np.ndarray:
        """Build the meta-feature vector for ensemble prediction."""
        return np.array([
            [trade_proba, return_score, anomaly_score, timing_suspicion, avg_sentiment]
        ])
