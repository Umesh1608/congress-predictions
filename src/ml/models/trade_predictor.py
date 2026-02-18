"""LightGBM classifier for predicting trade profitability."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.ml.models.base import BasePredictor

logger = logging.getLogger(__name__)

_lgbm_model = None


def _get_lgbm():
    """Lazy import LightGBM."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError(
            "lightgbm is required for TradePredictor. Install with: pip install -e '.[ml]'"
        )
    return lgb


class TradePredictor(BasePredictor):
    """LightGBM classifier predicting whether a trade will be profitable.

    Predicts probability that a trade is profitable at 5-day and 21-day horizons.
    """

    model_name = "trade_predictor"

    def __init__(self, **kwargs: Any) -> None:
        self.params = {
            "n_estimators": kwargs.get("n_estimators", 500),
            "max_depth": kwargs.get("max_depth", 6),
            "learning_rate": kwargs.get("learning_rate", 0.05),
            "class_weight": kwargs.get("class_weight", "balanced"),
            "random_state": kwargs.get("random_state", 42),
            "verbose": kwargs.get("verbose", -1),
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
        """Train LightGBM classifier."""
        lgb = _get_lgbm()

        self.model = lgb.LGBMClassifier(**self.params)

        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]

        self.model.fit(X_train, y_train, **fit_params)

        # Compute training metrics
        from src.ml.evaluation import evaluate_classifier

        train_preds = self.model.predict(X_train)
        train_proba = self.model.predict_proba(X_train)[:, 1]
        metrics = evaluate_classifier(y_train, train_preds, train_proba)

        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            val_proba = self.model.predict_proba(X_val)[:, 1]
            val_metrics = evaluate_classifier(y_val, val_preds, val_proba)
            metrics = {f"val_{k}": v for k, v in val_metrics.items()}

        logger.info("TradePredictor trained: %s", metrics)
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions (0 or 1)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of profitable trade (class 1)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba

    def get_feature_importance(self) -> dict[str, float]:
        """Return LightGBM feature importance."""
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        if self.feature_columns:
            return dict(zip(self.feature_columns, importance.tolist()))
        return {f"feature_{i}": float(v) for i, v in enumerate(importance)}
