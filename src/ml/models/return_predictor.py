"""XGBoost regressor for predicting trade return magnitude."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.ml.models.base import BasePredictor

logger = logging.getLogger(__name__)


def _get_xgb():
    """Lazy import XGBoost."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError(
            "xgboost is required for ReturnPredictor. Install with: pip install -e '.[ml]'"
        )
    return xgb


class ReturnPredictor(BasePredictor):
    """XGBoost regressor predicting expected return magnitude.

    Predicts the 5-day and 21-day return magnitude (continuous value).
    """

    model_name = "return_predictor"

    def __init__(self, **kwargs: Any) -> None:
        self.params = {
            "n_estimators": kwargs.get("n_estimators", 300),
            "max_depth": kwargs.get("max_depth", 5),
            "learning_rate": kwargs.get("learning_rate", 0.1),
            "random_state": kwargs.get("random_state", 42),
            "verbosity": kwargs.get("verbosity", 0),
            "n_jobs": kwargs.get("n_jobs", -1),
        }
        self.model = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train XGBoost regressor."""
        xgb = _get_xgb()

        self.model = xgb.XGBRegressor(**self.params)

        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        self.model.fit(X_train, y_train, **fit_params)

        from src.ml.evaluation import evaluate_regressor

        train_preds = self.model.predict(X_train)
        metrics = evaluate_regressor(y_train, train_preds)

        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            val_metrics = evaluate_regressor(y_val, val_preds)
            metrics = {f"val_{k}": v for k, v in val_metrics.items()}

        logger.info("ReturnPredictor trained: %s", metrics)
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted returns."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """For regressors, return raw predictions as 'scores'."""
        return self.predict(X)

    def get_feature_importance(self) -> dict[str, float]:
        """Return XGBoost feature importance."""
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        if self.feature_columns:
            return dict(zip(self.feature_columns, importance.tolist()))
        return {f"feature_{i}": float(v) for i, v in enumerate(importance)}
