"""AutoGluon TabularPredictor wrapper for automated model selection and stacking."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.models.base import BasePredictor

logger = logging.getLogger(__name__)


def _get_ag():
    """Lazy import AutoGluon."""
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        raise ImportError(
            "autogluon.tabular is required for AutoGluonPredictor. "
            "Install with: pip install -e '.[ml]'"
        )
    return TabularPredictor


class AutoGluonPredictor(BasePredictor):
    """AutoGluon wrapper providing automated model selection and multi-layer stacking.

    AutoGluon internally trains and ensembles LightGBM, CatBoost, XGBoost,
    NeuralNet, and other models with bagging and stacking.
    """

    model_name = "autogluon"

    def __init__(self, time_limit: int = 300, **kwargs: Any) -> None:
        self.time_limit = time_limit
        self.model = None  # Will be an AG TabularPredictor
        self._ag_dir: str | None = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train AutoGluon TabularPredictor with best_quality preset."""
        TabularPredictor = _get_ag()

        # Convert arrays to DataFrames
        col_names = self.feature_columns if self.feature_columns else [
            f"f{i}" for i in range(X_train.shape[1])
        ]
        train_df = pd.DataFrame(X_train, columns=col_names)
        train_df["target"] = y_train

        fit_kwargs: dict[str, Any] = {
            "time_limit": self.time_limit,
            "presets": "best_quality",
            "verbosity": 1,
        }

        predictor_kwargs: dict[str, Any] = {
            "label": "target",
            "eval_metric": "roc_auc",
        }

        if sample_weight is not None:
            train_df["weight"] = sample_weight
            predictor_kwargs["sample_weight"] = "weight"

        tuning_data = None
        if X_val is not None and y_val is not None:
            val_df = pd.DataFrame(X_val, columns=col_names)
            val_df["target"] = y_val
            tuning_data = val_df
            # best_quality uses bagging — need use_bag_holdout for tuning_data
            fit_kwargs["use_bag_holdout"] = True

        # Create a unique temp directory for AG models
        self._ag_dir = tempfile.mkdtemp(prefix="autogluon_")
        predictor_kwargs["path"] = self._ag_dir

        self.model = TabularPredictor(**predictor_kwargs).fit(
            train_data=train_df,
            tuning_data=tuning_data,
            **fit_kwargs,
        )

        # Compute validation metrics
        from src.ml.evaluation import evaluate_classifier

        if X_val is not None and y_val is not None:
            val_preds = self.predict(X_val)
            val_proba = self.predict_proba(X_val)
            metrics = evaluate_classifier(y_val, val_preds, val_proba)
        else:
            train_preds = self.predict(X_train)
            train_proba = self.predict_proba(X_train)
            metrics = evaluate_classifier(y_train, train_preds, train_proba)

        logger.info("AutoGluonPredictor trained: %s", metrics)
        logger.info("AutoGluon leaderboard:\n%s", self.model.leaderboard(silent=True).to_string())
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions (0 or 1)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        col_names = self.feature_columns if self.feature_columns else [
            f"f{i}" for i in range(X.shape[1])
        ]
        df = pd.DataFrame(X, columns=col_names)
        return self.model.predict(df).values.astype(float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of class 1 (profitable)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        col_names = self.feature_columns if self.feature_columns else [
            f"f{i}" for i in range(X.shape[1])
        ]
        df = pd.DataFrame(X, columns=col_names)
        proba = self.model.predict_proba(df)
        # AG returns DataFrame with columns [0, 1] for binary classification
        if isinstance(proba, pd.DataFrame):
            return proba[1].values if 1 in proba.columns else proba.iloc[:, -1].values
        return proba

    def save(self, path: str) -> None:
        """Save AutoGluon model directory and metadata."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # AG saves its own directory; copy it alongside the metadata file
        ag_save_dir = str(filepath.parent / "ag_models")
        if self._ag_dir and Path(self._ag_dir).exists():
            if Path(ag_save_dir).exists():
                shutil.rmtree(ag_save_dir)
            shutil.copytree(self._ag_dir, ag_save_dir)

        metadata = {
            "ag_dir": ag_save_dir,
            "feature_columns": self.feature_columns,
            "time_limit": self.time_limit,
        }
        with open(filepath, "w") as f:
            json.dump(metadata, f)

    def load(self, path: str) -> None:
        """Load AutoGluon model from saved directory."""
        TabularPredictor = _get_ag()

        with open(path) as f:
            metadata = json.load(f)

        self._ag_dir = metadata["ag_dir"]
        self.feature_columns = metadata.get("feature_columns", [])
        self.time_limit = metadata.get("time_limit", 300)
        self.model = TabularPredictor.load(self._ag_dir)

    def get_feature_importance(self) -> dict[str, float]:
        """Return AutoGluon feature importance."""
        if self.model is None:
            return {}
        try:
            importance = self.model.feature_importance(silent=True)
            return importance.to_dict()
        except Exception:
            return {}
