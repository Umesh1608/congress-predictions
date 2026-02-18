"""Base predictor interface for all ML models."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BasePredictor(ABC):
    """Abstract base class for all ML models.

    Subclasses must implement train(), predict(), and predict_proba().
    save/load use pickle by default but can be overridden.
    """

    model: Any = None
    model_name: str = "base"
    feature_columns: list[str] = []

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train the model and return metrics dict."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates (for classifiers) or raw scores."""

    def save(self, path: str) -> None:
        """Save model to disk."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"model": self.model, "feature_columns": self.feature_columns}, f)

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        self.model = data["model"]
        self.feature_columns = data.get("feature_columns", [])

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores if available."""
        return {}
