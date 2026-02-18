"""Model training orchestrator.

Manages the training pipeline: builds datasets, trains models with
walk-forward cross-validation, saves artifacts, and records results.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.ml.dataset import DatasetBuilder, TemporalSplitter
from src.ml.evaluation import evaluate_classifier, evaluate_regressor
from src.ml.models.anomaly_model import AnomalyDetector
from src.ml.models.base import BasePredictor
from src.ml.models.ensemble import EnsembleModel
from src.ml.models.return_predictor import ReturnPredictor
from src.ml.models.trade_predictor import TradePredictor
from src.models.ml import MLModelArtifact

logger = logging.getLogger(__name__)

ARTIFACT_BASE_DIR = "data/models"


class ModelTrainer:
    """Orchestrates training of all ML models."""

    def __init__(
        self,
        artifact_dir: str = ARTIFACT_BASE_DIR,
        n_splits: int = 5,
        train_months: int = 12,
        test_months: int = 3,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.splitter = TemporalSplitter(
            n_splits=n_splits,
            train_months=train_months,
            test_months=test_months,
        )
        self.dataset_builder = DatasetBuilder()

    async def train_all(self, session: AsyncSession) -> dict[str, dict[str, float]]:
        """Train all models and return their metrics.

        Returns dict mapping model_name → metrics.
        """
        logger.info("Building training dataset...")
        df = await self.dataset_builder.build_trade_dataset(session)

        if df.empty:
            logger.warning("No training data available. Skipping training.")
            return {}

        # Drop rows with missing labels
        df_labeled = df.dropna(subset=["profitable_5d"])
        if df_labeled.empty:
            logger.warning("No labeled data available. Skipping training.")
            return {}

        logger.info("Training dataset: %d samples", len(df_labeled))

        # Determine feature columns (everything except metadata and labels)
        exclude_cols = {
            "trade_id", "transaction_date", "ticker",
            "actual_return_5d", "actual_return_21d", "profitable_5d",
        }
        feature_cols = [c for c in df_labeled.columns if c not in exclude_cols]

        X = df_labeled[feature_cols].values.astype(float)
        y_class = df_labeled["profitable_5d"].values.astype(float)
        y_return = df_labeled["actual_return_5d"].fillna(0).values.astype(float)

        # Replace NaN with 0 in features
        X = np.nan_to_num(X, nan=0.0)

        results: dict[str, dict[str, float]] = {}
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # 1. Trade Predictor (LightGBM)
        logger.info("Training TradePredictor...")
        trade_pred = TradePredictor()
        trade_pred.feature_columns = feature_cols
        metrics = self._train_with_cv(trade_pred, X, y_class, "classifier")
        await self._save_artifact(session, trade_pred, version, metrics, feature_cols)
        results["trade_predictor"] = metrics

        # 2. Return Predictor (XGBoost)
        logger.info("Training ReturnPredictor...")
        return_pred = ReturnPredictor()
        return_pred.feature_columns = feature_cols
        metrics = self._train_with_cv(return_pred, X, y_return, "regressor")
        await self._save_artifact(session, return_pred, version, metrics, feature_cols)
        results["return_predictor"] = metrics

        # 3. Anomaly Detector (Isolation Forest)
        logger.info("Training AnomalyDetector...")
        # Filter out market features for anomaly detection
        anomaly_cols = [c for c in feature_cols if c not in AnomalyDetector.EXCLUDED_FEATURES]
        anomaly_col_idx = [feature_cols.index(c) for c in anomaly_cols]
        X_anomaly = X[:, anomaly_col_idx]

        anomaly_det = AnomalyDetector()
        anomaly_det.feature_columns = anomaly_cols
        metrics = anomaly_det.train(X_anomaly, y_class)
        await self._save_artifact(session, anomaly_det, version, metrics, anomaly_cols)
        results["anomaly_detector"] = metrics

        # 4. Ensemble (stacking)
        logger.info("Training EnsembleModel...")
        trade_proba = trade_pred.predict_proba(X)
        return_scores = return_pred.predict(X)
        anomaly_scores = anomaly_det.predict_proba(X_anomaly)

        # Get timing suspicion and sentiment from features
        timing_idx = feature_cols.index("timing_suspicion_score") if "timing_suspicion_score" in feature_cols else None
        sentiment_idx = feature_cols.index("avg_sentiment_7d") if "avg_sentiment_7d" in feature_cols else None

        meta_features = np.column_stack([
            trade_proba,
            return_scores,
            anomaly_scores,
            X[:, timing_idx] if timing_idx is not None else np.zeros(len(X)),
            X[:, sentiment_idx] if sentiment_idx is not None else np.zeros(len(X)),
        ])

        ensemble = EnsembleModel()
        ensemble.feature_columns = EnsembleModel.META_FEATURES
        metrics = self._train_with_cv(ensemble, meta_features, y_class, "classifier")
        await self._save_artifact(session, ensemble, version, metrics, EnsembleModel.META_FEATURES)
        results["ensemble"] = metrics

        logger.info("All models trained. Results: %s", results)
        return results

    def _train_with_cv(
        self,
        model: BasePredictor,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
    ) -> dict[str, float]:
        """Train model with simple train/test split.

        For walk-forward CV with actual dates, use the TemporalSplitter
        with a DataFrame. Here we do a simple 80/20 split.
        """
        n = len(X)
        split_idx = int(n * 0.8)

        if split_idx < 10 or (n - split_idx) < 5:
            # Not enough data for split, train on all
            model.train(X, y)
            if task_type == "classifier":
                preds = model.predict(X)
                proba = model.predict_proba(X)
                return evaluate_classifier(y, preds, proba)
            else:
                preds = model.predict(X)
                return evaluate_regressor(y, preds)

        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        model.train(X_train, y_train, X_val, y_val)

        # Evaluate on validation set
        val_preds = model.predict(X_val)
        if task_type == "classifier":
            val_proba = model.predict_proba(X_val)
            return evaluate_classifier(y_val, val_preds, val_proba)
        else:
            return evaluate_regressor(y_val, val_preds)

    async def _save_artifact(
        self,
        session: AsyncSession,
        model: BasePredictor,
        version: str,
        metrics: dict[str, float],
        feature_columns: list[str],
    ) -> None:
        """Save model to disk and record in database."""
        # Save to disk
        path = f"{self.artifact_dir}/{model.model_name}/{version}/model.pkl"
        model.save(path)

        # Deactivate previous active version
        await session.execute(
            update(MLModelArtifact)
            .where(MLModelArtifact.model_name == model.model_name)
            .where(MLModelArtifact.is_active.is_(True))
            .values(is_active=False)
        )

        # Record new artifact
        artifact = MLModelArtifact(
            model_name=model.model_name,
            model_version=version,
            artifact_path=path,
            metrics=metrics,
            feature_columns=feature_columns,
            training_config=getattr(model, "params", {}),
            trained_at=datetime.now(timezone.utc),
            is_active=True,
        )
        session.add(artifact)
        await session.commit()
        logger.info("Saved artifact: %s v%s → %s", model.model_name, version, path)
