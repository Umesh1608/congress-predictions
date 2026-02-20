"""Prediction service for scoring trades with trained ML models.

Lazy-loads active model artifacts from database, caches in memory,
and runs all models on individual or batched trades.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.ml.features import build_feature_vector
from src.ml.models.anomaly_model import AnomalyDetector
from src.ml.models.autogluon_predictor import AutoGluonPredictor
from src.ml.models.base import BasePredictor
from src.ml.models.ensemble import EnsembleModel
from src.ml.models.ft_transformer import FTTransformerPredictor
from src.ml.models.return_predictor import ReturnPredictor
from src.ml.models.trade_predictor import TradePredictor
from src.models.ml import MLModelArtifact, TradePrediction
from src.models.trade import TradeDisclosure

logger = logging.getLogger(__name__)

# Module-level singleton
_prediction_service: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    """Get or create the prediction service singleton."""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service


class PredictionService:
    """Runs all ML models on trades and stores predictions."""

    def __init__(self) -> None:
        self._models: dict[str, BasePredictor] = {}
        self._artifact_ids: dict[str, int] = {}
        self._loaded = False
        self._gnn_embeddings: dict | None = None

    async def _load_models(self, session: AsyncSession) -> None:
        """Load active model artifacts from database."""
        result = await session.execute(
            select(MLModelArtifact).where(MLModelArtifact.is_active.is_(True))
        )
        artifacts = result.scalars().all()

        model_classes: dict[str, type[BasePredictor]] = {
            "trade_predictor": TradePredictor,
            "return_predictor": ReturnPredictor,
            "anomaly_detector": AnomalyDetector,
            "ensemble": EnsembleModel,
            "autogluon": AutoGluonPredictor,
            "ft_transformer": FTTransformerPredictor,
        }

        for artifact in artifacts:
            cls = model_classes.get(artifact.model_name)
            if cls and artifact.artifact_path:
                try:
                    model = cls()
                    model.load(artifact.artifact_path)
                    model.feature_columns = artifact.feature_columns or []
                    self._models[artifact.model_name] = model
                    self._artifact_ids[artifact.model_name] = artifact.id
                    logger.info(
                        "Loaded model: %s v%s", artifact.model_name, artifact.model_version
                    )
                except Exception:
                    logger.exception("Failed to load model: %s", artifact.model_name)

        # Load GNN embeddings if FT-Transformer model is loaded
        if "ft_transformer" in self._models:
            try:
                from src.ml.gnn_embeddings import load_embeddings
                self._gnn_embeddings = load_embeddings()
            except Exception:
                logger.warning("Failed to load GNN embeddings for FT-Transformer")

        self._loaded = True

    async def predict_trade(
        self,
        session: AsyncSession,
        trade_id: int,
    ) -> list[dict[str, Any]]:
        """Run all models on a single trade and store predictions.

        Returns list of prediction dicts.
        """
        if not self._loaded:
            await self._load_models(session)

        if not self._models:
            logger.warning("No models loaded. Skipping prediction.")
            return []

        # Build feature vector
        features = await build_feature_vector(session, trade_id)
        if features is None:
            return []

        predictions: list[dict[str, Any]] = []

        # Get feature columns from trade predictor (the main model)
        trade_model = self._models.get("trade_predictor")
        if trade_model and trade_model.feature_columns:
            feature_cols = trade_model.feature_columns
            X = np.array([[features.get(c, 0.0) for c in feature_cols]])
            X = np.nan_to_num(X, nan=0.0)

            # Trade Predictor
            proba = trade_model.predict_proba(X)[0]
            label = "profitable" if proba > 0.5 else "unprofitable"
            pred = {
                "trade_id": trade_id,
                "model_artifact_id": self._artifact_ids.get("trade_predictor"),
                "prediction_type": "profitability",
                "predicted_value": float(proba),
                "predicted_label": label,
                "confidence": float(abs(proba - 0.5) * 2),
                "feature_vector": features,
            }
            predictions.append(pred)

            # Return Predictor
            return_model = self._models.get("return_predictor")
            if return_model:
                ret_pred = return_model.predict(X)[0]
                predictions.append({
                    "trade_id": trade_id,
                    "model_artifact_id": self._artifact_ids.get("return_predictor"),
                    "prediction_type": "return_5d",
                    "predicted_value": float(ret_pred),
                    "predicted_label": "positive" if ret_pred > 0 else "negative",
                    "confidence": min(abs(float(ret_pred)) * 10, 1.0),
                    "feature_vector": features,
                })

        # Anomaly Detector (uses filtered features)
        anomaly_model = self._models.get("anomaly_detector")
        if anomaly_model and anomaly_model.feature_columns:
            anomaly_features = AnomalyDetector.filter_features(features)
            X_anom = np.array([[anomaly_features.get(c, 0.0) for c in anomaly_model.feature_columns]])
            X_anom = np.nan_to_num(X_anom, nan=0.0)

            anomaly_score = anomaly_model.predict_proba(X_anom)[0]
            anomaly_label = anomaly_model.predict(X_anom)[0]
            predictions.append({
                "trade_id": trade_id,
                "model_artifact_id": self._artifact_ids.get("anomaly_detector"),
                "prediction_type": "anomaly",
                "predicted_value": float(anomaly_score),
                "predicted_label": "anomalous" if anomaly_label == -1 else "normal",
                "confidence": float(anomaly_score),
                "feature_vector": anomaly_features,
            })

        # AutoGluon predictor
        ag_model = self._models.get("autogluon")
        if ag_model and ag_model.feature_columns:
            ag_cols = ag_model.feature_columns
            X_ag = np.array([[features.get(c, 0.0) for c in ag_cols]])
            X_ag = np.nan_to_num(X_ag, nan=0.0)
            ag_proba = ag_model.predict_proba(X_ag)[0]
            ag_label = "profitable" if ag_proba > 0.5 else "unprofitable"
            predictions.append({
                "trade_id": trade_id,
                "model_artifact_id": self._artifact_ids.get("autogluon"),
                "prediction_type": "profitability",
                "predicted_value": float(ag_proba),
                "predicted_label": ag_label,
                "confidence": float(abs(ag_proba - 0.5) * 2),
                "feature_vector": features,
            })

        # FT-Transformer predictor (may need GNN embedding columns)
        ft_model = self._models.get("ft_transformer")
        if ft_model and ft_model.feature_columns:
            ft_cols = ft_model.feature_columns
            ft_features = dict(features)
            # Append GNN embeddings if available and model expects them
            gnn_cols = [c for c in ft_cols if c.startswith("gnn_")]
            if gnn_cols and self._gnn_embeddings:
                member_id = features.get("member_bioguide_id")
                ticker = features.get("ticker")
                member_embs = self._gnn_embeddings.get("Member", {})
                company_embs = self._gnn_embeddings.get("Company", {})
                m_emb = member_embs.get(member_id, np.zeros(64)) if member_id else np.zeros(64)
                c_emb = company_embs.get(ticker, np.zeros(64)) if ticker else np.zeros(64)
                for i in range(len(m_emb)):
                    ft_features[f"gnn_member_{i}"] = float(m_emb[i])
                for i in range(len(c_emb)):
                    ft_features[f"gnn_company_{i}"] = float(c_emb[i])
            X_ft = np.array([[ft_features.get(c, 0.0) for c in ft_cols]])
            X_ft = np.nan_to_num(X_ft, nan=0.0)
            ft_proba = ft_model.predict_proba(X_ft)[0]
            ft_label = "profitable" if ft_proba > 0.5 else "unprofitable"
            predictions.append({
                "trade_id": trade_id,
                "model_artifact_id": self._artifact_ids.get("ft_transformer"),
                "prediction_type": "profitability",
                "predicted_value": float(ft_proba),
                "predicted_label": ft_label,
                "confidence": float(abs(ft_proba - 0.5) * 2),
                "feature_vector": ft_features,
            })

        # Store predictions
        for pred in predictions:
            trade_prediction = TradePrediction(**pred)
            session.add(trade_prediction)

        await session.commit()
        return predictions

    async def batch_predict(
        self,
        session: AsyncSession,
        limit: int = 100,
    ) -> int:
        """Find trades without predictions and score them.

        Returns count of trades scored.
        """
        # Find trades without any predictions
        subquery = select(TradePrediction.trade_id).distinct()
        result = await session.execute(
            select(TradeDisclosure.id)
            .where(
                and_(
                    TradeDisclosure.ticker.isnot(None),
                    TradeDisclosure.id.notin_(subquery),
                )
            )
            .order_by(TradeDisclosure.transaction_date.desc())
            .limit(limit)
        )
        trade_ids = [row[0] for row in result.all()]

        if not trade_ids:
            logger.info("No unscored trades found.")
            return 0

        logger.info("Scoring %d trades...", len(trade_ids))
        scored = 0
        for trade_id in trade_ids:
            try:
                preds = await self.predict_trade(session, trade_id)
                if preds:
                    scored += 1
            except Exception:
                logger.exception("Failed to predict trade %d", trade_id)

        logger.info("Scored %d/%d trades", scored, len(trade_ids))
        return scored
