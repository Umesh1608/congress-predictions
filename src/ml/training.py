"""Model training orchestrator.

Manages the training pipeline: builds datasets using fast bulk SQL,
trains models with temporal cross-validation, tunes hyperparameters
with Optuna, saves artifacts, and records results.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src.ml.dataset_fast import build_dataset_fast, feature_columns
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
    """Orchestrates training of all ML models.

    Uses the fast bulk SQL dataset builder and temporal cross-validation.
    Supports CatBoost and Optuna hyperparameter tuning.
    """

    def __init__(
        self,
        artifact_dir: str = ARTIFACT_BASE_DIR,
        horizon: str = "180d",
        n_folds: int = 5,
        use_catboost: bool = True,
        use_optuna: bool = False,
        optuna_trials: int = 50,
        limit: int = 20000,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.horizon = horizon
        self.n_folds = n_folds
        self.use_catboost = use_catboost
        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials
        self.limit = limit

    async def train_all(self, session: AsyncSession) -> dict[str, dict[str, float]]:
        """Train all models and return their metrics.

        Returns dict mapping model_name -> averaged CV metrics.
        """
        logger.info("Building training dataset (horizon=%s)...", self.horizon)
        df = await build_dataset_fast(session, limit=self.limit, horizon=self.horizon)

        if df.empty:
            logger.warning("No training data available. Skipping training.")
            return {}

        logger.info("Training dataset: %d samples", len(df))

        # Optuna tuning (optional, run before main training)
        tuned_lgbm_params: dict = {}
        tuned_catboost_params: dict = {}
        if self.use_optuna:
            feat_cols = feature_columns(df)
            X_raw = df[feat_cols].values.astype(float)
            X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
            variances = np.var(X_raw, axis=0)
            keep_mask = variances > 1e-10
            X_tune = X_raw[:, keep_mask]
            y_tune = df["profitable"].values.astype(float)
            w_tune = df["sample_weight"].values.astype(float)

            logger.info("Running Optuna LightGBM tuning (%d trials)...", self.optuna_trials)
            tuned_lgbm_params = optuna_tune_lgbm(
                X_tune, y_tune, w_tune, n_folds=3, n_trials=self.optuna_trials
            )

            if self.use_catboost:
                logger.info("Running Optuna CatBoost tuning (%d trials)...", self.optuna_trials)
                tuned_catboost_params = optuna_tune_catboost(
                    X_tune, y_tune, w_tune, n_folds=3, n_trials=self.optuna_trials
                )

        # Run temporal CV training
        results = await train_models_cv(
            df,
            session,
            n_folds=self.n_folds,
            use_catboost=self.use_catboost,
            tuned_lgbm_params=tuned_lgbm_params,
            tuned_catboost_params=tuned_catboost_params,
            artifact_dir=self.artifact_dir,
        )

        logger.info("All models trained. Results: %s", results)
        return results


async def train_models_cv(
    df: pd.DataFrame,
    session: AsyncSession,
    n_folds: int = 5,
    use_catboost: bool = False,
    tuned_lgbm_params: dict | None = None,
    tuned_catboost_params: dict | None = None,
    artifact_dir: str = ARTIFACT_BASE_DIR,
) -> dict[str, dict[str, float]]:
    """Train all models with temporal k-fold cross-validation.

    Data is sorted by transaction_date. For k folds, each fold uses
    the first (i+1)/k of data for training and the next chunk for validation.
    This ensures no future leakage — always train on past, test on future.
    """
    feat_cols = feature_columns(df)

    # Feature selection: drop near-zero variance features
    X_raw = df[feat_cols].values.astype(float)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    variances = np.var(X_raw, axis=0)
    keep_mask = variances > 1e-10
    dropped = [feat_cols[i] for i in range(len(feat_cols)) if not keep_mask[i]]
    if dropped:
        logger.info("Dropping %d zero-variance features: %s", len(dropped), dropped)
    feat_cols = [feat_cols[i] for i in range(len(feat_cols)) if keep_mask[i]]

    X = X_raw[:, keep_mask]
    y_class = df["profitable"].values.astype(float)
    y_return = df["actual_return"].fillna(0).values.astype(float)
    sample_weights = df["sample_weight"].values.astype(float)

    n = len(X)
    logger.info("Features (%d): %s", len(feat_cols), feat_cols)
    logger.info("Running %d-fold temporal cross-validation on %d samples", n_folds, n)

    # Temporal CV: fold i trains on [0, split_i), validates on [split_i, split_i+1)
    min_train_frac = 0.5
    fold_size = int(n * (1 - min_train_frac) / n_folds)

    model_names = ["trade_predictor", "return_predictor", "anomaly_detector", "ensemble"]
    if use_catboost:
        model_names.append("catboost")
    fold_metrics: dict[str, list[dict]] = {name: [] for name in model_names}

    for fold in range(n_folds):
        train_end = int(n * min_train_frac) + fold * fold_size
        val_end = min(train_end + fold_size, n)
        if train_end >= n or val_end <= train_end:
            break

        X_train, X_val = X[:train_end], X[train_end:val_end]
        y_class_train, y_class_val = y_class[:train_end], y_class[train_end:val_end]
        y_return_train, y_return_val = y_return[:train_end], y_return[train_end:val_end]
        w_train = sample_weights[:train_end]

        logger.info(
            "  Fold %d/%d: train=%d, val=%d",
            fold + 1, n_folds, len(X_train), len(X_val),
        )

        # Trade Predictor (LightGBM) with sample weights
        lgbm_kwargs = dict(tuned_lgbm_params) if tuned_lgbm_params else {}
        tp = TradePredictor(**lgbm_kwargs)
        tp.feature_columns = feat_cols
        tp.train(X_train, y_class_train, X_val, y_class_val, sample_weight=w_train)
        m = evaluate_classifier(y_class_val, tp.predict(X_val), tp.predict_proba(X_val))
        fold_metrics["trade_predictor"].append(m)

        # Return Predictor (XGBoost) with sample weights
        rp = ReturnPredictor()
        rp.feature_columns = feat_cols
        rp.train(X_train, y_return_train, X_val, y_return_val, sample_weight=w_train)
        m = evaluate_regressor(y_return_val, rp.predict(X_val))
        fold_metrics["return_predictor"].append(m)

        # Anomaly Detector
        anomaly_cols = [c for c in feat_cols if c not in AnomalyDetector.EXCLUDED_FEATURES]
        anomaly_idx = [feat_cols.index(c) for c in anomaly_cols]
        X_a_train = X_train[:, anomaly_idx]
        ad = AnomalyDetector()
        ad.feature_columns = anomaly_cols
        m = ad.train(X_a_train, y_class_train)
        fold_metrics["anomaly_detector"].append(m)

        # CatBoost (optional)
        if use_catboost:
            m = _train_catboost_fold(
                X_train, y_class_train, X_val, y_class_val, feat_cols,
                sample_weight=w_train, tuned_params=tuned_catboost_params,
            )
            fold_metrics["catboost"].append(m)

        # Ensemble
        tp_proba = np.concatenate([tp.predict_proba(X_train), tp.predict_proba(X_val)])
        rp_scores = np.concatenate([rp.predict(X_train), rp.predict(X_val)])
        X_a_all = np.vstack([X_a_train, X_val[:, anomaly_idx]])
        ad_scores = ad.predict_proba(X_a_all)

        timing_idx = feat_cols.index("timing_suspicion_score") if "timing_suspicion_score" in feat_cols else None
        sentiment_idx = feat_cols.index("avg_sentiment_7d") if "avg_sentiment_7d" in feat_cols else None
        all_X = np.vstack([X_train, X_val])

        meta_cols = [tp_proba, rp_scores, ad_scores]
        meta_cols.append(all_X[:, timing_idx] if timing_idx is not None else np.zeros(len(all_X)))
        meta_cols.append(all_X[:, sentiment_idx] if sentiment_idx is not None else np.zeros(len(all_X)))

        meta = np.column_stack(meta_cols)
        meta_train, meta_val = meta[:len(X_train)], meta[len(X_train):]
        ens = EnsembleModel()
        ens.feature_columns = EnsembleModel.META_FEATURES
        ens.train(meta_train, y_class_train, meta_val, y_class_val)
        m = evaluate_classifier(y_class_val, ens.predict(meta_val), ens.predict_proba(meta_val))
        fold_metrics["ensemble"].append(m)

    # Average metrics across folds
    results: dict[str, dict[str, float]] = {}
    for model_name, folds in fold_metrics.items():
        if not folds:
            continue
        avg: dict[str, float] = {}
        for key in folds[0]:
            vals = [f[key] for f in folds if key in f and f[key] is not None]
            if vals:
                avg[key] = float(np.mean(vals))
                avg[f"{key}_std"] = float(np.std(vals))
        results[model_name] = avg

    # Log CV results
    for model_name, metrics in results.items():
        if "accuracy" in metrics:
            logger.info(
                "  %s (CV avg): accuracy=%.4f+-%.4f, AUC=%.4f+-%.4f, F1=%.4f+-%.4f",
                model_name,
                metrics.get("accuracy", 0), metrics.get("accuracy_std", 0),
                metrics.get("auc", 0), metrics.get("auc_std", 0),
                metrics.get("f1", 0), metrics.get("f1_std", 0),
            )

    # Feature importance from last LightGBM fold
    if hasattr(tp, "model") and hasattr(tp.model, "feature_importances_"):
        importances = tp.model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        logger.info("  Top 15 features (LightGBM):")
        for i in top_idx:
            logger.info("    %3d. %-35s %6.1f", i, feat_cols[i], importances[i])

    # Final train on all data (80/20 split for saved model)
    split_idx = int(n * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_class_train, y_class_val = y_class[:split_idx], y_class[split_idx:]
    y_return_train, y_return_val = y_return[:split_idx], y_return[split_idx:]
    w_final_train = sample_weights[:split_idx]

    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Save final models
    lgbm_kwargs = dict(tuned_lgbm_params) if tuned_lgbm_params else {}
    trade_pred = TradePredictor(**lgbm_kwargs)
    trade_pred.feature_columns = feat_cols
    trade_pred.train(X_train, y_class_train, X_val, y_class_val, sample_weight=w_final_train)
    await _save_artifact(session, trade_pred, version, results["trade_predictor"], feat_cols, artifact_dir)

    return_pred = ReturnPredictor()
    return_pred.feature_columns = feat_cols
    return_pred.train(X_train, y_return_train, X_val, y_return_val, sample_weight=w_final_train)
    await _save_artifact(session, return_pred, version, results["return_predictor"], feat_cols, artifact_dir)

    anomaly_cols = [c for c in feat_cols if c not in AnomalyDetector.EXCLUDED_FEATURES]
    anomaly_idx = [feat_cols.index(c) for c in anomaly_cols]
    anomaly_det = AnomalyDetector()
    anomaly_det.feature_columns = anomaly_cols
    anomaly_det.train(X[:, anomaly_idx], y_class)
    await _save_artifact(session, anomaly_det, version, results["anomaly_detector"], anomaly_cols, artifact_dir)

    # Ensemble on final split
    tp_proba = trade_pred.predict_proba(X)
    rp_scores = return_pred.predict(X)
    ad_scores = anomaly_det.predict_proba(X[:, anomaly_idx])
    timing_idx = feat_cols.index("timing_suspicion_score") if "timing_suspicion_score" in feat_cols else None
    sentiment_idx = feat_cols.index("avg_sentiment_7d") if "avg_sentiment_7d" in feat_cols else None
    meta = np.column_stack([
        tp_proba, rp_scores, ad_scores,
        X[:, timing_idx] if timing_idx is not None else np.zeros(n),
        X[:, sentiment_idx] if sentiment_idx is not None else np.zeros(n),
    ])
    ensemble = EnsembleModel()
    ensemble.feature_columns = EnsembleModel.META_FEATURES
    ensemble.train(meta[:split_idx], y_class_train, meta[split_idx:], y_class_val)
    await _save_artifact(session, ensemble, version, results["ensemble"], EnsembleModel.META_FEATURES, artifact_dir)

    return results


def _train_catboost_fold(
    X_train, y_train, X_val, y_val, feature_cols,
    sample_weight=None, tuned_params=None,
) -> dict[str, float]:
    """Train a CatBoost classifier for one CV fold."""
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        logger.warning("CatBoost not installed — skipping. Install with: pip install catboost")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0}

    cb_params = {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "auto_class_weights": "Balanced",
        "random_seed": 42,
        "verbose": 0,
        "eval_metric": "AUC",
    }
    if tuned_params:
        cb_params.update(tuned_params)

    model = CatBoostClassifier(**cb_params)
    train_pool = Pool(X_train, y_train, weight=sample_weight)
    val_pool = Pool(X_val, y_val)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=0)

    preds = model.predict(X_val).astype(float)
    proba = model.predict_proba(X_val)[:, 1]
    return evaluate_classifier(y_val, preds, proba)


async def _save_artifact(
    session: AsyncSession,
    model: BasePredictor,
    version: str,
    metrics: dict[str, float],
    feature_cols: list[str],
    artifact_dir: str = ARTIFACT_BASE_DIR,
) -> None:
    """Save model to disk and record in database."""
    path = f"{artifact_dir}/{model.model_name}/{version}/model.pkl"
    model.save(path)

    await session.execute(
        update(MLModelArtifact)
        .where(MLModelArtifact.model_name == model.model_name)
        .where(MLModelArtifact.is_active.is_(True))
        .values(is_active=False)
    )

    artifact = MLModelArtifact(
        model_name=model.model_name,
        model_version=version,
        artifact_path=path,
        metrics=metrics,
        feature_columns=feature_cols,
        training_config=getattr(model, "params", {}),
        trained_at=datetime.now(timezone.utc),
        is_active=True,
    )
    session.add(artifact)
    await session.commit()
    logger.info("Saved artifact: %s v%s -> %s", model.model_name, version, path)


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning
# ---------------------------------------------------------------------------

def optuna_tune_lgbm(
    X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray,
    n_folds: int = 3, n_trials: int = 50,
) -> dict:
    """Use Optuna to find optimal LightGBM hyperparameters with temporal CV."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed — using defaults. Install with: pip install optuna")
        return {}

    import lightgbm as lgb

    n = len(X)
    min_train_frac = 0.5
    fold_size = int(n * (1 - min_train_frac) / n_folds)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }

        aucs = []
        for fold in range(n_folds):
            train_end = int(n * min_train_frac) + fold * fold_size
            val_end = min(train_end + fold_size, n)
            if train_end >= n or val_end <= train_end:
                break

            X_train, X_val = X[:train_end], X[train_end:val_end]
            y_train, y_val = y[:train_end], y[train_end:val_end]
            w_train = sample_weights[:train_end]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], sample_weight=w_train)

            from sklearn.metrics import roc_auc_score
            proba = model.predict_proba(X_val)[:, 1]
            try:
                auc = roc_auc_score(y_val, proba)
            except ValueError:
                auc = 0.5
            aucs.append(auc)

        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Optuna LightGBM best AUC: %.4f", study.best_value)
    logger.info("Optuna LightGBM best params: %s", study.best_params)
    return study.best_params


def optuna_tune_catboost(
    X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray,
    n_folds: int = 3, n_trials: int = 50,
) -> dict:
    """Use Optuna to find optimal CatBoost hyperparameters with temporal CV."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        logger.warning("Optuna/CatBoost not installed — using defaults")
        return {}

    from sklearn.metrics import roc_auc_score

    n = len(X)
    min_train_frac = 0.5
    fold_size = int(n * (1 - min_train_frac) / n_folds)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1000, step=100),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "auto_class_weights": "Balanced",
            "random_seed": 42,
            "verbose": 0,
            "eval_metric": "AUC",
        }

        aucs = []
        for fold in range(n_folds):
            train_end = int(n * min_train_frac) + fold * fold_size
            val_end = min(train_end + fold_size, n)
            if train_end >= n or val_end <= train_end:
                break

            X_train, X_val = X[:train_end], X[train_end:val_end]
            y_train, y_val = y[:train_end], y[train_end:val_end]
            w_train = sample_weights[:train_end]

            model = CatBoostClassifier(**params)
            train_pool = Pool(X_train, y_train, weight=w_train)
            val_pool = Pool(X_val, y_val)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=0)

            proba = model.predict_proba(X_val)[:, 1]
            try:
                auc = roc_auc_score(y_val, proba)
            except ValueError:
                auc = 0.5
            aucs.append(auc)

        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Optuna CatBoost best AUC: %.4f", study.best_value)
    logger.info("Optuna CatBoost best params: %s", study.best_params)
    return study.best_params
