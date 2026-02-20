"""Fast batch model training CLI.

Thin wrapper around src.ml.training and src.ml.dataset_fast.
Run: python -m scripts.train_models [horizons...] [--catboost] [--tune] [--folds N]
"""

from __future__ import annotations

import asyncio
import logging
import sys

import numpy as np
import pandas as pd

from src.db.postgres import async_session_factory
from src.ml.dataset_fast import build_dataset_fast, feature_columns
from src.ml.training import (
    ModelTrainer,
    optuna_tune_catboost,
    optuna_tune_lgbm,
    train_models_cv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def main():
    args = sys.argv[1:]

    # Parse flags
    use_catboost = "--catboost" in args
    use_tune = "--tune" in args
    n_trials = 50
    for a in args:
        if a.startswith("--trials="):
            n_trials = int(a.split("=")[1])
    args = [a for a in args if not a.startswith("--")]

    # Parse horizons
    horizons = [a for a in args if a in ("5d", "21d", "63d", "90d", "180d")]
    if not horizons:
        horizons = ["90d", "180d"]

    if use_catboost:
        logger.info("CatBoost enabled")
    if use_tune:
        logger.info("Optuna tuning enabled (%d trials)", n_trials)

    # Parse folds
    n_folds = 5
    for a in sys.argv[1:]:
        if a.startswith("--folds="):
            n_folds = int(a.split("=")[1])
        elif a == "--folds":
            idx = sys.argv.index(a)
            if idx + 1 < len(sys.argv):
                n_folds = int(sys.argv[idx + 1])

    all_results: dict[str, dict] = {}

    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"  HORIZON: {horizon}")
        print(f"{'='*60}")

        async with async_session_factory() as session:
            df = await build_dataset_fast(session, limit=20000, horizon=horizon)
            if df.empty:
                logger.error("No training data for horizon %s!", horizon)
                continue

            # Optuna tuning (run before main training)
            tuned_lgbm_params: dict = {}
            tuned_catboost_params: dict = {}
            if use_tune:
                feat_cols = feature_columns(df)
                X_raw = df[feat_cols].values.astype(float)
                X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
                variances = np.var(X_raw, axis=0)
                keep_mask = variances > 1e-10
                X_tune = X_raw[:, keep_mask]
                y_tune = df["profitable"].values.astype(float)
                w_tune = df["sample_weight"].values.astype(float)
                dates_tune = pd.to_datetime(df["transaction_date"]).values

                logger.info("Running Optuna LightGBM tuning (%d trials)...", n_trials)
                tuned_lgbm_params = optuna_tune_lgbm(
                    X_tune, y_tune, w_tune, dates=dates_tune,
                    n_folds=3, n_trials=n_trials,
                )

                if use_catboost:
                    logger.info("Running Optuna CatBoost tuning (%d trials)...", n_trials)
                    tuned_catboost_params = optuna_tune_catboost(
                        X_tune, y_tune, w_tune, dates=dates_tune,
                        n_folds=3, n_trials=n_trials,
                    )

            results = await train_models_cv(
                df, session, n_folds=n_folds, use_catboost=use_catboost,
                tuned_lgbm_params=tuned_lgbm_params,
                tuned_catboost_params=tuned_catboost_params,
            )
            all_results[horizon] = results

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"  HORIZON COMPARISON ({n_folds}-fold temporal CV)")
    print("=" * 70)
    print(f"  {'Model':<25s} {'Horizon':<10s} {'Accuracy':>10s} {'AUC':>10s} {'F1':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for horizon, results in all_results.items():
        for model_name, metrics in results.items():
            if "accuracy" in metrics:
                acc_std = metrics.get("accuracy_std", 0)
                print(
                    f"  {model_name:<25s} {horizon:<10s} "
                    f"{metrics.get('accuracy', 0):>7.4f}+-{acc_std:.3f} "
                    f"{metrics.get('auc', 0):>10.4f} "
                    f"{metrics.get('f1', 0):>10.4f}"
                )
            elif "mae" in metrics:
                print(
                    f"  {model_name:<25s} {horizon:<10s} "
                    f"{'MAE:':>10s} "
                    f"{metrics.get('mae', 0):>10.4f} "
                    f"{metrics.get('r2', 0):>10.4f}"
                )


if __name__ == "__main__":
    asyncio.run(main())
