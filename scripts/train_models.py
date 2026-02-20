"""Fast batch model training CLI.

Thin wrapper around src.ml.training and src.ml.dataset_fast.
Run: python -m scripts.train_models [horizons...] [--catboost] [--tune] [--folds N]
     [--autogluon] [--ft-transformer] [--ag-time-limit=N] [--embed-dim=N]
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


def _parse_int_flag(args: list[str], flag: str, default: int) -> int:
    """Parse --flag=N from command line args."""
    for a in args:
        if a.startswith(f"--{flag}="):
            return int(a.split("=")[1])
    return default


async def main():
    raw_args = sys.argv[1:]

    # Parse flags
    use_catboost = "--catboost" in raw_args
    use_tune = "--tune" in raw_args
    use_autogluon = "--autogluon" in raw_args
    use_ft_transformer = "--ft-transformer" in raw_args

    n_trials = _parse_int_flag(raw_args, "trials", 50)
    ag_time_limit = _parse_int_flag(raw_args, "ag-time-limit", 300)
    embed_dim = _parse_int_flag(raw_args, "embed-dim", 64)

    args = [a for a in raw_args if not a.startswith("--")]

    # Parse horizons
    horizons = [a for a in args if a in ("5d", "21d", "63d", "90d", "180d")]
    if not horizons:
        horizons = ["90d", "180d"]

    if use_catboost:
        logger.info("CatBoost enabled")
    if use_autogluon:
        logger.info("AutoGluon enabled (time_limit=%ds)", ag_time_limit)
    if use_ft_transformer:
        logger.info("FT-Transformer enabled (embed_dim=%d)", embed_dim)
    if use_tune:
        logger.info("Optuna tuning enabled (%d trials)", n_trials)

    # Parse folds
    n_folds = _parse_int_flag(raw_args, "folds", 5)

    # Extract GNN embeddings if FT-Transformer requested
    gnn_embeddings = None
    gnn_embed_cols: list[str] | None = None
    if use_ft_transformer:
        gnn_embeddings = await _prepare_gnn_embeddings(embed_dim)

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

            # Augment with GNN embeddings if available
            if gnn_embeddings:
                from src.ml.gnn_embeddings import build_trade_embedding_columns
                df, gnn_embed_cols = build_trade_embedding_columns(
                    df, gnn_embeddings, embed_dim
                )
                logger.info("Augmented dataset with %d GNN embedding columns", len(gnn_embed_cols))

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
                use_autogluon=use_autogluon,
                ag_time_limit=ag_time_limit,
                use_ft_transformer=use_ft_transformer,
                gnn_embed_cols=gnn_embed_cols,
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


async def _prepare_gnn_embeddings(embed_dim: int) -> dict | None:
    """Extract graph from Neo4j and train GNN embeddings."""
    from src.ml.gnn_embeddings import (
        extract_graph_from_neo4j,
        load_embeddings,
        save_embeddings,
        train_gnn,
    )

    # Try loading cached embeddings first
    cached = load_embeddings()
    if cached is not None:
        logger.info("Using cached GNN embeddings")
        return cached

    # Extract graph and train GNN
    try:
        from src.db.neo4j import get_neo4j_session

        logger.info("Extracting graph from Neo4j for GNN training...")
        async with get_neo4j_session() as neo4j_session:
            graph_data = await extract_graph_from_neo4j(neo4j_session)

        logger.info("Training GNN embeddings (embed_dim=%d)...", embed_dim)
        embeddings = train_gnn(graph_data, embed_dim=embed_dim)
        save_embeddings(embeddings)
        return embeddings
    except Exception:
        logger.exception("Failed to generate GNN embeddings — continuing without them")
        return None


if __name__ == "__main__":
    asyncio.run(main())
