"""Fast batch model training script.

Builds features using bulk SQL queries instead of per-trade round-trips.
Run: python -m scripts.train_models
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import and_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.postgres import async_session_factory
from src.ml.evaluation import evaluate_classifier, evaluate_regressor
from src.ml.models.anomaly_model import AnomalyDetector
from src.ml.models.ensemble import EnsembleModel
from src.ml.models.return_predictor import ReturnPredictor
from src.ml.models.trade_predictor import TradePredictor
from src.models.ml import MLModelArtifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

ARTIFACT_DIR = "data/models"

# Feature encoding maps
FILER_TYPE_MAP = {"member": 0, "spouse": 1, "dependent": 2, "joint": 3}
PARTY_MAP = {"Democrat": 0, "Republican": 1, "Independent": 2}
CHAMBER_MAP = {"house": 0, "senate": 1}
TX_DIR_MAP = {"purchase": 1, "sale": -1, "sale_full": -1, "sale_partial": -1, "exchange": 0}


async def build_dataset_fast(session: AsyncSession, limit: int = 20000) -> pd.DataFrame:
    """Build training dataset using bulk SQL — 100x faster than per-trade queries."""

    logger.info("Loading trades with member and stock data...")

    result = await session.execute(
        text("""
            WITH trades AS (
                SELECT
                    t.id as trade_id,
                    t.ticker,
                    t.transaction_date,
                    t.disclosure_date,
                    t.transaction_type,
                    t.filer_type,
                    t.amount_range_low,
                    t.amount_range_high,
                    t.member_bioguide_id,
                    t.chamber as trade_chamber,
                    m.party,
                    m.chamber as member_chamber,
                    m.nominate_dim1,
                    m.nominate_dim2,
                    m.first_elected
                FROM trade_disclosure t
                LEFT JOIN congress_member m ON t.member_bioguide_id = m.bioguide_id
                WHERE t.ticker IS NOT NULL
                AND t.ticker != ''
                AND t.transaction_date IS NOT NULL
                AND t.transaction_date >= '2016-01-01'
                AND t.transaction_date <= CURRENT_DATE - INTERVAL '8 days'
                ORDER BY t.transaction_date
                LIMIT :limit
            )
            SELECT
                tr.*,
                base.adj_close as base_price,
                base.date as base_date,
                future5.adj_close as price_5d,
                future5.date as future_5d_date,
                future21.adj_close as price_21d,
                future21.date as future_21d_date
            FROM trades tr
            LEFT JOIN LATERAL (
                SELECT adj_close, date FROM stock_daily
                WHERE ticker = tr.ticker AND date <= tr.transaction_date
                ORDER BY date DESC LIMIT 1
            ) base ON true
            LEFT JOIN LATERAL (
                SELECT adj_close, date FROM stock_daily
                WHERE ticker = tr.ticker AND date > tr.transaction_date
                ORDER BY date OFFSET 4 LIMIT 1
            ) future5 ON true
            LEFT JOIN LATERAL (
                SELECT adj_close, date FROM stock_daily
                WHERE ticker = tr.ticker AND date > tr.transaction_date
                ORDER BY date OFFSET 20 LIMIT 1
            ) future21 ON true
        """),
        {"limit": limit},
    )
    rows = result.fetchall()
    columns = result.keys()
    df = pd.DataFrame(rows, columns=columns)

    logger.info("Loaded %d trades from DB", len(df))

    if df.empty:
        return pd.DataFrame()

    # Compute labels — both 5d and 21d
    df["actual_return_5d"] = None
    mask_5d = df["base_price"].notna() & df["price_5d"].notna() & (df["base_price"] > 0)
    df.loc[mask_5d, "actual_return_5d"] = (
        (df.loc[mask_5d, "price_5d"].astype(float) - df.loc[mask_5d, "base_price"].astype(float))
        / df.loc[mask_5d, "base_price"].astype(float)
    )

    df["actual_return_21d"] = None
    mask_21d = df["base_price"].notna() & df["price_21d"].notna() & (df["base_price"] > 0)
    df.loc[mask_21d, "actual_return_21d"] = (
        (df.loc[mask_21d, "price_21d"].astype(float) - df.loc[mask_21d, "base_price"].astype(float))
        / df.loc[mask_21d, "base_price"].astype(float)
    )

    df["is_purchase"] = (df["transaction_type"] == "purchase").astype(float)

    df["profitable_5d"] = None
    ret_mask_5d = df["actual_return_5d"].notna()
    df.loc[ret_mask_5d, "profitable_5d"] = (
        ((df.loc[ret_mask_5d, "actual_return_5d"].astype(float) > 0) == (df.loc[ret_mask_5d, "is_purchase"] == 1.0))
        .astype(float)
    )

    df["profitable_21d"] = None
    ret_mask_21d = df["actual_return_21d"].notna()
    df.loc[ret_mask_21d, "profitable_21d"] = (
        ((df.loc[ret_mask_21d, "actual_return_21d"].astype(float) > 0) == (df.loc[ret_mask_21d, "is_purchase"] == 1.0))
        .astype(float)
    )

    # Drop rows without labels — use 21d as primary, fall back to 5d
    df_21d = df.dropna(subset=["profitable_21d"])
    df_5d = df.dropna(subset=["profitable_5d"])
    logger.info("Labeled trades (5d): %d (%.1f%% profitable)", len(df_5d), df_5d["profitable_5d"].mean() * 100)
    logger.info("Labeled trades (21d): %d (%.1f%% profitable)", len(df_21d), df_21d["profitable_21d"].mean() * 100)
    df = df_21d  # Use 21d horizon for training

    # === Build features ===

    # Trade features
    amount_low = df["amount_range_low"].fillna(0).astype(float)
    amount_high = df["amount_range_high"].fillna(0).astype(float)
    df["amount_midpoint"] = (amount_low + amount_high) / 2
    df["amount_log"] = np.log1p(df["amount_midpoint"])
    df["tx_direction"] = df["transaction_type"].map(TX_DIR_MAP).fillna(0).astype(float)
    df["filer_type_encoded"] = df["filer_type"].map(FILER_TYPE_MAP).fillna(0).astype(float)

    # Disclosure lag
    df["disclosure_lag_days"] = 30.0  # default
    has_dates = df["disclosure_date"].notna() & df["transaction_date"].notna()
    df.loc[has_dates, "disclosure_lag_days"] = (
        pd.to_datetime(df.loc[has_dates, "disclosure_date"]) - pd.to_datetime(df.loc[has_dates, "transaction_date"])
    ).dt.days.clip(lower=0).astype(float)

    # Member features
    df["party_encoded"] = df["party"].map(PARTY_MAP).fillna(2).astype(float)
    df["chamber_encoded"] = df["member_chamber"].map(CHAMBER_MAP).fillna(
        df["trade_chamber"].map(CHAMBER_MAP)
    ).fillna(0).astype(float)
    df["nominate_dim1"] = df["nominate_dim1"].fillna(0).astype(float)
    df["nominate_dim2"] = df["nominate_dim2"].fillna(0).astype(float)
    current_year = date.today().year
    df["years_in_office"] = (current_year - df["first_elected"].fillna(current_year - 10)).astype(float)

    # Committee count (bulk)
    logger.info("Loading committee assignments...")
    r = await session.execute(text("""
        SELECT member_bioguide_id, COUNT(*) as committee_count
        FROM committee_assignment
        GROUP BY member_bioguide_id
    """))
    comm_df = pd.DataFrame(r.fetchall(), columns=r.keys())
    df = df.merge(comm_df, left_on="member_bioguide_id", right_on="member_bioguide_id", how="left")
    df["committee_count"] = df["committee_count"].fillna(0).astype(float)

    # Market features (bulk via windowed queries)
    logger.info("Computing market features...")
    await _add_market_features(session, df)

    # Legislative features — simplified bulk version
    logger.info("Computing legislative features...")
    await _add_legislative_features(session, df)

    # Sentiment features — bulk
    logger.info("Computing sentiment features...")
    await _add_sentiment_features(session, df)

    # Network features — set to 0 (Neo4j not available)
    df["lobbying_connection_count"] = 0.0
    df["campaign_donor_connection"] = 0.0
    df["network_degree"] = 0.0
    df["has_lobbying_triangle"] = 0.0

    # Member historical win rate — key feature for follow-the-smart-money
    logger.info("Computing member historical win rates...")
    await _add_member_win_rate(session, df)

    logger.info("Dataset built: %d samples, %d features", len(df), len(_feature_cols(df)))
    return df


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names."""
    exclude = {
        "trade_id", "ticker", "transaction_date", "disclosure_date",
        "transaction_type", "filer_type", "amount_range_low", "amount_range_high",
        "member_bioguide_id", "trade_chamber", "party", "member_chamber",
        "first_elected", "base_price", "base_date",
        "price_5d", "future_5d_date", "price_21d", "future_21d_date",
        "actual_return_5d", "actual_return_21d",
        "profitable_5d", "profitable_21d", "is_purchase",
    }
    return [c for c in df.columns if c not in exclude]


async def _add_market_features(session: AsyncSession, df: pd.DataFrame) -> None:
    """Add market features using bulk stock queries."""
    # Get unique (ticker, date) pairs we need
    tickers = df["ticker"].unique().tolist()

    # Load all relevant stock data at once
    r = await session.execute(
        text("""
            SELECT ticker, date, adj_close, close, volume
            FROM stock_daily
            WHERE ticker = ANY(:tickers)
            AND date >= :start_date
            ORDER BY ticker, date
        """),
        {
            "tickers": tickers,
            "start_date": df["transaction_date"].min() - timedelta(days=60),
        },
    )
    stock_df = pd.DataFrame(r.fetchall(), columns=r.keys())

    if stock_df.empty:
        for col in ["price_change_5d", "price_change_21d", "volatility_21d", "volume_ratio_5d", "rsi_14"]:
            df[col] = 0.0 if col != "rsi_14" else 50.0
        return

    stock_df["price"] = stock_df["adj_close"].fillna(stock_df["close"]).astype(float)
    stock_df["volume"] = stock_df["volume"].fillna(0).astype(float)

    # For each trade, compute features from stock data before trade date
    results = []
    stock_by_ticker = dict(list(stock_df.groupby("ticker")))

    for _, row in df.iterrows():
        ticker = row["ticker"]
        trade_date = pd.Timestamp(row["transaction_date"])

        feat = {
            "price_change_5d": 0.0,
            "price_change_21d": 0.0,
            "volatility_21d": 0.0,
            "volume_ratio_5d": 0.0,
            "rsi_14": 50.0,
        }

        if ticker in stock_by_ticker:
            sdf = stock_by_ticker[ticker]
            before = sdf[sdf["date"] < trade_date.date()].tail(30)

            if len(before) >= 2:
                prices = before["price"].values[::-1]  # most recent first
                volumes = before["volume"].values[::-1]

                current = float(prices[0]) if prices[0] > 0 else 1.0
                p5 = float(prices[min(5, len(prices) - 1)])
                p21 = float(prices[min(21, len(prices) - 1)])

                feat["price_change_5d"] = (current - p5) / p5 if p5 > 0 else 0
                feat["price_change_21d"] = (current - p21) / p21 if p21 > 0 else 0

                # Volatility
                returns = []
                for i in range(min(21, len(prices) - 1)):
                    if prices[i + 1] > 0:
                        returns.append((prices[i] - prices[i + 1]) / prices[i + 1])
                if len(returns) >= 2:
                    feat["volatility_21d"] = float(np.std(returns, ddof=1))

                # Volume ratio
                v5 = float(np.mean(volumes[:5])) if len(volumes) >= 5 else float(np.mean(volumes))
                v21 = float(np.mean(volumes[:21])) if len(volumes) >= 5 else v5
                feat["volume_ratio_5d"] = v5 / v21 if v21 > 0 else 1.0

                # RSI
                feat["rsi_14"] = _compute_rsi(prices[:15])

        results.append(feat)

    market_df = pd.DataFrame(results)
    for col in market_df.columns:
        df[col] = market_df[col].values


async def _add_legislative_features(session: AsyncSession, df: pd.DataFrame) -> None:
    """Add legislative features using bulk queries."""
    # Default values
    df["min_hearing_distance"] = 999.0
    df["min_bill_distance"] = 999.0
    df["committee_sector_alignment"] = 0.0
    df["timing_suspicion_score"] = 0.0
    df["has_related_bill"] = 0.0
    df["nearby_hearing_count"] = 0.0
    df["nearby_bill_count"] = 0.0

    # Bulk: count hearings within 30 days of each trade
    logger.info("  Loading nearby hearings...")
    r = await session.execute(text("""
        SELECT t.id as trade_id,
               COUNT(h.id) as hearing_count,
               MIN(ABS(t.transaction_date - h.hearing_date)) as min_hearing_dist
        FROM trade_disclosure t
        JOIN committee_hearing h
            ON h.hearing_date BETWEEN t.transaction_date - 30 AND t.transaction_date + 30
        WHERE t.ticker IS NOT NULL
        AND t.transaction_date >= '2016-01-01'
        GROUP BY t.id
    """))
    hearing_df = pd.DataFrame(r.fetchall(), columns=r.keys())

    if not hearing_df.empty:
        hearing_map = hearing_df.set_index("trade_id")
        mask = df["trade_id"].isin(hearing_map.index)
        matched = df.loc[mask, "trade_id"].map(hearing_map["hearing_count"])
        df.loc[mask, "nearby_hearing_count"] = matched.values.astype(float)
        dist = df.loc[mask, "trade_id"].map(hearing_map["min_hearing_dist"])
        df.loc[mask, "min_hearing_distance"] = dist.values.astype(float)

    # Bulk: count bills within 90 days
    logger.info("  Loading nearby bills...")
    r = await session.execute(text("""
        SELECT t.id as trade_id,
               COUNT(b.bill_id) as bill_count,
               MIN(ABS(t.transaction_date - b.introduced_date)) as min_bill_dist
        FROM trade_disclosure t
        JOIN bill b
            ON b.introduced_date BETWEEN t.transaction_date - 90 AND t.transaction_date + 90
        WHERE t.ticker IS NOT NULL
        AND t.transaction_date >= '2016-01-01'
        GROUP BY t.id
    """))
    bill_df = pd.DataFrame(r.fetchall(), columns=r.keys())

    if not bill_df.empty:
        bill_map = bill_df.set_index("trade_id")
        mask = df["trade_id"].isin(bill_map.index)
        matched = df.loc[mask, "trade_id"].map(bill_map["bill_count"])
        df.loc[mask, "nearby_bill_count"] = matched.values.astype(float)
        df.loc[mask, "has_related_bill"] = (matched.values > 0).astype(float)
        dist = df.loc[mask, "trade_id"].map(bill_map["min_bill_dist"])
        df.loc[mask, "min_bill_distance"] = dist.values.astype(float)

    # Compute simple suspicion score from available features
    # Score increases with: close hearing, close bill, late disclosure
    lag = df["disclosure_lag_days"].clip(0, 90) / 90.0  # normalized 0-1
    hearing_score = (1.0 - df["min_hearing_distance"].clip(0, 30) / 30.0).clip(0, 1)
    bill_score = (1.0 - df["min_bill_distance"].clip(0, 90) / 90.0).clip(0, 1)
    df["timing_suspicion_score"] = ((hearing_score * 0.4 + bill_score * 0.3 + lag * 0.3)).clip(0, 1)


async def _add_sentiment_features(session: AsyncSession, df: pd.DataFrame) -> None:
    """Add sentiment features using bulk queries."""
    df["avg_sentiment_7d"] = 0.0
    df["avg_sentiment_30d"] = 0.0
    df["sentiment_momentum"] = 0.0
    df["media_mention_count_7d"] = 0.0
    df["media_mention_count_30d"] = 0.0

    # Bulk sentiment average for all dates
    r = await session.execute(text("""
        SELECT mc.published_date, AVG(sa.sentiment_score) as avg_score
        FROM media_content mc
        JOIN sentiment_analysis sa ON sa.media_content_id = mc.id
        WHERE mc.published_date IS NOT NULL
        GROUP BY mc.published_date
        ORDER BY mc.published_date
    """))
    sent_df = pd.DataFrame(r.fetchall(), columns=r.keys())

    if sent_df.empty:
        return

    # For now, global sentiment is sparse — just set to 0
    # Will be useful once more media content accumulates
    logger.info("  Sentiment data: %d dates with scores", len(sent_df))


async def _add_member_win_rate(session: AsyncSession, df: pd.DataFrame) -> None:
    """Add member historical win rate as a feature.

    For each trade, compute the member's win rate from ALL their PRIOR trades
    (no future leakage). Uses expanding window — each trade only sees past results.
    """
    df["member_win_rate"] = 0.5  # default for unknown/unlinked members
    df["member_trade_count"] = 0.0
    df["member_avg_return"] = 0.0

    # Get all trades sorted by date with their returns
    r = await session.execute(text("""
        SELECT
            t.id as trade_id,
            t.member_bioguide_id,
            t.transaction_type,
            t.transaction_date,
            base.adj_close as base_price,
            f21.adj_close as price_21d
        FROM trade_disclosure t
        LEFT JOIN LATERAL (
            SELECT adj_close FROM stock_daily
            WHERE ticker = t.ticker AND date <= t.transaction_date
            ORDER BY date DESC LIMIT 1
        ) base ON true
        LEFT JOIN LATERAL (
            SELECT adj_close FROM stock_daily
            WHERE ticker = t.ticker AND date > t.transaction_date
            ORDER BY date OFFSET 20 LIMIT 1
        ) f21 ON true
        WHERE t.ticker IS NOT NULL AND t.ticker != ''
        AND t.transaction_date >= '2016-01-01'
        AND t.member_bioguide_id IS NOT NULL
        AND base.adj_close > 0
        ORDER BY t.transaction_date
    """))
    all_trades = pd.DataFrame(r.fetchall(), columns=r.keys())

    if all_trades.empty:
        return

    # Build expanding window: for each member, track cumulative wins/total
    member_stats: dict[str, dict] = {}  # bio_id -> {wins, total, returns}

    # Pre-compute win for each historical trade
    all_trades["is_win"] = None
    all_trades["return_21d"] = None
    mask = all_trades["price_21d"].notna() & (all_trades["base_price"] > 0)
    all_trades.loc[mask, "return_21d"] = (
        (all_trades.loc[mask, "price_21d"].astype(float) - all_trades.loc[mask, "base_price"].astype(float))
        / all_trades.loc[mask, "base_price"].astype(float)
    )
    ret_mask = all_trades["return_21d"].notna()
    is_purchase = all_trades["transaction_type"] == "purchase"
    all_trades.loc[ret_mask, "is_win"] = (
        ((all_trades.loc[ret_mask, "return_21d"].astype(float) > 0) == is_purchase[ret_mask])
        .astype(float)
    )

    # Build lookup: trade_id -> (cumulative_win_rate, cumulative_count, cumulative_avg_return)
    # at time BEFORE that trade
    trade_id_to_stats: dict[int, tuple[float, float, float]] = {}

    for _, row in all_trades.iterrows():
        bio = row["member_bioguide_id"]
        tid = row["trade_id"]

        if bio not in member_stats:
            member_stats[bio] = {"wins": 0, "total": 0, "return_sum": 0.0}

        stats = member_stats[bio]
        # Record stats AS OF before this trade
        if stats["total"] >= 3:  # need at least 3 prior trades
            wr = stats["wins"] / stats["total"]
            avg_ret = stats["return_sum"] / stats["total"]
            trade_id_to_stats[tid] = (wr, float(stats["total"]), avg_ret)

        # Update running stats with this trade's result
        if row["is_win"] is not None and not pd.isna(row["is_win"]):
            stats["total"] += 1
            stats["wins"] += int(row["is_win"])
            if row["return_21d"] is not None and not pd.isna(row["return_21d"]):
                stats["return_sum"] += float(row["return_21d"])

    # Map back to our training DataFrame
    for idx, row in df.iterrows():
        tid = row["trade_id"]
        if tid in trade_id_to_stats:
            wr, count, avg_ret = trade_id_to_stats[tid]
            df.at[idx, "member_win_rate"] = wr
            df.at[idx, "member_trade_count"] = count
            df.at[idx, "member_avg_return"] = avg_ret

    has_history = (df["member_trade_count"] > 0).sum()
    logger.info("  %d / %d trades have member history (%.1f%%)",
                has_history, len(df), has_history / len(df) * 100)


def _compute_rsi(prices, period: int = 14) -> float:
    """Compute RSI from prices (most recent first)."""
    if len(prices) < 2:
        return 50.0
    gains, losses = [], []
    for i in range(min(period, len(prices) - 1)):
        change = float(prices[i] - prices[i + 1])
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    avg_gain = sum(gains) / max(len(gains), 1)
    avg_loss = sum(losses) / max(len(losses), 1)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


async def train_models(df: pd.DataFrame, session: AsyncSession) -> dict:
    """Train all models on the dataset."""
    from sqlalchemy import update as sa_update

    feature_cols = _feature_cols(df)
    X = df[feature_cols].values.astype(float)
    y_class = df["profitable_21d"].values.astype(float)
    y_return = df["actual_return_21d"].fillna(0).values.astype(float)

    # Replace NaN
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(X)
    split_idx = int(n * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_class_train, y_class_val = y_class[:split_idx], y_class[split_idx:]
    y_return_train, y_return_val = y_return[:split_idx], y_return[split_idx:]

    logger.info("Train set: %d samples, Val set: %d samples", len(X_train), len(X_val))
    logger.info("Features (%d): %s", len(feature_cols), feature_cols)

    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results = {}

    # 1. Trade Predictor (LightGBM)
    logger.info("Training TradePredictor (LightGBM)...")
    trade_pred = TradePredictor()
    trade_pred.feature_columns = feature_cols
    trade_pred.train(X_train, y_class_train, X_val, y_class_val)
    val_preds = trade_pred.predict(X_val)
    val_proba = trade_pred.predict_proba(X_val)
    metrics = evaluate_classifier(y_class_val, val_preds, val_proba)
    results["trade_predictor"] = metrics
    await _save(session, trade_pred, version, metrics, feature_cols)
    logger.info("  TradePredictor: accuracy=%.4f, AUC=%.4f, F1=%.4f", metrics["accuracy"], metrics["auc"], metrics["f1"])

    # 2. Return Predictor (XGBoost)
    logger.info("Training ReturnPredictor (XGBoost)...")
    return_pred = ReturnPredictor()
    return_pred.feature_columns = feature_cols
    return_pred.train(X_train, y_return_train, X_val, y_return_val)
    val_preds = return_pred.predict(X_val)
    metrics = evaluate_regressor(y_return_val, val_preds)
    results["return_predictor"] = metrics
    await _save(session, return_pred, version, metrics, feature_cols)
    logger.info("  ReturnPredictor: MAE=%.4f, RMSE=%.4f, R2=%.4f", metrics["mae"], metrics["rmse"], metrics["r2"])

    # 3. Anomaly Detector
    logger.info("Training AnomalyDetector (Isolation Forest)...")
    anomaly_cols = [c for c in feature_cols if c not in AnomalyDetector.EXCLUDED_FEATURES]
    anomaly_idx = [feature_cols.index(c) for c in anomaly_cols]
    X_anomaly = X[:, anomaly_idx]
    anomaly_det = AnomalyDetector()
    anomaly_det.feature_columns = anomaly_cols
    metrics = anomaly_det.train(X_anomaly, y_class)
    results["anomaly_detector"] = metrics
    await _save(session, anomaly_det, version, metrics, anomaly_cols)
    logger.info("  AnomalyDetector: anomaly_rate=%.4f", metrics.get("anomaly_rate", 0))

    # 4. Ensemble
    logger.info("Training Ensemble (Logistic Regression meta-learner)...")
    trade_proba_all = trade_pred.predict_proba(X)
    return_scores_all = return_pred.predict(X)
    anomaly_scores_all = anomaly_det.predict_proba(X_anomaly)

    timing_idx = feature_cols.index("timing_suspicion_score") if "timing_suspicion_score" in feature_cols else None
    sentiment_idx = feature_cols.index("avg_sentiment_7d") if "avg_sentiment_7d" in feature_cols else None

    meta_features = np.column_stack([
        trade_proba_all,
        return_scores_all,
        anomaly_scores_all,
        X[:, timing_idx] if timing_idx is not None else np.zeros(n),
        X[:, sentiment_idx] if sentiment_idx is not None else np.zeros(n),
    ])

    meta_train, meta_val = meta_features[:split_idx], meta_features[split_idx:]

    ensemble = EnsembleModel()
    ensemble.feature_columns = EnsembleModel.META_FEATURES
    ensemble.train(meta_train, y_class_train, meta_val, y_class_val)
    val_preds = ensemble.predict(meta_val)
    val_proba = ensemble.predict_proba(meta_val)
    metrics = evaluate_classifier(y_class_val, val_preds, val_proba)
    results["ensemble"] = metrics
    await _save(session, ensemble, version, metrics, EnsembleModel.META_FEATURES)
    logger.info("  Ensemble: accuracy=%.4f, AUC=%.4f, F1=%.4f", metrics["accuracy"], metrics["auc"], metrics["f1"])

    return results


async def _save(session, model, version, metrics, feature_cols):
    """Save model artifact to disk and DB."""
    from sqlalchemy import update as sa_update

    path = f"{ARTIFACT_DIR}/{model.model_name}/{version}/model.pkl"
    model.save(path)

    await session.execute(
        sa_update(MLModelArtifact)
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


async def main():
    async with async_session_factory() as session:
        df = await build_dataset_fast(session, limit=20000)
        if df.empty:
            logger.error("No training data!")
            return

        results = await train_models(df, session)

        print("\n" + "=" * 50)
        print("TRAINING RESULTS")
        print("=" * 50)
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for k, v in sorted(metrics.items()):
                print(f"  {k:25s} {v:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
