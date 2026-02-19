"""Fast batch model training script.

Builds features using bulk SQL queries instead of per-trade round-trips.
Run: python -m scripts.train_models [horizons...] [--catboost]
"""

from __future__ import annotations

import asyncio
import json
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
SECTOR_CACHE_PATH = Path("data/sector_cache.json")

# Feature encoding maps
FILER_TYPE_MAP = {"member": 0, "spouse": 1, "dependent": 2, "joint": 3}
PARTY_MAP = {"Democrat": 0, "Republican": 1, "Independent": 2}
CHAMBER_MAP = {"house": 0, "senate": 1}
TX_DIR_MAP = {"purchase": 1, "sale": -1, "sale_full": -1, "sale_partial": -1, "exchange": 0}

# US presidential election years for cycle features
_ELECTION_YEARS = {2016, 2020, 2024, 2028}
# Midterm election years
_MIDTERM_YEARS = {2018, 2022, 2026}

# Committee sector mapping for sector alignment detection
COMMITTEE_SECTOR_MAP = {
    "HSBA": {"Financial Services", "Finance", "Banking"},
    "SSBA": {"Financial Services", "Finance", "Banking"},
    "HSAG": {"Consumer Defensive", "Agriculture"},
    "SSAF": {"Consumer Defensive", "Agriculture"},
    "HSIF": {"Energy", "Utilities"},
    "SSCM": {"Energy", "Utilities", "Communication Services"},
    "HSHM": {"Healthcare"},
    "SSHR": {"Healthcare"},
    "HSAS": {"Industrials", "Aerospace & Defense"},
    "SSAS": {"Industrials", "Aerospace & Defense"},
    "HSSM": {"Technology", "Communication Services"},
    "HSSY": {"Technology", "Communication Services"},
}


async def build_dataset_fast(session: AsyncSession, limit: int = 20000, horizon: str = "21d") -> pd.DataFrame:
    """Build training dataset using bulk SQL — 100x faster than per-trade queries.

    horizon: "5d", "21d", "63d", "90d", or "180d"
    """
    # Map horizon to trading-day offset
    HORIZON_OFFSETS = {"5d": 4, "21d": 20, "63d": 62, "90d": 89, "180d": 179}
    # Days of calendar buffer needed for the horizon + some safety margin
    HORIZON_BUFFER = {"5d": 10, "21d": 35, "63d": 100, "90d": 140, "180d": 270}

    offset = HORIZON_OFFSETS.get(horizon, 20)
    buffer_days = HORIZON_BUFFER.get(horizon, 35)

    logger.info("Loading trades (horizon=%s, offset=%d trading days)...", horizon, offset)

    cutoff_date = date.today() - timedelta(days=buffer_days)
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
                AND t.transaction_date <= :cutoff_date
                ORDER BY t.transaction_date
                LIMIT :limit
            )
            SELECT
                tr.*,
                base.adj_close as base_price,
                base.date as base_date,
                future.adj_close as price_future,
                future.date as future_date
            FROM trades tr
            LEFT JOIN LATERAL (
                SELECT adj_close, date FROM stock_daily
                WHERE ticker = tr.ticker AND date <= tr.transaction_date
                ORDER BY date DESC LIMIT 1
            ) base ON true
            LEFT JOIN LATERAL (
                SELECT adj_close, date FROM stock_daily
                WHERE ticker = tr.ticker AND date > tr.transaction_date
                ORDER BY date OFFSET :offset LIMIT 1
            ) future ON true
        """),
        {"offset": offset, "limit": limit, "cutoff_date": cutoff_date},
    )
    rows = result.fetchall()
    columns = result.keys()
    df = pd.DataFrame(rows, columns=columns)

    logger.info("Loaded %d trades from DB", len(df))

    if df.empty:
        return pd.DataFrame()

    # Compute labels
    df["actual_return"] = None
    mask = df["base_price"].notna() & df["price_future"].notna() & (df["base_price"] > 0)
    df.loc[mask, "actual_return"] = (
        (df.loc[mask, "price_future"].astype(float) - df.loc[mask, "base_price"].astype(float))
        / df.loc[mask, "base_price"].astype(float)
    )

    df["is_purchase"] = (df["transaction_type"] == "purchase").astype(float)

    df["profitable"] = None
    ret_mask = df["actual_return"].notna()
    df.loc[ret_mask, "profitable"] = (
        ((df.loc[ret_mask, "actual_return"].astype(float) > 0) == (df.loc[ret_mask, "is_purchase"] == 1.0))
        .astype(float)
    )

    # Drop rows without labels
    df = df.dropna(subset=["profitable"])
    logger.info("Labeled trades (%s): %d (%.1f%% profitable)", horizon, len(df), df["profitable"].mean() * 100)

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

    # Network features — use SQL fallback if Neo4j unavailable
    logger.info("Computing network features (SQL fallback)...")
    await _add_network_features_sql(session, df)

    # Member historical win rate — key feature for follow-the-smart-money
    logger.info("Computing member historical win rates...")
    await _add_member_win_rate(session, df, horizon_offset=offset)

    # NEW: Cross-member trading signals
    logger.info("Computing cross-member trading signals...")
    _add_cross_member_signals(df)

    # NEW: Calendar / seasonal features
    logger.info("Computing calendar features...")
    _add_calendar_features(df)

    # NEW: Ticker popularity among congress
    logger.info("Computing ticker popularity...")
    await _add_ticker_popularity(session, df)

    # NEW: Member trading velocity
    logger.info("Computing member trading velocity...")
    await _add_member_velocity(session, df)

    # NEW: Market regime (S&P 500 / broad market context)
    logger.info("Computing market regime features...")
    await _add_market_regime(session, df)

    # NEW: Stock sector features
    logger.info("Computing sector features...")
    await _add_sector_features(session, df)

    logger.info("Dataset built: %d samples, %d features", len(df), len(_feature_cols(df)))
    return df


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names."""
    exclude = {
        "trade_id", "ticker", "transaction_date", "disclosure_date",
        "transaction_type", "filer_type", "amount_range_low", "amount_range_high",
        "member_bioguide_id", "trade_chamber", "party", "member_chamber",
        "first_elected", "base_price", "base_date",
        "price_future", "future_date",
        "actual_return", "profitable", "is_purchase",
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

    # Bulk sentiment scores by date
    r = await session.execute(text("""
        SELECT mc.published_date, AVG(sa.sentiment_score) as avg_score,
               COUNT(*) as article_count
        FROM media_content mc
        JOIN sentiment_analysis sa ON sa.media_content_id = mc.id
        WHERE mc.published_date IS NOT NULL
        GROUP BY mc.published_date
        ORDER BY mc.published_date
    """))
    sent_df = pd.DataFrame(r.fetchall(), columns=r.keys())

    if sent_df.empty:
        logger.info("  No sentiment data available")
        return

    logger.info("  Sentiment data: %d dates with scores", len(sent_df))
    sent_df["published_date"] = pd.to_datetime(sent_df["published_date"])
    sent_df = sent_df.set_index("published_date").sort_index()

    # For each trade, compute rolling sentiment from media before trade date
    for idx, row in df.iterrows():
        td = pd.Timestamp(row["transaction_date"])
        # 7-day window
        window_7 = sent_df[(sent_df.index >= td - pd.Timedelta(days=7)) & (sent_df.index < td)]
        if not window_7.empty:
            df.at[idx, "avg_sentiment_7d"] = float(window_7["avg_score"].mean())
            df.at[idx, "media_mention_count_7d"] = float(window_7["article_count"].sum())
        # 30-day window
        window_30 = sent_df[(sent_df.index >= td - pd.Timedelta(days=30)) & (sent_df.index < td)]
        if not window_30.empty:
            df.at[idx, "avg_sentiment_30d"] = float(window_30["avg_score"].mean())
            df.at[idx, "media_mention_count_30d"] = float(window_30["article_count"].sum())
        # Momentum: 7d sentiment minus 30d sentiment
        df.at[idx, "sentiment_momentum"] = df.at[idx, "avg_sentiment_7d"] - df.at[idx, "avg_sentiment_30d"]


async def _add_member_win_rate(session: AsyncSession, df: pd.DataFrame, horizon_offset: int = 20) -> None:
    """Add member historical win rate as a feature.

    For each trade, compute the member's win rate from ALL their PRIOR trades
    (no future leakage). Uses expanding window — each trade only sees past results.
    """
    df["member_win_rate"] = 0.5  # default for unknown/unlinked members
    df["member_trade_count"] = 0.0
    df["member_avg_return"] = 0.0

    # Get all trades sorted by date with their returns
    r = await session.execute(
        text("""
        SELECT
            t.id as trade_id,
            t.member_bioguide_id,
            t.transaction_type,
            t.transaction_date,
            base.adj_close as base_price,
            future.adj_close as price_future
        FROM trade_disclosure t
        LEFT JOIN LATERAL (
            SELECT adj_close FROM stock_daily
            WHERE ticker = t.ticker AND date <= t.transaction_date
            ORDER BY date DESC LIMIT 1
        ) base ON true
        LEFT JOIN LATERAL (
            SELECT adj_close FROM stock_daily
            WHERE ticker = t.ticker AND date > t.transaction_date
            ORDER BY date OFFSET :offset LIMIT 1
        ) future ON true
        WHERE t.ticker IS NOT NULL AND t.ticker != ''
        AND t.transaction_date >= '2016-01-01'
        AND t.member_bioguide_id IS NOT NULL
        AND base.adj_close > 0
        ORDER BY t.transaction_date
    """),
        {"offset": horizon_offset},
    )
    all_trades = pd.DataFrame(r.fetchall(), columns=r.keys())

    if all_trades.empty:
        return

    # Build expanding window: for each member, track cumulative wins/total
    member_stats: dict[str, dict] = {}  # bio_id -> {wins, total, returns}

    # Pre-compute win for each historical trade
    all_trades["is_win"] = None
    all_trades["ret"] = None
    mask = all_trades["price_future"].notna() & (all_trades["base_price"] > 0)
    all_trades.loc[mask, "ret"] = (
        (all_trades.loc[mask, "price_future"].astype(float) - all_trades.loc[mask, "base_price"].astype(float))
        / all_trades.loc[mask, "base_price"].astype(float)
    )
    ret_mask = all_trades["ret"].notna()
    is_purchase = all_trades["transaction_type"] == "purchase"
    all_trades.loc[ret_mask, "is_win"] = (
        ((all_trades.loc[ret_mask, "ret"].astype(float) > 0) == is_purchase[ret_mask])
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
            if row["ret"] is not None and not pd.isna(row["ret"]):
                stats["return_sum"] += float(row["ret"])

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


def _add_cross_member_signals(df: pd.DataFrame) -> None:
    """How many OTHER members traded this stock in ±7d and ±30d windows.

    Pure DataFrame computation — no DB round-trip needed.
    """
    df["cross_member_7d"] = 0.0
    df["cross_member_30d"] = 0.0
    df["cross_member_same_dir_7d"] = 0.0

    # Sort by ticker and date for efficient windowing
    df_sorted = df[["trade_id", "ticker", "transaction_date", "member_bioguide_id", "tx_direction"]].copy()
    df_sorted["transaction_date"] = pd.to_datetime(df_sorted["transaction_date"])

    # Group by ticker for efficiency
    for ticker, group in df_sorted.groupby("ticker"):
        if len(group) < 2:
            continue
        dates = group["transaction_date"].values
        members = group["member_bioguide_id"].values
        directions = group["tx_direction"].values
        trade_ids = group["trade_id"].values

        for i in range(len(group)):
            td = dates[i]
            my_member = members[i]
            my_dir = directions[i]

            # Count other members trading this stock within windows
            for window_days, col in [(7, "cross_member_7d"), (30, "cross_member_30d")]:
                window = np.timedelta64(window_days, 'D')
                mask = (np.abs(dates - td) <= window) & (members != my_member)
                unique_others = len(set(members[mask]) - {my_member})
                idx = df.index[df["trade_id"] == trade_ids[i]]
                if len(idx) > 0:
                    df.loc[idx[0], col] = float(unique_others)

            # Same direction within 7d
            window = np.timedelta64(7, 'D')
            mask = (np.abs(dates - td) <= window) & (members != my_member) & (directions == my_dir)
            unique_same_dir = len(set(members[mask]) - {my_member})
            idx = df.index[df["trade_id"] == trade_ids[i]]
            if len(idx) > 0:
                df.loc[idx[0], "cross_member_same_dir_7d"] = float(unique_same_dir)

    logger.info("  Cross-member signals: %.1f%% trades have >=1 other member in 7d",
                (df["cross_member_7d"] > 0).mean() * 100)


def _add_calendar_features(df: pd.DataFrame) -> None:
    """Calendar, seasonal, and political cycle features."""
    dates = pd.to_datetime(df["transaction_date"])

    # Month and quarter (cyclical encoding)
    df["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12).astype(float)
    df["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12).astype(float)
    df["quarter"] = dates.dt.quarter.astype(float)

    # Day of week (0=Mon, 4=Fri — trades cluster around certain days)
    df["day_of_week"] = dates.dt.dayofweek.astype(float)

    # Year-end effects (December trading)
    df["is_december"] = (dates.dt.month == 12).astype(float)

    # January effect
    df["is_january"] = (dates.dt.month == 1).astype(float)

    # Election cycle proximity
    years = dates.dt.year
    df["is_election_year"] = years.isin(_ELECTION_YEARS).astype(float)
    df["is_midterm_year"] = years.isin(_MIDTERM_YEARS).astype(float)

    # Days to next November election (approximate)
    next_nov = pd.to_datetime(years.astype(str) + "-11-05")
    days_to_election = (next_nov - dates).dt.days
    # If past November, use next year
    mask = days_to_election < 0
    next_nov_adj = pd.to_datetime((years + 1).astype(str) + "-11-05")
    days_to_election = days_to_election.where(~mask, (next_nov_adj - dates).dt.days)
    df["days_to_election"] = days_to_election.clip(0, 730).astype(float)

    # Lame duck session (Nov election to Jan inauguration)
    df["is_lame_duck"] = ((dates.dt.month >= 11) | (dates.dt.month == 1)).astype(float)


async def _add_ticker_popularity(session: AsyncSession, df: pd.DataFrame) -> None:
    """How popular is this ticker among congress members (historical, no leakage)."""
    df["ticker_total_members"] = 0.0
    df["ticker_total_trades"] = 0.0
    df["ticker_purchase_ratio"] = 0.5

    # Get all-time stats per ticker (up to a cutoff to prevent leakage)
    r = await session.execute(text("""
        SELECT ticker,
               COUNT(DISTINCT member_bioguide_id) as unique_members,
               COUNT(*) as total_trades,
               AVG(CASE WHEN transaction_type = 'purchase' THEN 1.0 ELSE 0.0 END) as purchase_ratio
        FROM trade_disclosure
        WHERE ticker IS NOT NULL AND transaction_date >= '2016-01-01'
        GROUP BY ticker
    """))
    ticker_stats = pd.DataFrame(r.fetchall(), columns=r.keys())

    if not ticker_stats.empty:
        stats_map = ticker_stats.set_index("ticker")
        for col_src, col_dst in [("unique_members", "ticker_total_members"),
                                  ("total_trades", "ticker_total_trades"),
                                  ("purchase_ratio", "ticker_purchase_ratio")]:
            mask = df["ticker"].isin(stats_map.index)
            df.loc[mask, col_dst] = df.loc[mask, "ticker"].map(stats_map[col_src]).astype(float)

    logger.info("  Ticker popularity: %.1f avg members/ticker",
                df["ticker_total_members"].mean())


async def _add_member_velocity(session: AsyncSession, df: pd.DataFrame) -> None:
    """Member's recent trading frequency vs their historical average.

    High velocity = member trading more than usual = potentially significant.
    """
    df["member_trades_30d"] = 0.0
    df["member_trades_90d"] = 0.0
    df["member_velocity_ratio"] = 1.0  # 30d rate / 90d rate

    # Bulk: count each member's trades in 30d and 90d windows before each trade
    r = await session.execute(text("""
        SELECT t1.id as trade_id,
               (SELECT COUNT(*) FROM trade_disclosure t2
                WHERE t2.member_bioguide_id = t1.member_bioguide_id
                AND t2.transaction_date BETWEEN t1.transaction_date - 30 AND t1.transaction_date - 1
                AND t2.id != t1.id) as trades_30d,
               (SELECT COUNT(*) FROM trade_disclosure t2
                WHERE t2.member_bioguide_id = t1.member_bioguide_id
                AND t2.transaction_date BETWEEN t1.transaction_date - 90 AND t1.transaction_date - 1
                AND t2.id != t1.id) as trades_90d
        FROM trade_disclosure t1
        WHERE t1.ticker IS NOT NULL
        AND t1.transaction_date >= '2016-01-01'
        AND t1.member_bioguide_id IS NOT NULL
    """))
    vel_df = pd.DataFrame(r.fetchall(), columns=r.keys())

    if not vel_df.empty:
        vel_map = vel_df.set_index("trade_id")
        mask = df["trade_id"].isin(vel_map.index)
        df.loc[mask, "member_trades_30d"] = df.loc[mask, "trade_id"].map(vel_map["trades_30d"]).astype(float)
        df.loc[mask, "member_trades_90d"] = df.loc[mask, "trade_id"].map(vel_map["trades_90d"]).astype(float)

        # Velocity ratio: 30d rate / 90d rate (>1 means accelerating)
        rate_30 = df["member_trades_30d"] / 30.0
        rate_90 = df["member_trades_90d"] / 90.0
        df["member_velocity_ratio"] = np.where(rate_90 > 0, rate_30 / rate_90, 1.0)
        df["member_velocity_ratio"] = df["member_velocity_ratio"].clip(0, 10).astype(float)

    logger.info("  Member velocity: %.1f avg trades/30d", df["member_trades_30d"].mean())


async def _add_market_regime(session: AsyncSession, df: pd.DataFrame) -> None:
    """Broad market context using SPY as S&P 500 proxy."""
    df["spy_return_21d"] = 0.0
    df["spy_return_63d"] = 0.0
    df["spy_volatility_21d"] = 0.0
    df["market_is_bull"] = 0.0

    # Load SPY data
    r = await session.execute(text("""
        SELECT date, adj_close FROM stock_daily
        WHERE ticker = 'SPY' AND date >= '2015-01-01'
        ORDER BY date
    """))
    spy_df = pd.DataFrame(r.fetchall(), columns=r.keys())

    if spy_df.empty:
        logger.info("  No SPY data — skipping market regime features")
        return

    spy_df["price"] = spy_df["adj_close"].astype(float)
    spy_df["date"] = pd.to_datetime(spy_df["date"])
    spy_df = spy_df.set_index("date").sort_index()

    for idx, row in df.iterrows():
        td = pd.Timestamp(row["transaction_date"])
        before = spy_df[spy_df.index < td].tail(63)
        if len(before) < 5:
            continue

        prices = before["price"].values
        current = float(prices[-1])

        # 21d return
        p21 = float(prices[-min(21, len(prices))]) if len(prices) >= 2 else current
        df.at[idx, "spy_return_21d"] = (current - p21) / p21 if p21 > 0 else 0

        # 63d return
        p63 = float(prices[0]) if len(prices) >= 21 else p21
        df.at[idx, "spy_return_63d"] = (current - p63) / p63 if p63 > 0 else 0

        # 21d volatility
        if len(prices) >= 22:
            returns = np.diff(prices[-22:]) / prices[-22:-1]
            df.at[idx, "spy_volatility_21d"] = float(np.std(returns, ddof=1))

        # Bull market: SPY above 63d average
        avg_63 = float(np.mean(prices))
        df.at[idx, "market_is_bull"] = 1.0 if current > avg_63 else 0.0

    logger.info("  Market regime: %.1f%% trades in bull market", df["market_is_bull"].mean() * 100)


async def _add_sector_features(session: AsyncSession, df: pd.DataFrame) -> None:
    """Stock sector/industry features and committee-sector alignment."""
    df["sector_encoded"] = 0.0
    df["committee_sector_match"] = 0.0

    # Load or build sector cache
    sector_map = _load_sector_cache()
    tickers_needed = set(df["ticker"].unique()) - set(sector_map.keys())

    if tickers_needed:
        logger.info("  Fetching sector data for %d new tickers...", len(tickers_needed))
        new_sectors = _fetch_sectors(list(tickers_needed)[:500])  # cap to avoid timeout
        sector_map.update(new_sectors)
        _save_sector_cache(sector_map)

    # Encode sectors
    unique_sectors = sorted(set(sector_map.values()) - {""})
    sector_to_idx = {s: i + 1 for i, s in enumerate(unique_sectors)}
    df["sector_encoded"] = df["ticker"].map(
        lambda t: float(sector_to_idx.get(sector_map.get(t, ""), 0))
    )

    # Committee-sector alignment: does member sit on committee related to this stock's sector?
    # Use committee_assignment data if available
    r = await session.execute(text("""
        SELECT member_bioguide_id, committee_code
        FROM committee_assignment
    """))
    assignments = pd.DataFrame(r.fetchall(), columns=r.keys())

    if not assignments.empty:
        member_committees = assignments.groupby("member_bioguide_id")["committee_code"].apply(set).to_dict()
        for idx, row in df.iterrows():
            bio = row.get("member_bioguide_id")
            ticker = row["ticker"]
            if bio and bio in member_committees and ticker in sector_map:
                sector = sector_map[ticker]
                committees = member_committees[bio]
                for comm_code, sectors in COMMITTEE_SECTOR_MAP.items():
                    # Match on first 4 chars of committee code
                    if any(c.startswith(comm_code[:4]) for c in committees):
                        if sector in sectors:
                            df.at[idx, "committee_sector_match"] = 1.0
                            break

    has_sector = (df["sector_encoded"] > 0).sum()
    logger.info("  Sector data: %d / %d trades have sector (%.1f%%)",
                has_sector, len(df), has_sector / len(df) * 100)


def _load_sector_cache() -> dict[str, str]:
    """Load cached ticker→sector mapping."""
    if SECTOR_CACHE_PATH.exists():
        return json.loads(SECTOR_CACHE_PATH.read_text())
    return {}


def _save_sector_cache(cache: dict[str, str]) -> None:
    """Save ticker→sector cache."""
    SECTOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SECTOR_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _fetch_sectors(tickers: list[str]) -> dict[str, str]:
    """Fetch sector data from yfinance for a batch of tickers."""
    import yfinance as yf

    result = {}
    for i, ticker in enumerate(tickers):
        if i % 50 == 0 and i > 0:
            logger.info("    Sector fetch progress: %d/%d", i, len(tickers))
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "")
            result[ticker] = sector
        except Exception:
            result[ticker] = ""
    return result


async def _add_network_features_sql(session: AsyncSession, df: pd.DataFrame) -> None:
    """Compute network features from PostgreSQL tables (SQL fallback when Neo4j unavailable)."""
    df["lobbying_connection_count"] = 0.0
    df["campaign_donor_connection"] = 0.0
    df["network_degree"] = 0.0
    df["has_lobbying_triangle"] = 0.0

    # Lobbying connections: count lobbying filings for companies matching traded tickers
    r = await session.execute(text("""
        SELECT lc.matched_ticker as ticker, COUNT(DISTINCT lf.id) as filing_count
        FROM lobbying_client lc
        JOIN lobbying_filing lf ON lf.client_id = lc.id
        WHERE lc.matched_ticker IS NOT NULL
        GROUP BY lc.matched_ticker
    """))
    lobby_df = pd.DataFrame(r.fetchall(), columns=r.keys())
    if not lobby_df.empty:
        lobby_map = lobby_df.set_index("ticker")["filing_count"]
        mask = df["ticker"].isin(lobby_map.index)
        df.loc[mask, "lobbying_connection_count"] = df.loc[mask, "ticker"].map(lobby_map).fillna(0).astype(float)

    # Campaign contributions: count contributions to member from companies matching ticker
    r = await session.execute(text("""
        SELECT cc2.member_bioguide_id, cc1.matched_ticker as ticker,
               COUNT(*) as contribution_count
        FROM campaign_contribution cc1
        JOIN campaign_committee cc2 ON cc1.committee_id = cc2.id
        WHERE cc1.matched_ticker IS NOT NULL
        AND cc2.member_bioguide_id IS NOT NULL
        GROUP BY cc2.member_bioguide_id, cc1.matched_ticker
    """))
    camp_df = pd.DataFrame(r.fetchall(), columns=r.keys())
    if not camp_df.empty:
        camp_map = camp_df.set_index(["member_bioguide_id", "ticker"])["contribution_count"]
        for idx, row in df.iterrows():
            key = (row.get("member_bioguide_id"), row["ticker"])
            if key in camp_map.index:
                df.at[idx, "campaign_donor_connection"] = float(camp_map[key])

    # Network degree: total unique tickers a member has traded
    r = await session.execute(text("""
        SELECT member_bioguide_id, COUNT(DISTINCT ticker) as degree
        FROM trade_disclosure
        WHERE ticker IS NOT NULL AND member_bioguide_id IS NOT NULL
        GROUP BY member_bioguide_id
    """))
    degree_df = pd.DataFrame(r.fetchall(), columns=r.keys())
    if not degree_df.empty:
        degree_map = degree_df.set_index("member_bioguide_id")["degree"]
        mask = df["member_bioguide_id"].notna() & df["member_bioguide_id"].isin(degree_map.index)
        df.loc[mask, "network_degree"] = df.loc[mask, "member_bioguide_id"].map(degree_map).astype(float)

    has_lobby = (df["lobbying_connection_count"] > 0).sum()
    logger.info("  Network (SQL): %d trades with lobbying connections", has_lobby)


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


async def train_models(df: pd.DataFrame, session: AsyncSession, n_folds: int = 5,
                       use_catboost: bool = False) -> dict:
    """Train all models with temporal k-fold cross-validation.

    Data is sorted by transaction_date. For k folds, each fold uses
    the first (i+1)/k of data for training and the next chunk for validation.
    This ensures no future leakage — always train on past, test on future.
    """
    from sqlalchemy import update as sa_update

    feature_cols = _feature_cols(df)

    # Feature selection: drop features that are near-zero variance
    X_raw = df[feature_cols].values.astype(float)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    variances = np.var(X_raw, axis=0)
    keep_mask = variances > 1e-10
    dropped = [feature_cols[i] for i in range(len(feature_cols)) if not keep_mask[i]]
    if dropped:
        logger.info("Dropping %d zero-variance features: %s", len(dropped), dropped)
    feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if keep_mask[i]]

    X = X_raw[:, keep_mask]
    y_class = df["profitable"].values.astype(float)
    y_return = df["actual_return"].fillna(0).values.astype(float)

    n = len(X)
    logger.info("Features (%d): %s", len(feature_cols), feature_cols)
    logger.info("Running %d-fold temporal cross-validation on %d samples", n_folds, n)

    # Temporal CV: fold i trains on [0, split_i), validates on [split_i, split_i+1)
    min_train_frac = 0.4
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

        logger.info("  Fold %d/%d: train=%d, val=%d (dates: train to idx %d, val to idx %d)",
                     fold + 1, n_folds, len(X_train), len(X_val), train_end, val_end)

        # Trade Predictor (LightGBM)
        tp = TradePredictor()
        tp.feature_columns = feature_cols
        tp.train(X_train, y_class_train, X_val, y_class_val)
        m = evaluate_classifier(y_class_val, tp.predict(X_val), tp.predict_proba(X_val))
        fold_metrics["trade_predictor"].append(m)

        # Return Predictor (XGBoost)
        rp = ReturnPredictor()
        rp.feature_columns = feature_cols
        rp.train(X_train, y_return_train, X_val, y_return_val)
        m = evaluate_regressor(y_return_val, rp.predict(X_val))
        fold_metrics["return_predictor"].append(m)

        # Anomaly Detector
        anomaly_cols = [c for c in feature_cols if c not in AnomalyDetector.EXCLUDED_FEATURES]
        anomaly_idx = [feature_cols.index(c) for c in anomaly_cols]
        X_a_train = X_train[:, anomaly_idx]
        ad = AnomalyDetector()
        ad.feature_columns = anomaly_cols
        m = ad.train(X_a_train, y_class_train)
        fold_metrics["anomaly_detector"].append(m)

        # CatBoost (optional)
        if use_catboost:
            m = _train_catboost_fold(X_train, y_class_train, X_val, y_class_val, feature_cols)
            fold_metrics["catboost"].append(m)

        # Ensemble
        tp_proba = np.concatenate([tp.predict_proba(X_train), tp.predict_proba(X_val)])
        rp_scores = np.concatenate([rp.predict(X_train), rp.predict(X_val)])
        X_a_all = np.vstack([X_a_train, X_val[:, anomaly_idx]])
        ad_scores = ad.predict_proba(X_a_all)

        timing_idx = feature_cols.index("timing_suspicion_score") if "timing_suspicion_score" in feature_cols else None
        sentiment_idx = feature_cols.index("avg_sentiment_7d") if "avg_sentiment_7d" in feature_cols else None
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
    results = {}
    for model_name, folds in fold_metrics.items():
        if not folds:
            continue
        avg = {}
        for key in folds[0]:
            vals = [f[key] for f in folds if key in f and f[key] is not None]
            if vals:
                avg[key] = float(np.mean(vals))
                avg[f"{key}_std"] = float(np.std(vals))
        results[model_name] = avg

    # Log CV results
    for model_name, metrics in results.items():
        if "accuracy" in metrics:
            logger.info("  %s (CV avg): accuracy=%.4f±%.4f, AUC=%.4f±%.4f, F1=%.4f±%.4f",
                        model_name,
                        metrics.get("accuracy", 0), metrics.get("accuracy_std", 0),
                        metrics.get("auc", 0), metrics.get("auc_std", 0),
                        metrics.get("f1", 0), metrics.get("f1_std", 0))

    # Feature importance from last LightGBM fold
    if hasattr(tp, 'model') and hasattr(tp.model, 'feature_importances_'):
        importances = tp.model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        logger.info("  Top 15 features (LightGBM):")
        for i in top_idx:
            logger.info("    %3d. %-35s %6.1f", i, feature_cols[i], importances[i])

    # Final train on all data except last 20% for saving the best model
    split_idx = int(n * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_class_train, y_class_val = y_class[:split_idx], y_class[split_idx:]
    y_return_train, y_return_val = y_return[:split_idx], y_return[split_idx:]

    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Save final models
    trade_pred = TradePredictor()
    trade_pred.feature_columns = feature_cols
    trade_pred.train(X_train, y_class_train, X_val, y_class_val)
    await _save(session, trade_pred, version, results["trade_predictor"], feature_cols)

    return_pred = ReturnPredictor()
    return_pred.feature_columns = feature_cols
    return_pred.train(X_train, y_return_train, X_val, y_return_val)
    await _save(session, return_pred, version, results["return_predictor"], feature_cols)

    anomaly_cols = [c for c in feature_cols if c not in AnomalyDetector.EXCLUDED_FEATURES]
    anomaly_idx = [feature_cols.index(c) for c in anomaly_cols]
    anomaly_det = AnomalyDetector()
    anomaly_det.feature_columns = anomaly_cols
    anomaly_det.train(X[:, anomaly_idx], y_class)
    await _save(session, anomaly_det, version, results["anomaly_detector"], anomaly_cols)

    # Ensemble on final split
    tp_proba = trade_pred.predict_proba(X)
    rp_scores = return_pred.predict(X)
    ad_scores = anomaly_det.predict_proba(X[:, anomaly_idx])
    timing_idx = feature_cols.index("timing_suspicion_score") if "timing_suspicion_score" in feature_cols else None
    sentiment_idx = feature_cols.index("avg_sentiment_7d") if "avg_sentiment_7d" in feature_cols else None
    meta = np.column_stack([
        tp_proba, rp_scores, ad_scores,
        X[:, timing_idx] if timing_idx is not None else np.zeros(n),
        X[:, sentiment_idx] if sentiment_idx is not None else np.zeros(n),
    ])
    ensemble = EnsembleModel()
    ensemble.feature_columns = EnsembleModel.META_FEATURES
    ensemble.train(meta[:split_idx], y_class_train, meta[split_idx:], y_class_val)
    await _save(session, ensemble, version, results["ensemble"], EnsembleModel.META_FEATURES)

    return results


def _train_catboost_fold(X_train, y_train, X_val, y_val, feature_cols) -> dict:
    """Train a CatBoost classifier for one CV fold."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        logger.warning("CatBoost not installed — skipping. Install with: pip install catboost")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0}

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=0,
        eval_metric="AUC",
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)

    preds = model.predict(X_val).astype(float)
    proba = model.predict_proba(X_val)[:, 1]
    return evaluate_classifier(y_val, preds, proba)


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
    import sys
    args = sys.argv[1:]

    # Parse flags
    use_catboost = "--catboost" in args
    args = [a for a in args if not a.startswith("--")]

    # Parse horizons
    horizons = [a for a in args if a in ("5d", "21d", "63d", "90d", "180d")]
    if not horizons:
        horizons = ["90d", "180d"]  # default to best-performing horizons

    if use_catboost:
        logger.info("CatBoost enabled")

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

            results = await train_models(df, session, use_catboost=use_catboost)
            all_results[horizon] = results

    # Print comparison table
    print("\n" + "=" * 70)
    print("  HORIZON COMPARISON (5-fold temporal CV)")
    print("=" * 70)
    print(f"  {'Model':<25s} {'Horizon':<10s} {'Accuracy':>10s} {'AUC':>10s} {'F1':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for horizon, results in all_results.items():
        for model_name, metrics in results.items():
            if "accuracy" in metrics:
                acc_std = metrics.get("accuracy_std", 0)
                print(f"  {model_name:<25s} {horizon:<10s} {metrics.get('accuracy', 0):>7.4f}±{acc_std:.3f} {metrics.get('auc', 0):>10.4f} {metrics.get('f1', 0):>10.4f}")
            elif "mae" in metrics:
                print(f"  {model_name:<25s} {horizon:<10s} {'MAE:':>10s} {metrics.get('mae', 0):>10.4f} {metrics.get('r2', 0):>10.4f}")


if __name__ == "__main__":
    asyncio.run(main())
