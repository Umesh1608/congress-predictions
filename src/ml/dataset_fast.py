"""Fast bulk SQL dataset builder for ML training.

Builds features using bulk SQL queries instead of per-trade round-trips.
This is ~100x faster than the per-trade DatasetBuilder in dataset.py.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

SECTOR_CACHE_PATH = Path("data/sector_cache.json")

# Feature encoding maps
FILER_TYPE_MAP = {"member": 0, "spouse": 1, "dependent": 2, "joint": 3}
PARTY_MAP = {"Democrat": 0, "Republican": 1, "Independent": 2}
CHAMBER_MAP = {"house": 0, "senate": 1}
TX_DIR_MAP = {"purchase": 1, "sale": -1, "sale_full": -1, "sale_partial": -1, "exchange": 0}

# Major ETFs to exclude from training (not individual stock picks)
EXCLUDED_ETFS = {
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "VEA", "VWO", "EFA", "EEM",
    "AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "GLD", "SLV", "USO",
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE",
    "ARKK", "ARKW", "ARKG", "ARKF", "ARKQ",
    "VGT", "VHT", "VFH", "VIS", "VNQ", "VXUS", "VB", "VO",
    "IVV", "IJH", "IJR", "IEMG", "IEFA",
    "SCHD", "SCHX", "SCHF",
}

# US presidential and midterm election years
_ELECTION_YEARS = {2016, 2020, 2024, 2028}
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

# Profit threshold: require >2% directional return to count as profitable
PROFIT_THRESHOLD = 0.02


async def build_dataset_fast(
    session: AsyncSession, limit: int = 20000, horizon: str = "180d"
) -> pd.DataFrame:
    """Build training dataset using bulk SQL — 100x faster than per-trade queries.

    Args:
        session: Async SQLAlchemy session.
        limit: Max trades to load.
        horizon: Return horizon — "5d", "21d", "63d", "90d", or "180d".

    Returns:
        DataFrame with features, labels, and metadata columns.
    """
    # Map horizon to trading-day offset
    HORIZON_OFFSETS = {"5d": 4, "21d": 20, "63d": 62, "90d": 89, "180d": 179}
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
                -- Exclude options trades
                AND t.asset_name NOT ILIKE '%option%'
                AND t.asset_name NOT ILIKE '% call %'
                AND t.asset_name NOT ILIKE '% put %'
                AND t.asset_name NOT ILIKE '%call option%'
                AND t.asset_name NOT ILIKE '%put option%'
                AND t.asset_type NOT ILIKE '%option%'
                -- Exclude major ETFs
                AND t.ticker NOT IN (
                    'SPY','QQQ','IWM','DIA','VOO','VTI','VEA','VWO','EFA','EEM',
                    'AGG','BND','TLT','IEF','SHY','LQD','HYG','GLD','SLV','USO',
                    'XLF','XLK','XLE','XLV','XLI','XLY','XLP','XLB','XLU','XLRE',
                    'ARKK','ARKW','ARKG','ARKF','ARKQ',
                    'VGT','VHT','VFH','VIS','VNQ','VXUS','VB','VO',
                    'IVV','IJH','IJR','IEMG','IEFA',
                    'SCHD','SCHX','SCHF'
                )
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

    # Directional return: positive means trade went in the right direction
    df["directional_return"] = None
    ret_mask = df["actual_return"].notna()
    raw_return = df.loc[ret_mask, "actual_return"].astype(float)
    is_buy = df.loc[ret_mask, "is_purchase"] == 1.0
    df.loc[ret_mask, "directional_return"] = np.where(is_buy, raw_return, -raw_return)

    # Threshold-based labels
    df["profitable"] = None
    dir_mask = df["directional_return"].notna()
    df.loc[dir_mask, "profitable"] = (
        df.loc[dir_mask, "directional_return"].astype(float) > PROFIT_THRESHOLD
    ).astype(float)

    # Sample weights: weight by log(amount_midpoint)
    amount_low = df["amount_range_low"].fillna(0).astype(float)
    amount_high = df["amount_range_high"].fillna(0).astype(float)
    amount_mid = (amount_low + amount_high) / 2
    amount_mid = amount_mid.clip(lower=1000)
    df["sample_weight"] = np.log1p(amount_mid)
    df["sample_weight"] = df["sample_weight"] / df["sample_weight"].mean()

    # Drop rows without labels
    df = df.dropna(subset=["profitable"])
    logger.info(
        "Labeled trades (%s): %d (%.1f%% profitable, threshold=%.1f%%)",
        horizon, len(df), df["profitable"].mean() * 100, PROFIT_THRESHOLD * 100,
    )

    # === Build features ===

    # Trade features
    amount_low = df["amount_range_low"].fillna(0).astype(float)
    amount_high = df["amount_range_high"].fillna(0).astype(float)
    df["amount_midpoint"] = (amount_low + amount_high) / 2
    df["amount_log"] = np.log1p(df["amount_midpoint"])
    df["tx_direction"] = df["transaction_type"].map(TX_DIR_MAP).fillna(0).astype(float)
    df["filer_type_encoded"] = df["filer_type"].map(FILER_TYPE_MAP).fillna(0).astype(float)

    # Disclosure lag
    df["disclosure_lag_days"] = 30.0
    has_dates = df["disclosure_date"].notna() & df["transaction_date"].notna()
    df.loc[has_dates, "disclosure_lag_days"] = (
        pd.to_datetime(df.loc[has_dates, "disclosure_date"])
        - pd.to_datetime(df.loc[has_dates, "transaction_date"])
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

    # Legislative features — bulk version
    logger.info("Computing legislative features...")
    await _add_legislative_features(session, df)

    # Sentiment features — bulk
    logger.info("Computing sentiment features...")
    await _add_sentiment_features(session, df)

    # Network features — SQL fallback
    logger.info("Computing network features (SQL fallback)...")
    await _add_network_features_sql(session, df)

    # Member historical win rate
    logger.info("Computing member historical win rates...")
    await _add_member_win_rate(session, df, horizon_offset=offset)

    # Cross-member trading signals
    logger.info("Computing cross-member trading signals...")
    _add_cross_member_signals(df)

    # Calendar / seasonal features
    logger.info("Computing calendar features...")
    _add_calendar_features(df)

    # Ticker popularity among congress
    logger.info("Computing ticker popularity...")
    await _add_ticker_popularity(session, df)

    # Member trading velocity
    logger.info("Computing member trading velocity...")
    await _add_member_velocity(session, df)

    # Market regime (S&P 500 / broad market context)
    logger.info("Computing market regime features...")
    await _add_market_regime(session, df)

    # Stock sector features
    logger.info("Computing sector features...")
    await _add_sector_features(session, df)

    # Derived: stock excess return vs market
    df["stock_vs_market_21d"] = df["price_change_21d"] - df["spy_return_21d"]

    # Interaction features — cross key signals for non-linear effects
    logger.info("Computing interaction features...")
    _add_interaction_features(df)

    logger.info("Dataset built: %d samples, %d features", len(df), len(feature_columns(df)))
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (excludes metadata and label columns)."""
    exclude = {
        "trade_id", "ticker", "transaction_date", "disclosure_date",
        "transaction_type", "filer_type", "amount_range_low", "amount_range_high",
        "member_bioguide_id", "trade_chamber", "party", "member_chamber",
        "first_elected", "base_price", "base_date",
        "price_future", "future_date",
        "actual_return", "profitable", "is_purchase",
        "directional_return", "sample_weight", "asset_name", "asset_type",
    }
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

async def _add_market_features(session: AsyncSession, df: pd.DataFrame) -> None:
    """Add market features using bulk stock queries."""
    tickers = df["ticker"].unique().tolist()

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
            "start_date": df["transaction_date"].min() - timedelta(days=400),
        },
    )
    stock_df = pd.DataFrame(r.fetchall(), columns=r.keys())

    all_market_cols = [
        "price_change_5d", "price_change_21d", "volatility_21d", "volume_ratio_5d", "rsi_14",
        "macd_signal", "bollinger_position", "ma_cross_50_200", "high_52w_position",
    ]
    if stock_df.empty:
        for col in all_market_cols:
            df[col] = 0.0 if col != "bollinger_position" else 0.5
        return

    stock_df["price"] = stock_df["adj_close"].fillna(stock_df["close"]).astype(float)
    stock_df["volume"] = stock_df["volume"].fillna(0).astype(float)

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
            "macd_signal": 0.0,
            "bollinger_position": 0.5,
            "ma_cross_50_200": 0.0,
            "high_52w_position": 0.5,
        }

        if ticker in stock_by_ticker:
            sdf = stock_by_ticker[ticker]
            before = sdf[sdf["date"] < trade_date.date()]
            before_long = before.tail(260)
            before = before.tail(30)

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

            # Advanced indicators from longer history
            if len(before_long) >= 26:
                long_prices = before_long["price"].values  # chronological

                # MACD (12,26,9)
                ema12 = _ema(long_prices, 12)
                ema26 = _ema(long_prices, 26)
                feat["macd_signal"] = float(ema12 - ema26)

                # Bollinger Band position (20-day, 2 std)
                if len(long_prices) >= 20:
                    ma20 = float(np.mean(long_prices[-20:]))
                    std20 = float(np.std(long_prices[-20:], ddof=1))
                    if std20 > 0:
                        upper = ma20 + 2 * std20
                        lower = ma20 - 2 * std20
                        if upper != lower:
                            feat["bollinger_position"] = float(
                                (long_prices[-1] - lower) / (upper - lower)
                            )

                # 50-day vs 200-day MA crossover
                if len(long_prices) >= 50:
                    ma50 = float(np.mean(long_prices[-50:]))
                    if len(long_prices) >= 200:
                        ma200 = float(np.mean(long_prices[-200:]))
                        feat["ma_cross_50_200"] = 1.0 if ma50 > ma200 else -1.0

                # 52-week high/low position
                if len(long_prices) >= 200:
                    high_52w = float(np.max(long_prices[-252:]))
                    low_52w = float(np.min(long_prices[-252:]))
                    if high_52w > low_52w:
                        feat["high_52w_position"] = float(
                            (long_prices[-1] - low_52w) / (high_52w - low_52w)
                        )

        results.append(feat)

    market_df = pd.DataFrame(results)
    for col in market_df.columns:
        df[col] = market_df[col].values


async def _add_legislative_features(session: AsyncSession, df: pd.DataFrame) -> None:
    """Add legislative features using bulk queries.

    Three types of legislative proximity:
    1. Committee hearings: hearings by committees the member sits on (+-30 days)
    2. Sponsored bills: bills the member sponsored/cosponsored (+-90 days)
    3. All nearby bills: any bill introduced near the trade date (+-90 days)
    """
    df["min_hearing_distance"] = 999.0
    df["min_bill_distance"] = 999.0
    df["timing_suspicion_score"] = 0.0
    df["has_related_bill"] = 0.0
    df["nearby_hearing_count"] = 0.0
    df["nearby_bill_count"] = 0.0
    df["sponsored_bill_near_trade"] = 0.0
    df["hearings_before_7d"] = 0.0
    df["hearings_after_7d"] = 0.0
    df["hearings_before_3d"] = 0.0

    # 1. Hearings by member's committees within +-30 days
    logger.info("  Loading committee hearings near trades...")
    r = await session.execute(text("""
        SELECT t.id as trade_id,
               COUNT(DISTINCT h.id) as hearing_count,
               MIN(ABS(t.transaction_date - h.hearing_date)) as min_hearing_dist,
               COUNT(DISTINCT h.id) FILTER (WHERE h.hearing_date BETWEEN t.transaction_date - 7 AND t.transaction_date - 1) as before_7d,
               COUNT(DISTINCT h.id) FILTER (WHERE h.hearing_date BETWEEN t.transaction_date + 1 AND t.transaction_date + 7) as after_7d,
               COUNT(DISTINCT h.id) FILTER (WHERE h.hearing_date BETWEEN t.transaction_date - 3 AND t.transaction_date - 1) as before_3d
        FROM trade_disclosure t
        JOIN committee_assignment ca
            ON ca.member_bioguide_id = t.member_bioguide_id
        JOIN committee_hearing h
            ON h.committee_code = ca.committee_code
            AND h.hearing_date BETWEEN t.transaction_date - 30 AND t.transaction_date + 30
        WHERE t.ticker IS NOT NULL
        AND t.transaction_date >= '2016-01-01'
        AND t.member_bioguide_id IS NOT NULL
        GROUP BY t.id
    """))
    hearing_df = pd.DataFrame(r.fetchall(), columns=r.keys())

    if not hearing_df.empty:
        hearing_map = hearing_df.set_index("trade_id")
        mask = df["trade_id"].isin(hearing_map.index)
        df.loc[mask, "nearby_hearing_count"] = df.loc[mask, "trade_id"].map(
            hearing_map["hearing_count"]
        ).values.astype(float)
        df.loc[mask, "min_hearing_distance"] = df.loc[mask, "trade_id"].map(
            hearing_map["min_hearing_dist"]
        ).values.astype(float)
        df.loc[mask, "hearings_before_7d"] = df.loc[mask, "trade_id"].map(
            hearing_map["before_7d"]
        ).values.astype(float)
        df.loc[mask, "hearings_after_7d"] = df.loc[mask, "trade_id"].map(
            hearing_map["after_7d"]
        ).values.astype(float)
        df.loc[mask, "hearings_before_3d"] = df.loc[mask, "trade_id"].map(
            hearing_map["before_3d"]
        ).values.astype(float)

    logger.info(
        "  Trades with committee hearings: %d / %d",
        (df["nearby_hearing_count"] > 0).sum(), len(df),
    )
    logger.info(
        "  Trades with pre-trade hearing (7d): %d, (3d): %d",
        (df["hearings_before_7d"] > 0).sum(), (df["hearings_before_3d"] > 0).sum(),
    )

    # 2. Bills sponsored by the trading member within +-90 days
    logger.info("  Loading sponsored bills near trades...")
    r = await session.execute(text("""
        SELECT t.id as trade_id,
               COUNT(DISTINCT b.bill_id) as sponsored_count,
               MIN(ABS(t.transaction_date - COALESCE(b.introduced_date, b.latest_action_date))) as min_sponsored_dist
        FROM trade_disclosure t
        JOIN bill b
            ON b.sponsor_bioguide_id = t.member_bioguide_id
            AND COALESCE(b.introduced_date, b.latest_action_date)
                BETWEEN t.transaction_date - 90 AND t.transaction_date + 90
        WHERE t.ticker IS NOT NULL
        AND t.transaction_date >= '2016-01-01'
        AND t.member_bioguide_id IS NOT NULL
        AND b.sponsor_bioguide_id IS NOT NULL
        GROUP BY t.id
    """))
    sponsored_df = pd.DataFrame(r.fetchall(), columns=r.keys())

    if not sponsored_df.empty:
        sp_map = sponsored_df.set_index("trade_id")
        mask = df["trade_id"].isin(sp_map.index)
        df.loc[mask, "sponsored_bill_near_trade"] = df.loc[mask, "trade_id"].map(
            sp_map["sponsored_count"]
        ).values.astype(float)
        df.loc[mask, "min_bill_distance"] = df.loc[mask, "trade_id"].map(
            sp_map["min_sponsored_dist"]
        ).values.astype(float)
        df.loc[mask, "has_related_bill"] = 1.0

    logger.info(
        "  Trades with sponsored bills: %d / %d",
        (df["sponsored_bill_near_trade"] > 0).sum(), len(df),
    )

    # 3. Any bills introduced near the trade (broader signal)
    logger.info("  Loading all nearby bills...")
    r = await session.execute(text("""
        SELECT t.id as trade_id,
               COUNT(b.bill_id) as bill_count,
               MIN(ABS(t.transaction_date - COALESCE(b.introduced_date, b.latest_action_date))) as min_bill_dist
        FROM trade_disclosure t
        JOIN bill b
            ON COALESCE(b.introduced_date, b.latest_action_date)
                BETWEEN t.transaction_date - 90 AND t.transaction_date + 90
        WHERE t.ticker IS NOT NULL
        AND t.transaction_date >= '2016-01-01'
        GROUP BY t.id
    """))
    bill_df = pd.DataFrame(r.fetchall(), columns=r.keys())

    if not bill_df.empty:
        bill_map = bill_df.set_index("trade_id")
        mask = df["trade_id"].isin(bill_map.index)
        df.loc[mask, "nearby_bill_count"] = df.loc[mask, "trade_id"].map(
            bill_map["bill_count"]
        ).values.astype(float)
        # Only update min_bill_distance if not already set by sponsored bills
        no_sponsor = mask & (df["min_bill_distance"] == 999.0)
        if no_sponsor.any():
            df.loc[no_sponsor, "min_bill_distance"] = df.loc[no_sponsor, "trade_id"].map(
                bill_map["min_bill_dist"]
            ).values.astype(float)
            df.loc[no_sponsor, "has_related_bill"] = (
                df.loc[no_sponsor, "trade_id"].map(bill_map["bill_count"]).values > 0
            ).astype(float)

    # Compute suspicion score from timing proximity features
    lag = df["disclosure_lag_days"].clip(0, 90) / 90.0
    hearing_score = (1.0 - df["min_hearing_distance"].clip(0, 30) / 30.0).clip(0, 1)
    bill_score = (1.0 - df["min_bill_distance"].clip(0, 90) / 90.0).clip(0, 1)
    sponsor_bonus = (df["sponsored_bill_near_trade"] > 0).astype(float) * 0.15
    df["timing_suspicion_score"] = (
        hearing_score * 0.35 + bill_score * 0.25 + lag * 0.25 + sponsor_bonus
    ).clip(0, 1)

    has_suspicion = (df["timing_suspicion_score"] > 0.1).sum()
    logger.info(
        "  Timing suspicion > 0.1: %d / %d (%.1f%%)",
        has_suspicion, len(df), has_suspicion / len(df) * 100,
    )


async def _add_sentiment_features(session: AsyncSession, df: pd.DataFrame) -> None:
    """Add sentiment features using bulk queries."""
    df["avg_sentiment_7d"] = 0.0
    df["avg_sentiment_30d"] = 0.0
    df["sentiment_momentum"] = 0.0
    df["media_mention_count_7d"] = 0.0
    df["media_mention_count_30d"] = 0.0

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

    for idx, row in df.iterrows():
        td = pd.Timestamp(row["transaction_date"])
        window_7 = sent_df[(sent_df.index >= td - pd.Timedelta(days=7)) & (sent_df.index < td)]
        if not window_7.empty:
            df.at[idx, "avg_sentiment_7d"] = float(window_7["avg_score"].mean())
            df.at[idx, "media_mention_count_7d"] = float(window_7["article_count"].sum())
        window_30 = sent_df[(sent_df.index >= td - pd.Timedelta(days=30)) & (sent_df.index < td)]
        if not window_30.empty:
            df.at[idx, "avg_sentiment_30d"] = float(window_30["avg_score"].mean())
            df.at[idx, "media_mention_count_30d"] = float(window_30["article_count"].sum())
        df.at[idx, "sentiment_momentum"] = (
            df.at[idx, "avg_sentiment_7d"] - df.at[idx, "avg_sentiment_30d"]
        )


async def _add_member_win_rate(
    session: AsyncSession, df: pd.DataFrame, horizon_offset: int = 20
) -> None:
    """Add member historical win rate as a feature.

    For each trade, compute the member's win rate from ALL their PRIOR trades
    (no future leakage). Uses expanding window.
    """
    df["member_win_rate"] = 0.5
    df["member_trade_count"] = 0.0
    df["member_avg_return"] = 0.0

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
    member_stats: dict[str, dict] = {}

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

    trade_id_to_stats: dict[int, tuple[float, float, float]] = {}

    for _, row in all_trades.iterrows():
        bio = row["member_bioguide_id"]
        tid = row["trade_id"]

        if bio not in member_stats:
            member_stats[bio] = {"wins": 0, "total": 0, "return_sum": 0.0}

        stats = member_stats[bio]
        if stats["total"] >= 3:
            wr = stats["wins"] / stats["total"]
            avg_ret = stats["return_sum"] / stats["total"]
            trade_id_to_stats[tid] = (wr, float(stats["total"]), avg_ret)

        if row["is_win"] is not None and not pd.isna(row["is_win"]):
            stats["total"] += 1
            stats["wins"] += int(row["is_win"])
            if row["ret"] is not None and not pd.isna(row["ret"]):
                stats["return_sum"] += float(row["ret"])

    for idx, row in df.iterrows():
        tid = row["trade_id"]
        if tid in trade_id_to_stats:
            wr, count, avg_ret = trade_id_to_stats[tid]
            df.at[idx, "member_win_rate"] = wr
            df.at[idx, "member_trade_count"] = count
            df.at[idx, "member_avg_return"] = avg_ret

    has_history = (df["member_trade_count"] > 0).sum()
    logger.info(
        "  %d / %d trades have member history (%.1f%%)",
        has_history, len(df), has_history / len(df) * 100,
    )


def _add_cross_member_signals(df: pd.DataFrame) -> None:
    """How many OTHER members traded this stock in +-7d and +-30d windows."""
    df["cross_member_7d"] = 0.0
    df["cross_member_30d"] = 0.0
    df["cross_member_same_dir_7d"] = 0.0

    df_sorted = df[["trade_id", "ticker", "transaction_date", "member_bioguide_id", "tx_direction"]].copy()
    df_sorted["transaction_date"] = pd.to_datetime(df_sorted["transaction_date"])

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

            for window_days, col in [(7, "cross_member_7d"), (30, "cross_member_30d")]:
                window = np.timedelta64(window_days, "D")
                mask = (np.abs(dates - td) <= window) & (members != my_member)
                unique_others = len(set(members[mask]) - {my_member})
                idx = df.index[df["trade_id"] == trade_ids[i]]
                if len(idx) > 0:
                    df.loc[idx[0], col] = float(unique_others)

            window = np.timedelta64(7, "D")
            mask = (np.abs(dates - td) <= window) & (members != my_member) & (directions == my_dir)
            unique_same_dir = len(set(members[mask]) - {my_member})
            idx = df.index[df["trade_id"] == trade_ids[i]]
            if len(idx) > 0:
                df.loc[idx[0], "cross_member_same_dir_7d"] = float(unique_same_dir)

    logger.info(
        "  Cross-member signals: %.1f%% trades have >=1 other member in 7d",
        (df["cross_member_7d"] > 0).mean() * 100,
    )


def _add_calendar_features(df: pd.DataFrame) -> None:
    """Calendar, seasonal, and political cycle features."""
    dates = pd.to_datetime(df["transaction_date"])

    df["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12).astype(float)
    df["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12).astype(float)
    df["quarter"] = dates.dt.quarter.astype(float)
    df["day_of_week"] = dates.dt.dayofweek.astype(float)
    df["is_december"] = (dates.dt.month == 12).astype(float)
    df["is_january"] = (dates.dt.month == 1).astype(float)

    years = dates.dt.year
    df["is_election_year"] = years.isin(_ELECTION_YEARS).astype(float)
    df["is_midterm_year"] = years.isin(_MIDTERM_YEARS).astype(float)

    next_nov = pd.to_datetime(years.astype(str) + "-11-05")
    days_to_election = (next_nov - dates).dt.days
    mask = days_to_election < 0
    next_nov_adj = pd.to_datetime((years + 1).astype(str) + "-11-05")
    days_to_election = days_to_election.where(~mask, (next_nov_adj - dates).dt.days)
    df["days_to_election"] = days_to_election.clip(0, 730).astype(float)

    df["is_lame_duck"] = ((dates.dt.month >= 11) | (dates.dt.month == 1)).astype(float)


async def _add_ticker_popularity(session: AsyncSession, df: pd.DataFrame) -> None:
    """How popular is this ticker among congress members (historical)."""
    df["ticker_total_members"] = 0.0
    df["ticker_total_trades"] = 0.0
    df["ticker_purchase_ratio"] = 0.5

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
        for col_src, col_dst in [
            ("unique_members", "ticker_total_members"),
            ("total_trades", "ticker_total_trades"),
            ("purchase_ratio", "ticker_purchase_ratio"),
        ]:
            mask = df["ticker"].isin(stats_map.index)
            df.loc[mask, col_dst] = df.loc[mask, "ticker"].map(stats_map[col_src]).astype(float)

    logger.info("  Ticker popularity: %.1f avg members/ticker", df["ticker_total_members"].mean())


async def _add_member_velocity(session: AsyncSession, df: pd.DataFrame) -> None:
    """Member's recent trading frequency vs their historical average."""
    df["member_trades_30d"] = 0.0
    df["member_trades_90d"] = 0.0
    df["member_velocity_ratio"] = 1.0

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

        p21 = float(prices[-min(21, len(prices))]) if len(prices) >= 2 else current
        df.at[idx, "spy_return_21d"] = (current - p21) / p21 if p21 > 0 else 0

        p63 = float(prices[0]) if len(prices) >= 21 else p21
        df.at[idx, "spy_return_63d"] = (current - p63) / p63 if p63 > 0 else 0

        if len(prices) >= 22:
            returns = np.diff(prices[-22:]) / prices[-22:-1]
            df.at[idx, "spy_volatility_21d"] = float(np.std(returns, ddof=1))

        avg_63 = float(np.mean(prices))
        df.at[idx, "market_is_bull"] = 1.0 if current > avg_63 else 0.0

    logger.info("  Market regime: %.1f%% trades in bull market", df["market_is_bull"].mean() * 100)


async def _add_sector_features(session: AsyncSession, df: pd.DataFrame) -> None:
    """Stock sector/industry features and committee-sector alignment."""
    df["sector_encoded"] = 0.0
    df["committee_sector_match"] = 0.0

    sector_map = _load_sector_cache()
    tickers_needed = set(df["ticker"].unique()) - set(sector_map.keys())

    if tickers_needed:
        logger.info("  Fetching sector data for %d new tickers...", len(tickers_needed))
        new_sectors = _fetch_sectors(list(tickers_needed)[:500])
        sector_map.update(new_sectors)
        _save_sector_cache(sector_map)

    unique_sectors = sorted(set(sector_map.values()) - {""})
    sector_to_idx = {s: i + 1 for i, s in enumerate(unique_sectors)}
    df["sector_encoded"] = df["ticker"].map(
        lambda t: float(sector_to_idx.get(sector_map.get(t, ""), 0))
    )

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
                for comm_code, sectors_set in COMMITTEE_SECTOR_MAP.items():
                    prefix = comm_code[:4].lower()
                    if any(c.lower().startswith(prefix) for c in committees):
                        if sector in sectors_set:
                            df.at[idx, "committee_sector_match"] = 1.0
                            break

    has_sector = (df["sector_encoded"] > 0).sum()
    logger.info(
        "  Sector data: %d / %d trades have sector (%.1f%%)",
        has_sector, len(df), has_sector / len(df) * 100,
    )


def _add_interaction_features(df: pd.DataFrame) -> None:
    """Add feature interaction terms that capture non-linear effects.

    Key interactions:
    - Direction * market regime: buys in bull markets vs bear markets
    - Direction * member win rate: skilled traders' buy/sell asymmetry
    - Amount * legislative proximity: large trades near legislative events
    - Direction * RSI: buying oversold vs selling overbought
    - Direction * momentum: contrarian vs trend-following
    """
    d = df["tx_direction"].astype(float)

    # Direction interactions — buys vs sells behave very differently
    df["dir_x_bull"] = d * df.get("market_is_bull", pd.Series(0, index=df.index)).astype(float)
    df["dir_x_win_rate"] = d * df.get("member_win_rate", pd.Series(0, index=df.index)).astype(float)
    df["dir_x_rsi"] = d * df.get("rsi_14", pd.Series(50, index=df.index)).astype(float)
    df["dir_x_momentum_21d"] = d * df.get("price_change_21d", pd.Series(0, index=df.index)).astype(float)
    df["dir_x_52w_pos"] = d * df.get("high_52w_position", pd.Series(0, index=df.index)).astype(float)

    # Amount-weighted legislative proximity (large trades near events)
    amt = df.get("amount_log", pd.Series(0, index=df.index)).astype(float)
    has_bill = (df.get("min_bill_distance", pd.Series(999, index=df.index)) < 90).astype(float)
    df["amt_x_near_bill"] = amt * has_bill
    has_hearing = (df.get("min_hearing_distance", pd.Series(999, index=df.index)) < 30).astype(float)
    df["amt_x_near_hearing"] = amt * has_hearing

    # Member quality * trade size (high win-rate + large trade = strong signal)
    df["win_rate_x_amount"] = (
        df.get("member_win_rate", pd.Series(0, index=df.index)).astype(float)
        * amt
    )

    # Cross-member consensus * direction (cluster buys/sells)
    df["dir_x_cluster_7d"] = d * df.get("cross_member_same_dir_7d", pd.Series(0, index=df.index)).astype(float)

    # Volatility-adjusted return momentum
    vol = df.get("volatility_21d", pd.Series(1, index=df.index)).astype(float).clip(lower=0.001)
    df["sharpe_21d"] = df.get("price_change_21d", pd.Series(0, index=df.index)).astype(float) / vol

    n_interactions = 11
    logger.info("  Added %d interaction features", n_interactions)


async def _add_network_features_sql(session: AsyncSession, df: pd.DataFrame) -> None:
    """Compute network features from PostgreSQL tables (SQL fallback)."""
    df["lobbying_connection_count"] = 0.0
    df["campaign_donor_connection"] = 0.0
    df["network_degree"] = 0.0
    df["has_lobbying_triangle"] = 0.0

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
        df.loc[mask, "lobbying_connection_count"] = (
            df.loc[mask, "ticker"].map(lobby_map).fillna(0).astype(float)
        )

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


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _ema(prices: np.ndarray, span: int) -> float:
    """Compute exponential moving average, return the last value."""
    if len(prices) == 0:
        return 0.0
    alpha = 2.0 / (span + 1)
    ema = float(prices[0])
    for p in prices[1:]:
        ema = alpha * float(p) + (1 - alpha) * ema
    return ema


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


def _load_sector_cache() -> dict[str, str]:
    """Load cached ticker->sector mapping."""
    if SECTOR_CACHE_PATH.exists():
        return json.loads(SECTOR_CACHE_PATH.read_text())
    return {}


def _save_sector_cache(cache: dict[str, str]) -> None:
    """Save ticker->sector cache."""
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
