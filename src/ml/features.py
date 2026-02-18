"""Feature engineering for ML models.

Computes 6 feature groups from trade, member, market, legislative,
sentiment, and network data. All features are returned as flat dicts
suitable for DataFrame construction.
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.financial import StockDaily
from src.models.legislation import CommitteeHearing
from src.models.media import MediaContent, SentimentAnalysis
from src.models.member import CommitteeAssignment, CongressMember
from src.models.trade import TradeDisclosure
from src.processing.timing_analysis import analyze_trade_context

logger = logging.getLogger(__name__)

# Encoding maps
FILER_TYPE_MAP = {"member": 0, "spouse": 1, "dependent": 2, "joint": 3}
PARTY_MAP = {"Democrat": 0, "Republican": 1, "Independent": 2}
CHAMBER_MAP = {"house": 0, "senate": 1}
TRANSACTION_TYPE_MAP = {"purchase": 1, "sale": -1, "sale_full": -1, "sale_partial": -1, "exchange": 0}


def trade_features(trade: dict[str, Any]) -> dict[str, float]:
    """Compute features from a trade record.

    Args:
        trade: Dict with TradeDisclosure fields.
    """
    amount_low = float(trade.get("amount_range_low") or 0)
    amount_high = float(trade.get("amount_range_high") or 0)
    amount_midpoint = (amount_low + amount_high) / 2 if (amount_low or amount_high) else 0

    tx_type = trade.get("transaction_type", "")
    is_purchase = 1.0 if tx_type == "purchase" else 0.0
    tx_direction = float(TRANSACTION_TYPE_MAP.get(tx_type, 0))

    filer = trade.get("filer_type", "member")
    filer_encoded = float(FILER_TYPE_MAP.get(filer, 0))

    # Disclosure lag
    disclosure_lag = None
    tx_date = trade.get("transaction_date")
    disc_date = trade.get("disclosure_date")
    if tx_date and disc_date:
        if isinstance(tx_date, date) and isinstance(disc_date, date):
            disclosure_lag = (disc_date - tx_date).days

    return {
        "amount_midpoint": amount_midpoint,
        "amount_log": math.log1p(amount_midpoint),
        "is_purchase": is_purchase,
        "tx_direction": tx_direction,
        "filer_type_encoded": filer_encoded,
        "disclosure_lag_days": float(disclosure_lag) if disclosure_lag is not None else 30.0,
    }


def member_features(member: dict[str, Any]) -> dict[str, float]:
    """Compute features from a member record.

    Args:
        member: Dict with CongressMember fields.
    """
    party = member.get("party", "")
    party_encoded = float(PARTY_MAP.get(party, 2))
    chamber = member.get("chamber", "")
    chamber_encoded = float(CHAMBER_MAP.get(chamber, 0))

    nominate_dim1 = float(member.get("nominate_dim1") or 0)
    nominate_dim2 = float(member.get("nominate_dim2") or 0)

    first_elected = member.get("first_elected")
    current_year = date.today().year
    years_in_office = float(current_year - first_elected) if first_elected else 10.0

    committee_count = float(member.get("committee_count", 0))

    return {
        "party_encoded": party_encoded,
        "chamber_encoded": chamber_encoded,
        "nominate_dim1": nominate_dim1,
        "nominate_dim2": nominate_dim2,
        "years_in_office": years_in_office,
        "committee_count": committee_count,
    }


async def market_features(
    session: AsyncSession,
    ticker: str | None,
    trade_date: date,
) -> dict[str, float]:
    """Compute market features from StockDaily data.

    Calculates price changes, volatility, volume ratio, and RSI
    using data BEFORE the trade date (no future leakage).
    """
    defaults = {
        "price_change_5d": 0.0,
        "price_change_21d": 0.0,
        "volatility_21d": 0.0,
        "volume_ratio_5d": 0.0,
        "rsi_14": 50.0,
    }

    if not ticker:
        return defaults

    # Fetch 30 trading days of data before the trade
    start = trade_date - timedelta(days=60)
    result = await session.execute(
        select(StockDaily)
        .where(
            and_(
                StockDaily.ticker == ticker,
                StockDaily.date >= start,
                StockDaily.date < trade_date,
            )
        )
        .order_by(StockDaily.date.desc())
    )
    rows = result.scalars().all()

    if len(rows) < 2:
        return defaults

    prices = [float(r.adj_close or r.close or 0) for r in rows]
    volumes = [float(r.volume or 0) for r in rows]

    # Price changes (most recent first in rows)
    current_price = prices[0] if prices[0] > 0 else 1
    price_5d = prices[min(5, len(prices) - 1)] if len(prices) > 1 else current_price
    price_21d = prices[min(21, len(prices) - 1)] if len(prices) > 1 else current_price

    pct_5d = (current_price - price_5d) / price_5d if price_5d > 0 else 0
    pct_21d = (current_price - price_21d) / price_21d if price_21d > 0 else 0

    # Volatility: std of daily returns over 21 days
    daily_returns = []
    for i in range(min(21, len(prices) - 1)):
        if prices[i + 1] > 0:
            daily_returns.append((prices[i] - prices[i + 1]) / prices[i + 1])
    volatility = _std(daily_returns) if daily_returns else 0

    # Volume ratio: avg volume 5d / avg volume 21d
    vol_5d = sum(volumes[:5]) / max(len(volumes[:5]), 1)
    vol_21d = sum(volumes[:21]) / max(len(volumes[:21]), 1)
    volume_ratio = vol_5d / vol_21d if vol_21d > 0 else 1.0

    # RSI-14
    rsi = _compute_rsi(prices[:15])

    return {
        "price_change_5d": pct_5d,
        "price_change_21d": pct_21d,
        "volatility_21d": volatility,
        "volume_ratio_5d": volume_ratio,
        "rsi_14": rsi,
    }


async def legislative_features(
    session: AsyncSession,
    trade_id: int,
) -> dict[str, float]:
    """Compute legislative features by reusing timing_analysis module."""
    ctx = await analyze_trade_context(session, trade_id)

    if ctx is None:
        return {
            "min_hearing_distance": 999.0,
            "min_bill_distance": 999.0,
            "committee_sector_alignment": 0.0,
            "timing_suspicion_score": 0.0,
            "has_related_bill": 0.0,
            "nearby_hearing_count": 0.0,
            "nearby_bill_count": 0.0,
        }

    return {
        "min_hearing_distance": float(ctx.min_hearing_distance_days or 999),
        "min_bill_distance": float(ctx.min_bill_distance_days or 999),
        "committee_sector_alignment": 1.0 if ctx.committee_sector_alignment else 0.0,
        "timing_suspicion_score": ctx.timing_suspicion_score,
        "has_related_bill": 1.0 if ctx.nearby_bills else 0.0,
        "nearby_hearing_count": float(len(ctx.nearby_hearings)),
        "nearby_bill_count": float(len(ctx.nearby_bills)),
    }


async def sentiment_features(
    session: AsyncSession,
    member_bioguide_id: str | None,
    ticker: str | None,
    trade_date: date,
) -> dict[str, float]:
    """Compute sentiment features from media content and NLP analysis."""
    defaults = {
        "avg_sentiment_7d": 0.0,
        "avg_sentiment_30d": 0.0,
        "sentiment_momentum": 0.0,
        "media_mention_count_7d": 0.0,
        "media_mention_count_30d": 0.0,
    }

    if not member_bioguide_id and not ticker:
        return defaults

    date_30d = trade_date - timedelta(days=30)
    date_7d = trade_date - timedelta(days=7)

    # Build base query filtering by member or ticker mentions
    conditions = [
        MediaContent.published_date >= date_30d,
        MediaContent.published_date < trade_date,
    ]

    # Query media content count and average sentiment for 30d window
    query_30d = (
        select(
            func.count(MediaContent.id).label("count"),
            func.avg(SentimentAnalysis.sentiment_score).label("avg_score"),
        )
        .outerjoin(SentimentAnalysis, SentimentAnalysis.media_content_id == MediaContent.id)
        .where(and_(*conditions))
    )

    query_7d = (
        select(
            func.count(MediaContent.id).label("count"),
            func.avg(SentimentAnalysis.sentiment_score).label("avg_score"),
        )
        .outerjoin(SentimentAnalysis, SentimentAnalysis.media_content_id == MediaContent.id)
        .where(
            and_(
                MediaContent.published_date >= date_7d,
                MediaContent.published_date < trade_date,
            )
        )
    )

    result_30d = await session.execute(query_30d)
    row_30d = result_30d.one()
    result_7d = await session.execute(query_7d)
    row_7d = result_7d.one()

    avg_7d = float(row_7d.avg_score or 0)
    avg_30d = float(row_30d.avg_score or 0)

    return {
        "avg_sentiment_7d": avg_7d,
        "avg_sentiment_30d": avg_30d,
        "sentiment_momentum": avg_7d - avg_30d,
        "media_mention_count_7d": float(row_7d.count or 0),
        "media_mention_count_30d": float(row_30d.count or 0),
    }


def network_features_from_dict(network_data: dict[str, Any]) -> dict[str, float]:
    """Compute network features from pre-queried graph data.

    Args:
        network_data: Dict from Neo4j queries with connection info.
    """
    return {
        "lobbying_connection_count": float(network_data.get("lobbying_connections", 0)),
        "campaign_donor_connection": 1.0 if network_data.get("has_campaign_donor") else 0.0,
        "network_degree": float(network_data.get("degree", 0)),
        "has_lobbying_triangle": 1.0 if network_data.get("has_suspicious_triangle") else 0.0,
    }


async def build_feature_vector(
    session: AsyncSession,
    trade_id: int,
    network_data: dict[str, Any] | None = None,
) -> dict[str, float] | None:
    """Build complete feature vector for a single trade.

    Orchestrates all feature groups and returns a flat dict.
    Returns None if the trade is not found.
    """
    trade_obj = await session.get(TradeDisclosure, trade_id)
    if not trade_obj:
        return None

    trade_dict = {
        "amount_range_low": trade_obj.amount_range_low,
        "amount_range_high": trade_obj.amount_range_high,
        "transaction_type": trade_obj.transaction_type,
        "filer_type": trade_obj.filer_type,
        "transaction_date": trade_obj.transaction_date,
        "disclosure_date": trade_obj.disclosure_date,
    }

    # Get member info
    member_dict: dict[str, Any] = {}
    if trade_obj.member_bioguide_id:
        member_obj = await session.get(CongressMember, trade_obj.member_bioguide_id)
        if member_obj:
            # Count committees
            committee_result = await session.execute(
                select(func.count(CommitteeAssignment.id)).where(
                    CommitteeAssignment.member_bioguide_id == member_obj.bioguide_id
                )
            )
            committee_count = committee_result.scalar() or 0

            member_dict = {
                "party": member_obj.party,
                "chamber": member_obj.chamber,
                "nominate_dim1": member_obj.nominate_dim1,
                "nominate_dim2": member_obj.nominate_dim2,
                "first_elected": member_obj.first_elected,
                "committee_count": committee_count,
            }

    # Compute all feature groups
    features: dict[str, float] = {}
    features.update(trade_features(trade_dict))
    features.update(member_features(member_dict))
    features.update(await market_features(session, trade_obj.ticker, trade_obj.transaction_date))
    features.update(await legislative_features(session, trade_id))
    features.update(
        await sentiment_features(
            session,
            trade_obj.member_bioguide_id,
            trade_obj.ticker,
            trade_obj.transaction_date,
        )
    )

    if network_data:
        features.update(network_features_from_dict(network_data))
    else:
        features.update(network_features_from_dict({}))

    return features


def _std(values: list[float]) -> float:
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _compute_rsi(prices: list[float], period: int = 14) -> float:
    """Compute RSI from a list of prices (most recent first)."""
    if len(prices) < 2:
        return 50.0

    gains = []
    losses = []
    for i in range(min(period, len(prices) - 1)):
        change = prices[i] - prices[i + 1]  # most recent first
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains) / max(len(gains), 1)
    avg_loss = sum(losses) / max(len(losses), 1)

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
