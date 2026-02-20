"""Trade Screener — ranks bullish signals into buy recommendations with position sizing."""

from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.financial import StockDaily
from src.models.ml import TradePrediction
from src.models.signal import Signal
from src.models.trade import TradeDisclosure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions (no DB, easily testable)
# ---------------------------------------------------------------------------


def compute_composite_score(
    avg_strength: float,
    ml_confidence: float,
    best_win_rate: float,
    freshness: float,
    count_bonus: float,
) -> float:
    """Weighted composite score for ticker ranking.

    Weights: strength 30%, ml 25%, win_rate 20%, freshness 15%, count 10%.
    """
    score = (
        0.30 * avg_strength
        + 0.25 * ml_confidence
        + 0.20 * best_win_rate
        + 0.15 * freshness
        + 0.10 * count_bonus
    )
    return max(0.0, min(1.0, score))


def compute_freshness(created_at: datetime) -> float:
    """Linear decay from 1.0 (brand new) to 0.0 (7 days old)."""
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_hours = (now - created_at).total_seconds() / 3600
    return max(0.0, min(1.0, 1.0 - age_hours / 168.0))


def compute_count_bonus(signal_count: int) -> float:
    """Corroboration bonus: 1 signal = 0.2, 5+ signals = 1.0."""
    return min(signal_count / 5.0, 1.0)


def compute_allocations(
    recommendations: list[dict],
    portfolio_size: float,
) -> list[dict]:
    """Position sizing via score-weighted allocation with 10%-40% clamp.

    Mutates and returns the input list with allocation_pct, suggested_amount,
    and suggested_shares populated.
    """
    if not recommendations:
        return recommendations

    total_score = sum(r["composite_score"] for r in recommendations)
    if total_score == 0:
        # Equal allocation fallback
        equal_pct = 1.0 / len(recommendations)
        for r in recommendations:
            r["allocation_pct"] = equal_pct
    else:
        for r in recommendations:
            r["allocation_pct"] = r["composite_score"] / total_score

    # Clamp to [10%, 40%]
    for r in recommendations:
        r["allocation_pct"] = max(0.10, min(0.40, r["allocation_pct"]))

    # Re-normalize to sum to 1.0
    total_alloc = sum(r["allocation_pct"] for r in recommendations)
    if total_alloc > 0:
        for r in recommendations:
            r["allocation_pct"] = r["allocation_pct"] / total_alloc

    # Compute dollar amounts and shares
    for r in recommendations:
        r["suggested_amount"] = round(portfolio_size * r["allocation_pct"], 2)
        price = r.get("current_price")
        if price and price > 0:
            r["suggested_shares"] = math.floor(r["suggested_amount"] / price)
        else:
            r["suggested_shares"] = 0

    return recommendations


# ---------------------------------------------------------------------------
# TradeScreener (async, DB-backed)
# ---------------------------------------------------------------------------


class TradeScreener:
    """Fetches active bullish signals and produces ranked buy recommendations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_recommendations(
        self,
        portfolio_size: float = 3000.0,
        top_n: int = 5,
        min_signal_strength: float = 0.3,
    ) -> list[dict]:
        """Main entry: fetch signals, score, rank, size positions."""
        now = datetime.now(timezone.utc)

        # 1. Fetch active bullish signals
        query = select(Signal).where(
            Signal.is_active.is_(True),
            Signal.direction == "bullish",
            Signal.strength >= min_signal_strength,
            Signal.ticker.isnot(None),
        ).where(
            (Signal.expires_at > now) | (Signal.expires_at.is_(None))
        )
        result = await self.session.execute(query)
        signals = result.scalars().all()

        if not signals:
            return []

        # 2. Group by ticker
        ticker_signals: dict[str, list] = defaultdict(list)
        for s in signals:
            if s.ticker:
                ticker_signals[s.ticker].append(s)

        # 3. Score each ticker
        scored: list[dict] = []
        for ticker, sigs in ticker_signals.items():
            avg_strength = sum(float(s.strength) for s in sigs) / len(sigs)
            freshness = max(compute_freshness(s.created_at) for s in sigs)
            count_bonus = compute_count_bonus(len(sigs))

            ml_confidence = await self._get_ml_confidence(ticker, now)
            best_win_rate = await self._get_best_member_win_rate(ticker, sigs)
            current_price = await self._get_current_price(ticker)
            risk = await self._get_risk_metrics(ticker)
            evidence = self._collect_evidence(sigs)

            composite = compute_composite_score(
                avg_strength, ml_confidence, best_win_rate, freshness, count_bonus
            )

            scored.append({
                "ticker": ticker,
                "action": "BUY",
                "composite_score": round(composite, 4),
                "avg_signal_strength": round(avg_strength, 4),
                "ml_confidence": round(ml_confidence, 4),
                "best_member_win_rate": round(best_win_rate, 4),
                "freshness_score": round(freshness, 4),
                "corroboration_score": round(count_bonus, 4),
                "current_price": float(current_price) if current_price else None,
                "evidence": evidence,
                "risk": risk,
            })

        # 4. Rank and take top_n
        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        top = scored[:top_n]

        # 5. Position sizing
        top = compute_allocations(top, portfolio_size)
        return top

    async def _get_ml_confidence(self, ticker: str, now: datetime) -> float:
        """Max prediction confidence for this ticker in last 30 days (profitable label)."""
        from datetime import timedelta

        cutoff = now - timedelta(days=30)
        query = (
            select(func.max(TradePrediction.confidence))
            .join(TradeDisclosure, TradePrediction.trade_id == TradeDisclosure.id)
            .where(
                TradeDisclosure.ticker == ticker,
                TradePrediction.predicted_label.in_(["buy", "profitable"]),
                TradePrediction.created_at >= cutoff,
            )
        )
        result = await self.session.execute(query)
        val = result.scalar()
        return float(val) if val else 0.5

    async def _get_best_member_win_rate(
        self, ticker: str, sigs: list
    ) -> float:
        """Best win rate among members who signaled this ticker."""
        member_ids = list({
            s.member_bioguide_id for s in sigs if s.member_bioguide_id
        })
        if not member_ids:
            return 0.5

        # Count trades with positive actual_return_5d for these members
        total_q = (
            select(func.count(TradePrediction.id))
            .join(TradeDisclosure, TradePrediction.trade_id == TradeDisclosure.id)
            .where(
                TradeDisclosure.member_bioguide_id.in_(member_ids),
                TradePrediction.actual_return_5d.isnot(None),
            )
        )
        total_result = await self.session.execute(total_q)
        total = total_result.scalar() or 0

        if total < 5:
            return 0.5  # not enough data

        wins_q = (
            select(func.count(TradePrediction.id))
            .join(TradeDisclosure, TradePrediction.trade_id == TradeDisclosure.id)
            .where(
                TradeDisclosure.member_bioguide_id.in_(member_ids),
                TradePrediction.actual_return_5d > 0,
            )
        )
        wins_result = await self.session.execute(wins_q)
        wins = wins_result.scalar() or 0

        return wins / total

    async def _get_current_price(self, ticker: str) -> float | None:
        """Most recent close from StockDaily. Falls back to yfinance."""
        query = (
            select(StockDaily.close)
            .where(StockDaily.ticker == ticker)
            .order_by(StockDaily.date.desc())
            .limit(1)
        )
        result = await self.session.execute(query)
        val = result.scalar()
        if val:
            return float(val)

        # Fallback: yfinance
        try:
            price = await asyncio.to_thread(self._yfinance_price, ticker)
            return price
        except Exception:
            logger.warning("Could not fetch price for %s", ticker)
            return None

    @staticmethod
    def _yfinance_price(ticker: str) -> float | None:
        """Synchronous yfinance price fetch (run in thread)."""
        try:
            import yfinance as yf
            data = yf.Ticker(ticker).history(period="5d")
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except Exception:
            pass
        return None

    async def _get_risk_metrics(self, ticker: str) -> dict:
        """Compute volatility_21d, RSI_14, price_change_5d from StockDaily."""
        query = (
            select(StockDaily.close, StockDaily.date)
            .where(StockDaily.ticker == ticker)
            .order_by(StockDaily.date.desc())
            .limit(30)
        )
        result = await self.session.execute(query)
        rows = result.all()

        if len(rows) < 5:
            return {"volatility_21d": None, "rsi_14": None, "price_change_5d": None}

        closes = [float(r[0]) for r in reversed(rows) if r[0] is not None]

        # Price change 5d
        if len(closes) >= 5:
            price_change_5d = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] else None
        else:
            price_change_5d = None

        # Daily returns for volatility
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] != 0:
                returns.append((closes[i] - closes[i - 1]) / closes[i - 1])

        # Volatility (21d annualized std)
        if len(returns) >= 5:
            mean_ret = sum(returns) / len(returns)
            var = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
            volatility_21d = var ** 0.5 * (252 ** 0.5)  # annualized
        else:
            volatility_21d = None

        # RSI 14
        rsi_14 = self._compute_rsi(returns, 14) if len(returns) >= 14 else None

        return {
            "volatility_21d": round(volatility_21d, 4) if volatility_21d is not None else None,
            "rsi_14": round(rsi_14, 2) if rsi_14 is not None else None,
            "price_change_5d": round(price_change_5d, 4) if price_change_5d is not None else None,
        }

    @staticmethod
    def _compute_rsi(returns: list[float], period: int = 14) -> float | None:
        """Compute RSI from a list of daily returns."""
        if len(returns) < period:
            return None

        recent = returns[-period:]
        gains = [r for r in recent if r > 0]
        losses = [-r for r in recent if r < 0]

        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _collect_evidence(sigs: list) -> dict:
        """Extract member names and signal types from signals."""
        member_names: set[str] = set()
        signal_types: set[str] = set()
        strongest_type: str | None = None
        max_strength = 0.0

        for s in sigs:
            signal_types.add(s.signal_type)
            if float(s.strength) > max_strength:
                max_strength = float(s.strength)
                strongest_type = s.signal_type

            evidence = s.evidence or {}
            # Extract member names from evidence JSONB
            sources = evidence.get("sources", [])
            for src in sources:
                if isinstance(src, dict) and src.get("detail"):
                    detail = src["detail"]
                    if isinstance(detail, str) and "member" in detail.lower():
                        member_names.add(detail)

            # Also try member_bioguide_id
            if s.member_bioguide_id:
                member_names.add(s.member_bioguide_id)

        return {
            "member_names": sorted(member_names),
            "signal_count": len(sigs),
            "signal_types": sorted(signal_types),
            "strongest_type": strongest_type,
        }
