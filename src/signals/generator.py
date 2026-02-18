"""Signal generators that create trading signals from ML predictions and data fusion.

Five signal types:
1. trade_follow - Follow high-confidence ML predictions
2. anomaly_alert - Flag unusual trading patterns
3. network_signal - Suspicious lobbying/donation connections
4. sentiment_divergence - Trade direction contradicts sentiment
5. insider_cluster - Multiple members trading same stock
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.media import MediaContent, SentimentAnalysis
from src.models.ml import TradePrediction
from src.models.signal import Signal
from src.models.trade import TradeDisclosure
from src.signals.scorer import score_signal

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generates trading signals from multiple data sources."""

    async def generate_all(self, session: AsyncSession) -> list[dict[str, Any]]:
        """Run all signal generators and return created signals."""
        all_signals: list[dict[str, Any]] = []

        generators = [
            self.generate_trade_follow_signals,
            self.generate_anomaly_alerts,
            self.generate_sentiment_divergence_signals,
            self.generate_insider_cluster_signals,
        ]

        for gen_func in generators:
            try:
                signals = await gen_func(session)
                all_signals.extend(signals)
            except Exception:
                logger.exception("Signal generator %s failed", gen_func.__name__)

        logger.info("Generated %d total signals", len(all_signals))
        return all_signals

    async def generate_trade_follow_signals(
        self, session: AsyncSession
    ) -> list[dict[str, Any]]:
        """Generate follow signals from high-confidence ML predictions.

        Looks for predictions with confidence > 0.7 in the last 7 days
        that don't already have a signal.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)

        result = await session.execute(
            select(TradePrediction, TradeDisclosure)
            .join(TradeDisclosure, TradePrediction.trade_id == TradeDisclosure.id)
            .where(
                and_(
                    TradePrediction.prediction_type == "profitability",
                    TradePrediction.confidence > 0.7,
                    TradePrediction.created_at >= cutoff,
                )
            )
        )

        signals: list[dict[str, Any]] = []
        for prediction, trade in result.all():
            # Check if signal already exists
            existing = await session.execute(
                select(Signal.id).where(
                    and_(
                        Signal.signal_type == "trade_follow",
                        Signal.ticker == trade.ticker,
                        Signal.member_bioguide_id == trade.member_bioguide_id,
                        Signal.created_at >= cutoff,
                    )
                )
            )
            if existing.scalar_one_or_none():
                continue

            is_buy = prediction.predicted_label == "profitable"
            direction = "bullish" if is_buy else "bearish"

            signal_data = {
                "signal_type": "trade_follow",
                "member_bioguide_id": trade.member_bioguide_id,
                "ticker": trade.ticker,
                "direction": direction,
                "confidence": float(prediction.confidence or 0),
                "evidence": {
                    "trade_id": trade.id,
                    "prediction_id": prediction.id,
                    "member_name": trade.member_name,
                    "transaction_type": trade.transaction_type,
                    "predicted_value": float(prediction.predicted_value or 0),
                },
                "disclosure_lag_days": (
                    (trade.disclosure_date - trade.transaction_date).days
                    if trade.disclosure_date and trade.transaction_date
                    else None
                ),
            }

            scored = score_signal(signal_data)
            signal = Signal(
                signal_type="trade_follow",
                member_bioguide_id=trade.member_bioguide_id,
                ticker=trade.ticker,
                direction=direction,
                strength=scored["strength"],
                confidence=scored["confidence"],
                evidence=signal_data["evidence"],
                expires_at=datetime.now(timezone.utc) + timedelta(days=7),
                is_active=True,
            )
            session.add(signal)
            signals.append(signal_data)

        if signals:
            await session.commit()
            logger.info("Generated %d trade_follow signals", len(signals))

        return signals

    async def generate_anomaly_alerts(
        self, session: AsyncSession
    ) -> list[dict[str, Any]]:
        """Generate anomaly alerts from ML anomaly detection scores."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)

        result = await session.execute(
            select(TradePrediction, TradeDisclosure)
            .join(TradeDisclosure, TradePrediction.trade_id == TradeDisclosure.id)
            .where(
                and_(
                    TradePrediction.prediction_type == "anomaly",
                    TradePrediction.predicted_label == "anomalous",
                    TradePrediction.created_at >= cutoff,
                )
            )
        )

        signals: list[dict[str, Any]] = []
        for prediction, trade in result.all():
            existing = await session.execute(
                select(Signal.id).where(
                    and_(
                        Signal.signal_type == "anomaly_alert",
                        Signal.ticker == trade.ticker,
                        Signal.member_bioguide_id == trade.member_bioguide_id,
                        Signal.created_at >= cutoff,
                    )
                )
            )
            if existing.scalar_one_or_none():
                continue

            signal_data = {
                "signal_type": "anomaly_alert",
                "member_bioguide_id": trade.member_bioguide_id,
                "ticker": trade.ticker,
                "direction": "neutral",
                "confidence": float(prediction.confidence or 0),
                "evidence": {
                    "trade_id": trade.id,
                    "anomaly_score": float(prediction.predicted_value or 0),
                    "member_name": trade.member_name,
                    "transaction_type": trade.transaction_type,
                    "amount_range": f"${trade.amount_range_low}-${trade.amount_range_high}",
                },
            }

            scored = score_signal(signal_data)
            signal = Signal(
                signal_type="anomaly_alert",
                member_bioguide_id=trade.member_bioguide_id,
                ticker=trade.ticker,
                direction="neutral",
                strength=scored["strength"],
                confidence=scored["confidence"],
                evidence=signal_data["evidence"],
                expires_at=datetime.now(timezone.utc) + timedelta(days=21),
                is_active=True,
            )
            session.add(signal)
            signals.append(signal_data)

        if signals:
            await session.commit()
            logger.info("Generated %d anomaly_alert signals", len(signals))

        return signals

    async def generate_sentiment_divergence_signals(
        self, session: AsyncSession
    ) -> list[dict[str, Any]]:
        """Generate signals when trade direction contradicts media sentiment.

        Bullish signal: member buys while sentiment is negative (contrarian)
        Bearish signal: member sells while sentiment is very positive (insider)
        """
        cutoff_date = date.today() - timedelta(days=7)

        # Get recent trades
        result = await session.execute(
            select(TradeDisclosure)
            .where(
                and_(
                    TradeDisclosure.transaction_date >= cutoff_date,
                    TradeDisclosure.ticker.isnot(None),
                    TradeDisclosure.member_bioguide_id.isnot(None),
                )
            )
        )
        trades = result.scalars().all()

        signals: list[dict[str, Any]] = []
        for trade in trades:
            # Get avg sentiment for this member in last 30 days
            sent_result = await session.execute(
                select(func.avg(SentimentAnalysis.sentiment_score))
                .join(MediaContent, SentimentAnalysis.media_content_id == MediaContent.id)
                .where(
                    and_(
                        MediaContent.member_bioguide_ids.contains(
                            [trade.member_bioguide_id]
                        ),
                        MediaContent.published_date >= cutoff_date - timedelta(days=23),
                    )
                )
            )
            avg_sentiment = sent_result.scalar()
            if avg_sentiment is None:
                continue

            avg_sentiment = float(avg_sentiment)
            is_purchase = trade.transaction_type == "purchase"

            # Divergence: buy when sentiment negative, or sell when sentiment positive
            divergence = False
            direction = "neutral"
            if is_purchase and avg_sentiment < -0.2:
                divergence = True
                direction = "bullish"  # contrarian buy
            elif not is_purchase and avg_sentiment > 0.2:
                divergence = True
                direction = "bearish"  # insider sell

            if not divergence:
                continue

            # Check for existing signal
            existing = await session.execute(
                select(Signal.id).where(
                    and_(
                        Signal.signal_type == "sentiment_divergence",
                        Signal.ticker == trade.ticker,
                        Signal.member_bioguide_id == trade.member_bioguide_id,
                        Signal.created_at >= datetime.now(timezone.utc) - timedelta(days=7),
                    )
                )
            )
            if existing.scalar_one_or_none():
                continue

            signal_data = {
                "signal_type": "sentiment_divergence",
                "member_bioguide_id": trade.member_bioguide_id,
                "ticker": trade.ticker,
                "direction": direction,
                "confidence": min(abs(avg_sentiment) * 2, 1.0),
                "evidence": {
                    "trade_id": trade.id,
                    "member_name": trade.member_name,
                    "transaction_type": trade.transaction_type,
                    "avg_sentiment_30d": avg_sentiment,
                    "divergence_type": "contrarian_buy" if is_purchase else "insider_sell",
                },
            }

            scored = score_signal(signal_data)
            signal = Signal(
                signal_type="sentiment_divergence",
                member_bioguide_id=trade.member_bioguide_id,
                ticker=trade.ticker,
                direction=direction,
                strength=scored["strength"],
                confidence=scored["confidence"],
                evidence=signal_data["evidence"],
                expires_at=datetime.now(timezone.utc) + timedelta(days=14),
                is_active=True,
            )
            session.add(signal)
            signals.append(signal_data)

        if signals:
            await session.commit()
            logger.info("Generated %d sentiment_divergence signals", len(signals))

        return signals

    async def generate_insider_cluster_signals(
        self, session: AsyncSession
    ) -> list[dict[str, Any]]:
        """Detect multiple members trading the same stock in the same direction.

        Looks for 3+ members trading the same ticker in the same direction
        within a 7-day window.
        """
        cutoff_date = date.today() - timedelta(days=7)

        # Find tickers with multiple members trading
        result = await session.execute(
            select(
                TradeDisclosure.ticker,
                TradeDisclosure.transaction_type,
                func.count(func.distinct(TradeDisclosure.member_bioguide_id)).label("member_count"),
            )
            .where(
                and_(
                    TradeDisclosure.transaction_date >= cutoff_date,
                    TradeDisclosure.ticker.isnot(None),
                    TradeDisclosure.member_bioguide_id.isnot(None),
                )
            )
            .group_by(TradeDisclosure.ticker, TradeDisclosure.transaction_type)
            .having(func.count(func.distinct(TradeDisclosure.member_bioguide_id)) >= 3)
        )
        clusters = result.all()

        signals: list[dict[str, Any]] = []
        for ticker, tx_type, member_count in clusters:
            # Check for existing signal
            existing = await session.execute(
                select(Signal.id).where(
                    and_(
                        Signal.signal_type == "insider_cluster",
                        Signal.ticker == ticker,
                        Signal.created_at >= datetime.now(timezone.utc) - timedelta(days=7),
                    )
                )
            )
            if existing.scalar_one_or_none():
                continue

            is_purchase = tx_type == "purchase"
            direction = "bullish" if is_purchase else "bearish"

            # Get member names
            members_result = await session.execute(
                select(
                    TradeDisclosure.member_name,
                    TradeDisclosure.member_bioguide_id,
                )
                .where(
                    and_(
                        TradeDisclosure.ticker == ticker,
                        TradeDisclosure.transaction_type == tx_type,
                        TradeDisclosure.transaction_date >= cutoff_date,
                    )
                )
                .distinct()
            )
            members = [(r[0], r[1]) for r in members_result.all()]

            signal_data = {
                "signal_type": "insider_cluster",
                "member_bioguide_id": None,
                "ticker": ticker,
                "direction": direction,
                "confidence": min(member_count / 10, 1.0),
                "evidence": {
                    "cluster_size": member_count,
                    "transaction_type": tx_type,
                    "members": [{"name": m[0], "bioguide_id": m[1]} for m in members[:10]],
                },
            }

            scored = score_signal(signal_data)
            signal = Signal(
                signal_type="insider_cluster",
                ticker=ticker,
                direction=direction,
                strength=scored["strength"],
                confidence=scored["confidence"],
                evidence=signal_data["evidence"],
                expires_at=datetime.now(timezone.utc) + timedelta(days=10),
                is_active=True,
            )
            session.add(signal)
            signals.append(signal_data)

        if signals:
            await session.commit()
            logger.info("Generated %d insider_cluster signals", len(signals))

        return signals
