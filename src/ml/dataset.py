"""Dataset builder and temporal splitter for ML training.

Builds labeled datasets from historical trades with walk-forward
temporal cross-validation to prevent future data leakage.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.ml.features import build_feature_vector
from src.models.financial import StockDaily
from src.models.trade import TradeDisclosure

logger = logging.getLogger(__name__)


class TemporalSplitter:
    """Walk-forward temporal cross-validation splitter.

    Ensures no future data leakage by splitting strictly on dates.
    Each fold trains on `train_months` of data and tests on the
    following `test_months`.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_months: int = 12,
        test_months: int = 3,
    ) -> None:
        self.n_splits = n_splits
        self.train_months = train_months
        self.test_months = test_months

    def split(
        self, df: pd.DataFrame, date_column: str = "transaction_date"
    ) -> list[tuple[list[int], list[int]]]:
        """Generate train/test index splits based on temporal ordering.

        Returns list of (train_indices, test_indices) tuples.
        """
        if df.empty or date_column not in df.columns:
            return []

        df = df.sort_values(date_column).reset_index(drop=True)
        dates = pd.to_datetime(df[date_column])
        min_date = dates.min()
        max_date = dates.max()

        total_months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month)
        needed = self.train_months + self.test_months

        if total_months < needed:
            logger.warning(
                "Not enough data for temporal split: %d months available, %d needed",
                total_months,
                needed,
            )
            return []

        splits = []
        for i in range(self.n_splits):
            # Slide the window forward
            train_start = min_date + pd.DateOffset(months=i * self.test_months)
            train_end = train_start + pd.DateOffset(months=self.train_months)
            test_end = train_end + pd.DateOffset(months=self.test_months)

            if test_end > max_date + pd.DateOffset(days=1):
                break

            train_mask = (dates >= train_start) & (dates < train_end)
            test_mask = (dates >= train_end) & (dates < test_end)

            train_idx = df.index[train_mask].tolist()
            test_idx = df.index[test_mask].tolist()

            if train_idx and test_idx:
                splits.append((train_idx, test_idx))

        return splits


class DatasetBuilder:
    """Builds labeled datasets from historical trades for ML training."""

    async def build_trade_dataset(
        self,
        session: AsyncSession,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 10000,
    ) -> pd.DataFrame:
        """Build feature matrix with labels from historical trades.

        Labels:
        - actual_return_5d: 5-trading-day return after trade
        - actual_return_21d: 21-trading-day return after trade
        - profitable_5d: 1 if trade was profitable at 5d horizon
        """
        conditions = [TradeDisclosure.ticker.isnot(None)]
        if start_date:
            conditions.append(TradeDisclosure.transaction_date >= start_date)
        if end_date:
            conditions.append(TradeDisclosure.transaction_date <= end_date)

        result = await session.execute(
            select(TradeDisclosure)
            .where(and_(*conditions))
            .order_by(TradeDisclosure.transaction_date)
            .limit(limit)
        )
        trades = result.scalars().all()

        if not trades:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for trade in trades:
            features = await build_feature_vector(session, trade.id)
            if features is None:
                continue

            # Compute actual returns as labels
            return_5d = await self._compute_return(
                session, trade.ticker, trade.transaction_date, 5
            )
            return_21d = await self._compute_return(
                session, trade.ticker, trade.transaction_date, 21
            )

            # Profitable = positive return for purchase, negative for sale
            is_purchase = trade.transaction_type == "purchase"
            profitable_5d = None
            if return_5d is not None:
                profitable_5d = 1 if (return_5d > 0) == is_purchase else 0

            row = {
                "trade_id": trade.id,
                "transaction_date": trade.transaction_date,
                "ticker": trade.ticker,
                **features,
                "actual_return_5d": return_5d,
                "actual_return_21d": return_21d,
                "profitable_5d": profitable_5d,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    async def _compute_return(
        self,
        session: AsyncSession,
        ticker: str | None,
        trade_date: date,
        days: int,
    ) -> float | None:
        """Compute the return for a ticker over N trading days after trade_date."""
        if not ticker:
            return None

        # Get price on trade date
        base_result = await session.execute(
            select(StockDaily)
            .where(
                and_(
                    StockDaily.ticker == ticker,
                    StockDaily.date <= trade_date,
                )
            )
            .order_by(StockDaily.date.desc())
            .limit(1)
        )
        base_row = base_result.scalar_one_or_none()
        if not base_row or not base_row.adj_close:
            return None

        # Get price N calendar days later (approximate trading days)
        target_date = trade_date + timedelta(days=int(days * 1.5))
        future_result = await session.execute(
            select(StockDaily)
            .where(
                and_(
                    StockDaily.ticker == ticker,
                    StockDaily.date > trade_date,
                    StockDaily.date <= target_date,
                )
            )
            .order_by(StockDaily.date)
        )
        future_rows = future_result.scalars().all()

        if len(future_rows) < days:
            # Use last available price if not enough data
            if future_rows:
                target_row = future_rows[-1]
            else:
                return None
        else:
            target_row = future_rows[days - 1]

        if not target_row.adj_close:
            return None

        base_price = float(base_row.adj_close)
        target_price = float(target_row.adj_close)

        if base_price == 0:
            return None

        return (target_price - base_price) / base_price
