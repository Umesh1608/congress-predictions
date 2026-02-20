"""Celery tasks for ML model training and prediction."""

from __future__ import annotations

import asyncio
import logging

from src.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="src.tasks.ml_tasks.train_all_models")
def train_all_models():
    """Retrain all ML models with latest data."""
    asyncio.run(_train_all())


async def _train_all():
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from src.config import settings
    from src.ml.training import ModelTrainer

    engine = create_async_engine(settings.database_url, pool_size=5)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as session:
            trainer = ModelTrainer(
                horizon="180d",
                n_folds=5,
                use_catboost=True,
            )
            results = await trainer.train_all(session)
            logger.info("Training complete: %s", results)
    finally:
        await engine.dispose()


@celery_app.task(name="src.tasks.ml_tasks.run_batch_predictions")
def run_batch_predictions():
    """Score unscored trades with trained ML models."""
    asyncio.run(_batch_predict())


async def _batch_predict():
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from src.config import settings
    from src.ml.predictor import PredictionService

    engine = create_async_engine(settings.database_url, pool_size=5)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as session:
            service = PredictionService()
            count = await service.batch_predict(session, limit=100)
            logger.info("Batch prediction complete: %d trades scored", count)
    finally:
        await engine.dispose()


@celery_app.task(name="src.tasks.ml_tasks.backfill_actual_returns")
def backfill_actual_returns():
    """Backfill actual 5d and 21d returns for predictions."""
    asyncio.run(_backfill_returns())


async def _backfill_returns():
    from datetime import timedelta

    from sqlalchemy import and_, select, update
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from src.config import settings
    from src.models.financial import StockDaily
    from src.models.ml import TradePrediction
    from src.models.trade import TradeDisclosure

    engine = create_async_engine(settings.database_url, pool_size=5)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as session:
            # Find predictions missing actual returns where enough time has passed
            result = await session.execute(
                select(TradePrediction, TradeDisclosure)
                .join(TradeDisclosure, TradePrediction.trade_id == TradeDisclosure.id)
                .where(
                    and_(
                        TradePrediction.actual_return_5d.is_(None),
                        TradeDisclosure.ticker.isnot(None),
                    )
                )
                .limit(500)
            )

            updated = 0
            for prediction, trade in result.all():
                if not trade.ticker or not trade.transaction_date:
                    continue

                # Try 5d return
                target_5d = trade.transaction_date + timedelta(days=8)
                price_result = await session.execute(
                    select(StockDaily)
                    .where(
                        and_(
                            StockDaily.ticker == trade.ticker,
                            StockDaily.date > trade.transaction_date,
                            StockDaily.date <= target_5d,
                        )
                    )
                    .order_by(StockDaily.date)
                )
                future_prices = price_result.scalars().all()

                base_result = await session.execute(
                    select(StockDaily)
                    .where(
                        and_(
                            StockDaily.ticker == trade.ticker,
                            StockDaily.date <= trade.transaction_date,
                        )
                    )
                    .order_by(StockDaily.date.desc())
                    .limit(1)
                )
                base_price_row = base_result.scalar_one_or_none()

                if not base_price_row or not base_price_row.adj_close:
                    continue

                base_price = float(base_price_row.adj_close)
                if base_price == 0:
                    continue

                updates = {}
                if len(future_prices) >= 5 and future_prices[4].adj_close:
                    updates["actual_return_5d"] = (
                        float(future_prices[4].adj_close) - base_price
                    ) / base_price

                # Try 21d return
                target_21d = trade.transaction_date + timedelta(days=32)
                price_result_21d = await session.execute(
                    select(StockDaily)
                    .where(
                        and_(
                            StockDaily.ticker == trade.ticker,
                            StockDaily.date > trade.transaction_date,
                            StockDaily.date <= target_21d,
                        )
                    )
                    .order_by(StockDaily.date)
                )
                future_21d = price_result_21d.scalars().all()

                if len(future_21d) >= 21 and future_21d[20].adj_close:
                    updates["actual_return_21d"] = (
                        float(future_21d[20].adj_close) - base_price
                    ) / base_price

                if updates:
                    await session.execute(
                        update(TradePrediction)
                        .where(TradePrediction.id == prediction.id)
                        .values(**updates)
                    )
                    updated += 1

            await session.commit()
            logger.info("Backfilled returns for %d predictions", updated)
    finally:
        await engine.dispose()
