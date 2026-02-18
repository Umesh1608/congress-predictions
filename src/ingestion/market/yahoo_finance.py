"""Market data collector using yfinance."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from decimal import Decimal

import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_stock_history(
    ticker: str, start_date: date | None = None, end_date: date | None = None
) -> list[dict]:
    """Fetch daily OHLCV data for a ticker.

    Returns list of dicts ready for StockDaily upsert.
    """
    if start_date is None:
        start_date = date.today() - timedelta(days=365)
    if end_date is None:
        end_date = date.today()

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date.isoformat(), end=end_date.isoformat())
    except Exception:
        logger.exception("Failed to fetch yfinance data for %s", ticker)
        return []

    if df.empty:
        logger.warning("No data returned for %s", ticker)
        return []

    records = []
    for idx, row in df.iterrows():
        records.append({
            "ticker": ticker,
            "date": idx.date(),
            "open": Decimal(str(round(row["Open"], 4))) if row.get("Open") else None,
            "high": Decimal(str(round(row["High"], 4))) if row.get("High") else None,
            "low": Decimal(str(round(row["Low"], 4))) if row.get("Low") else None,
            "close": Decimal(str(round(row["Close"], 4))) if row.get("Close") else None,
            "adj_close": Decimal(str(round(row["Close"], 4))) if row.get("Close") else None,
            "volume": int(row["Volume"]) if row.get("Volume") else None,
        })

    logger.info("Fetched %d daily records for %s", len(records), ticker)
    return records
