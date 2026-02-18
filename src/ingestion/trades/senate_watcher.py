"""Collector for Senate Stock Watcher (free, S3-hosted JSON)."""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from src.ingestion.base import BaseCollector
from src.ingestion.trades.house_watcher import AMOUNT_RANGES

logger = logging.getLogger(__name__)

SENATE_WATCHER_URL = (
    "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"
)


class SenateWatcherCollector(BaseCollector):
    source_name = "senate_watcher"

    async def collect(self) -> list[dict[str, Any]]:
        data = await self.fetch_json(SENATE_WATCHER_URL)
        if not isinstance(data, list):
            logger.error("Unexpected response format from Senate Watcher")
            return []
        return data

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        ticker = raw.get("ticker", "").strip()
        if not ticker or ticker == "--" or ticker == "N/A":
            ticker = None

        transaction_date = self._parse_date(raw.get("transaction_date", ""))
        if transaction_date is None:
            return None

        disclosure_date = self._parse_date(raw.get("disclosure_date", ""))

        tx_type = (raw.get("type", "") or "").lower().strip()
        if "purchase" in tx_type:
            tx_type = "purchase"
        elif "sale" in tx_type:
            if "full" in tx_type:
                tx_type = "sale_full"
            elif "partial" in tx_type:
                tx_type = "sale_partial"
            else:
                tx_type = "sale"
        elif "exchange" in tx_type:
            tx_type = "exchange"

        amount_str = raw.get("amount", "")
        amount_low, amount_high = AMOUNT_RANGES.get(amount_str, (None, None))

        owner = (raw.get("owner", "") or "").lower().strip()
        if "spouse" in owner:
            filer_type = "spouse"
        elif "dependent" in owner or "child" in owner:
            filer_type = "dependent"
        elif "joint" in owner:
            filer_type = "joint"
        else:
            filer_type = "member"

        return {
            "member_name": (raw.get("senator", "") or raw.get("representative", "")).strip(),
            "filer_type": filer_type,
            "ticker": ticker,
            "asset_name": raw.get("asset_description", "").strip(),
            "asset_type": raw.get("asset_type"),
            "transaction_type": tx_type,
            "transaction_date": transaction_date,
            "disclosure_date": disclosure_date,
            "amount_range_low": amount_low,
            "amount_range_high": amount_high,
            "chamber": "senate",
            "source": self.source_name,
            "filing_url": raw.get("ptr_link"),
            "raw_data": raw,
        }

    @staticmethod
    def _parse_date(date_str: str) -> date | None:
        if not date_str:
            return None
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y"):
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue
        return None
