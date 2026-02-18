"""Collector for House Stock Watcher (free, S3-hosted JSON)."""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from src.ingestion.base import BaseCollector

logger = logging.getLogger(__name__)

# Amount range string to numeric bounds
AMOUNT_RANGES: dict[str, tuple[Decimal, Decimal]] = {
    "$1,001 - $15,000": (Decimal("1001"), Decimal("15000")),
    "$15,001 - $50,000": (Decimal("15001"), Decimal("50000")),
    "$50,001 - $100,000": (Decimal("50001"), Decimal("100000")),
    "$100,001 - $250,000": (Decimal("100001"), Decimal("250000")),
    "$250,001 - $500,000": (Decimal("250001"), Decimal("500000")),
    "$500,001 - $1,000,000": (Decimal("500001"), Decimal("1000000")),
    "$1,000,001 - $5,000,000": (Decimal("1000001"), Decimal("5000000")),
    "$5,000,001 - $25,000,000": (Decimal("5000001"), Decimal("25000000")),
    "$25,000,001 - $50,000,000": (Decimal("25000001"), Decimal("50000000")),
    "$50,000,001 +": (Decimal("50000001"), Decimal("50000001")),
}

HOUSE_WATCHER_URL = (
    "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
)


class HouseWatcherCollector(BaseCollector):
    source_name = "house_watcher"

    async def collect(self) -> list[dict[str, Any]]:
        data = await self.fetch_json(HOUSE_WATCHER_URL)
        if not isinstance(data, list):
            logger.error("Unexpected response format from House Watcher")
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
        if owner in ("joint", ""):
            filer_type = "member"
        elif "spouse" in owner:
            filer_type = "spouse"
        elif "dependent" in owner or "child" in owner:
            filer_type = "dependent"
        else:
            filer_type = "member"

        return {
            "member_name": raw.get("representative", "").strip(),
            "filer_type": filer_type,
            "ticker": ticker,
            "asset_name": raw.get("asset_description", "").strip(),
            "asset_type": raw.get("cap_gains_over_200_usd", None) and "Stock",
            "transaction_type": tx_type,
            "transaction_date": transaction_date,
            "disclosure_date": disclosure_date,
            "amount_range_low": amount_low,
            "amount_range_high": amount_high,
            "chamber": "house",
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
