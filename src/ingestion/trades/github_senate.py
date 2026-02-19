"""Collector for Senate trade data from GitHub (jeremiak's CSV dataset).

Source: https://github.com/jeremiak/us-senate-financial-disclosure-data
Data range: 2012 through Jan 2024
Format: CSV with columns: report-id, report-title, filer, filed-date,
    filed-time, owner, asset-name, ticker, transaction-type,
    transaction-date, amount, asset-type, type, comment

This is a free fallback when the Senate Stock Watcher S3 endpoint is down.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from src.ingestion.base import BaseCollector
from src.ingestion.trades.house_watcher import AMOUNT_RANGES

logger = logging.getLogger(__name__)

GITHUB_SENATE_CSV_URL = (
    "https://raw.githubusercontent.com/jeremiak/"
    "us-senate-financial-disclosure-data/master/output/transactions.csv"
)


class GitHubSenateCollector(BaseCollector):
    """Collect Senate trades from jeremiak's GitHub CSV dataset."""

    source_name = "github_senate"

    async def collect(self) -> list[dict[str, Any]]:
        response = await self.client.get(GITHUB_SENATE_CSV_URL)
        response.raise_for_status()

        text = response.text
        reader = csv.DictReader(io.StringIO(text))
        records = list(reader)

        logger.info("[%s] Parsed %d rows from CSV", self.source_name, len(records))
        return records

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        ticker = (raw.get("ticker") or "").strip()
        if not ticker or ticker == "--" or ticker == "N/A":
            ticker = None
        elif len(ticker) > 20:
            # Truncated or malformed ticker — skip it
            ticker = None

        transaction_date = self._parse_date(
            raw.get("transaction-date") or raw.get("date") or ""
        )
        if transaction_date is None:
            return None

        disclosure_date = self._parse_date(raw.get("filed-date") or "")

        # Transaction type normalization
        tx_type_raw = (
            raw.get("transaction-type") or raw.get("type") or ""
        ).lower().strip()
        if "purchase" in tx_type_raw or "buy" in tx_type_raw:
            tx_type = "purchase"
        elif "sale" in tx_type_raw:
            if "full" in tx_type_raw:
                tx_type = "sale_full"
            elif "partial" in tx_type_raw:
                tx_type = "sale_partial"
            else:
                tx_type = "sale"
        elif "exchange" in tx_type_raw:
            tx_type = "exchange"
        else:
            tx_type = tx_type_raw or "unknown"

        # Amount parsing
        amount_str = (raw.get("amount") or raw.get("amount-range") or "").strip()
        amount_low, amount_high = AMOUNT_RANGES.get(amount_str, (None, None))

        # Owner / filer type
        owner = (raw.get("owner") or "").lower().strip()
        if "spouse" in owner:
            filer_type = "spouse"
        elif "dependent" in owner or "child" in owner:
            filer_type = "dependent"
        elif "joint" in owner:
            filer_type = "joint"
        elif owner in ("self", ""):
            filer_type = "member"
        else:
            filer_type = "member"

        # Member name: from 'filer' column, may also appear in 'transactor'
        member_name = (
            raw.get("filer") or raw.get("transactor") or ""
        ).strip()
        if not member_name:
            return None

        return {
            "member_name": member_name,
            "filer_type": filer_type,
            "ticker": ticker,
            "asset_name": (raw.get("asset-name") or "").strip(),
            "asset_type": (raw.get("asset-type") or None),
            "transaction_type": tx_type,
            "transaction_date": transaction_date,
            "disclosure_date": disclosure_date,
            "amount_range_low": amount_low,
            "amount_range_high": amount_high,
            "chamber": "senate",
            "source": self.source_name,
            "filing_url": None,
            "raw_data": raw,
        }

    @staticmethod
    def _parse_date(date_str: str) -> date | None:
        if not date_str:
            return None
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y"):
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue
        return None
