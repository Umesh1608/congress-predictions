"""Collector for Financial Modeling Prep congressional trading API."""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from src.config import settings
from src.ingestion.base import BaseCollector, RateLimiter

logger = logging.getLogger(__name__)

FMP_BASE_URL = "https://financialmodelingprep.com/stable"


class FMPHouseCollector(BaseCollector):
    source_name = "fmp_house"
    rate_limiter = RateLimiter(max_calls=5, period_seconds=1.0)

    async def collect(self) -> list[dict[str, Any]]:
        if not settings.fmp_api_key:
            logger.warning("FMP API key not configured, skipping")
            return []
        url = f"{FMP_BASE_URL}/house-disclosure"
        data = await self.fetch_json(url, params={"apikey": settings.fmp_api_key})
        return data if isinstance(data, list) else []

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        return _transform_fmp_record(raw, chamber="house", source=self.source_name)


class FMPSenateCollector(BaseCollector):
    source_name = "fmp_senate"
    rate_limiter = RateLimiter(max_calls=5, period_seconds=1.0)

    async def collect(self) -> list[dict[str, Any]]:
        if not settings.fmp_api_key:
            logger.warning("FMP API key not configured, skipping")
            return []
        url = f"{FMP_BASE_URL}/senate-trading"
        data = await self.fetch_json(url, params={"apikey": settings.fmp_api_key})
        return data if isinstance(data, list) else []

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        return _transform_fmp_record(raw, chamber="senate", source=self.source_name)


def _transform_fmp_record(
    raw: dict[str, Any], chamber: str, source: str
) -> dict[str, Any] | None:
    ticker = (raw.get("symbol") or raw.get("ticker") or "").strip()
    if not ticker or ticker == "--":
        ticker = None

    transaction_date = _parse_date(raw.get("transactionDate") or raw.get("transaction_date", ""))
    if transaction_date is None:
        return None

    disclosure_date = _parse_date(raw.get("disclosureDate") or raw.get("disclosure_date", ""))

    tx_type = (raw.get("type") or raw.get("transactionType") or "").lower().strip()
    if "purchase" in tx_type or "buy" in tx_type:
        tx_type = "purchase"
    elif "sale" in tx_type or "sell" in tx_type:
        tx_type = "sale"
    elif "exchange" in tx_type:
        tx_type = "exchange"

    amount_low = _parse_decimal(raw.get("amount") or raw.get("amountFrom"))
    amount_high = _parse_decimal(raw.get("amountTo") or raw.get("amount"))

    owner = (raw.get("owner") or "").lower()
    if "spouse" in owner:
        filer_type = "spouse"
    elif "dependent" in owner or "child" in owner:
        filer_type = "dependent"
    elif "joint" in owner:
        filer_type = "joint"
    else:
        filer_type = "member"

    member_name = (
        raw.get("representative")
        or raw.get("senator")
        or raw.get("firstName", "") + " " + raw.get("lastName", "")
    ).strip()

    if not member_name:
        return None

    return {
        "member_name": member_name,
        "filer_type": filer_type,
        "ticker": ticker,
        "asset_name": raw.get("assetDescription", raw.get("asset_description", "")).strip(),
        "asset_type": raw.get("assetType", raw.get("asset_type")),
        "transaction_type": tx_type,
        "transaction_date": transaction_date,
        "disclosure_date": disclosure_date,
        "amount_range_low": amount_low,
        "amount_range_high": amount_high,
        "chamber": chamber,
        "source": source,
        "filing_url": raw.get("link") or raw.get("ptrLink"),
        "raw_data": raw,
    }


def _parse_date(date_str: str) -> date | None:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _parse_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        cleaned = str(value).replace("$", "").replace(",", "").strip()
        return Decimal(cleaned) if cleaned else None
    except (InvalidOperation, ValueError):
        return None
