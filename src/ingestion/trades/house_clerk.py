"""Collector for House trade data scraped from disclosures-clerk.house.gov.

Scrapes PTR (Periodic Transaction Report) filings from the House clerk's
financial disclosure search, downloads PDFs, and extracts trade data.

This is a free fallback when the House Stock Watcher S3 endpoint is down.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from decimal import Decimal
from html.parser import HTMLParser
from io import BytesIO
from typing import Any

from src.ingestion.base import BaseCollector, RateLimiter
from src.ingestion.trades.house_watcher import AMOUNT_RANGES

logger = logging.getLogger(__name__)

HOUSE_CLERK_SEARCH_URL = (
    "https://disclosures-clerk.house.gov/FinancialDisclosure/ViewMemberSearchResult"
)
HOUSE_CLERK_BASE_URL = "https://disclosures-clerk.house.gov/"

# Rate limit to be respectful to government servers
_house_clerk_rate_limiter = RateLimiter(max_calls=2, period_seconds=1.0)


class _SearchResultParser(HTMLParser):
    """Parse the HTML table returned by the House clerk search."""

    def __init__(self) -> None:
        super().__init__()
        self.in_td = False
        self.current_cell = ""
        self.current_row: list[str] = []
        self.rows: list[list[str]] = []
        self.current_href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "td":
            self.in_td = True
            self.current_cell = ""
        elif tag == "a" and self.in_td:
            for name, val in attrs:
                if name == "href" and val:
                    self.current_href = val
        elif tag == "tr":
            self.current_row = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "td":
            self.in_td = False
            cell = self.current_cell.strip()
            if self.current_href:
                cell = f"{cell}|||{self.current_href}"
                self.current_href = None
            self.current_row.append(cell)
        elif tag == "tr" and self.current_row:
            self.rows.append(self.current_row)

    def handle_data(self, data: str) -> None:
        if self.in_td:
            self.current_cell += data


def _parse_ptr_pdf_text(text: str, member_name: str, pdf_url: str) -> list[dict[str, Any]]:
    """Parse trade rows from extracted PTR PDF text.

    PTR format (after text extraction):
    - Header with filer name, status, state/district
    - Transaction table with: ID, Owner, Asset, Transaction Type, Date,
      Notification Date, Amount, Cap. Gains > $200?
    """
    trades: list[dict[str, Any]] = []

    # Extract state/district
    state_match = re.search(r"State/District:\s*([A-Z]{2}\d{0,2})", text)
    state = state_match.group(1) if state_match else ""

    # Pattern for transaction lines in PTR PDFs
    # Format: asset description (TICKER) [TYPE]\n TX_TYPE DATE DATE AMOUNT
    # The text extraction concatenates these, so we need a flexible regex
    #
    # Common patterns after pypdf extraction:
    # "Apple Inc. (AAPL) [ST]\nP 01/15/202502/01/2025$1,001 - $15,000"
    # "GSK plc American Depositary Shares\n(GSK) [ST]\nS 07/28/202508/11/2025$1,001 - $15,000"

    # Split into lines for processing
    lines = text.split("\n")

    # State machine to find transaction blocks
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for lines containing a transaction type code (P, S, S (partial), E)
        # followed by dates and amounts
        tx_match = re.search(
            r"([PSE](?:\s*\((?:partial|full)\))?)\s+"
            r"(\d{2}/\d{2}/\d{4})"
            r"(\d{2}/\d{2}/\d{4})"
            r"(\$[\d,]+\s*-\s*\$[\d,]+(?:\+)?|\$[\d,]+\+)",
            line,
        )
        if not tx_match:
            # Also try the format where type/dates/amount are smushed together
            tx_match = re.search(
                r"\b([PSE])\s+"
                r"(\d{2}/\d{2}/\d{4})"
                r"(\d{2}/\d{2}/\d{4})"
                r"(\$[\d,]+\s*-\s*\$[\d,]+(?:\+)?|\$[\d,]+\+)",
                line,
            )

        if tx_match:
            tx_code = tx_match.group(1).strip()
            tx_date_str = tx_match.group(2)
            notif_date_str = tx_match.group(3)
            amount_str = tx_match.group(4).strip()

            # Look backwards for asset description and ticker
            asset_text = ""
            ticker = None
            # Collect text before the transaction code on this line
            pre_tx = line[: tx_match.start()].strip()
            if pre_tx:
                asset_text = pre_tx
            else:
                # Look at previous lines for asset info
                j = i - 1
                collected = []
                while j >= 0 and len(collected) < 3:
                    prev = lines[j].strip()
                    if not prev or prev.startswith("ID") or prev.startswith("Filing"):
                        break
                    collected.insert(0, prev)
                    j -= 1
                asset_text = " ".join(collected)

            # Extract ticker from parentheses, e.g., "(AAPL)"
            ticker_match = re.search(r"\(([A-Z]{1,5})\)", asset_text)
            if ticker_match:
                ticker = ticker_match.group(1)

            # Clean asset name
            asset_name = re.sub(r"\s*\([A-Z]{1,5}\)\s*", " ", asset_text)
            asset_name = re.sub(r"\[(?:ST|OP|OT|CS|EF|MF|BN|DC|OI)\]", "", asset_name)
            asset_name = asset_name.strip()

            # Extract owner from earlier in the text block
            owner = "Self"
            # Check if "SP" (spouse) or "DC" (dependent child) or "JT" (joint)
            # appears before the asset name in nearby context
            owner_check = " ".join(lines[max(0, i - 3) : i + 1])
            if re.search(r"\bSP\b", owner_check):
                owner = "Spouse"
            elif re.search(r"\bDC\b", owner_check):
                owner = "Dependent"
            elif re.search(r"\bJT\b", owner_check):
                owner = "Joint"

            trades.append(
                {
                    "member_name": member_name,
                    "ticker": ticker,
                    "asset_name": asset_name,
                    "transaction_code": tx_code,
                    "transaction_date_str": tx_date_str,
                    "disclosure_date_str": notif_date_str,
                    "amount_str": amount_str,
                    "owner": owner,
                    "state_district": state,
                    "pdf_url": pdf_url,
                }
            )
        i += 1

    return trades


class HouseClerkCollector(BaseCollector):
    """Collect House trades by scraping the House clerk's financial disclosure search."""

    source_name = "house_clerk"
    rate_limiter = _house_clerk_rate_limiter

    def __init__(self, years: list[int] | None = None) -> None:
        super().__init__()
        if years is None:
            # Default to current year only to keep initial seed fast
            # (~1500 PTR PDFs per year, ~2 req/sec = ~12min per year)
            self.years = [date.today().year]
        else:
            self.years = years

    async def collect(self) -> list[dict[str, Any]]:
        """Search for all PTR filings and download/parse each PDF."""
        all_trades: list[dict[str, Any]] = []

        for year in self.years:
            logger.info("[%s] Fetching PTR filings for %d", self.source_name, year)
            filings = await self._search_filings(year)
            ptr_filings = [f for f in filings if "PTR" in f.get("filing_type", "")]
            logger.info(
                "[%s] Found %d PTR filings for %d", self.source_name, len(ptr_filings), year
            )

            for filing in ptr_filings:
                pdf_url = filing.get("pdf_url", "")
                if not pdf_url:
                    continue

                try:
                    if self.rate_limiter:
                        await self.rate_limiter.acquire()

                    trades = await self._download_and_parse_ptr(
                        pdf_url, filing.get("name", "")
                    )
                    all_trades.extend(trades)
                except Exception as e:
                    logger.debug(
                        "[%s] Failed to parse PDF %s: %s", self.source_name, pdf_url, e
                    )

        logger.info("[%s] Extracted %d total trades from PDFs", self.source_name, len(all_trades))
        return all_trades

    async def _search_filings(self, year: int) -> list[dict[str, str]]:
        """Search the House clerk for all filings in a given year."""
        if self.rate_limiter:
            await self.rate_limiter.acquire()

        response = await self.client.post(
            HOUSE_CLERK_SEARCH_URL,
            json={"LastName": "", "FilingYear": str(year), "State": "", "District": ""},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        parser = _SearchResultParser()
        parser.feed(response.text)

        filings = []
        for row in parser.rows:
            if len(row) < 4:
                continue
            name_parts = row[0].split("|||")
            name = name_parts[0].strip()
            pdf_path = name_parts[1] if len(name_parts) > 1 else ""

            pdf_url = ""
            if pdf_path:
                pdf_url = HOUSE_CLERK_BASE_URL + pdf_path.lstrip("/")

            filings.append(
                {
                    "name": name,
                    "office": row[1].strip() if len(row) > 1 else "",
                    "filing_year": row[2].strip() if len(row) > 2 else "",
                    "filing_type": row[3].strip() if len(row) > 3 else "",
                    "pdf_url": pdf_url,
                }
            )

        return filings

    async def _download_and_parse_ptr(
        self, pdf_url: str, member_name: str
    ) -> list[dict[str, Any]]:
        """Download a PTR PDF and extract trade records."""
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.warning("pypdf not installed, cannot parse House PTR PDFs")
            return []

        response = await self.client.get(pdf_url)
        response.raise_for_status()

        # Verify it's actually a PDF
        if not response.content.startswith(b"%PDF"):
            return []

        reader = PdfReader(BytesIO(response.content))
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        return _parse_ptr_pdf_text(full_text, member_name, pdf_url)

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        ticker = (raw.get("ticker") or "").strip()
        if not ticker or ticker == "--" or ticker == "N/A":
            ticker = None

        transaction_date = self._parse_date(raw.get("transaction_date_str", ""))
        if transaction_date is None:
            return None

        disclosure_date = self._parse_date(raw.get("disclosure_date_str", ""))

        # Transaction type from code
        tx_code = (raw.get("transaction_code") or "").strip().upper()
        if tx_code == "P":
            tx_type = "purchase"
        elif tx_code.startswith("S"):
            if "partial" in raw.get("transaction_code", "").lower():
                tx_type = "sale_partial"
            elif "full" in raw.get("transaction_code", "").lower():
                tx_type = "sale_full"
            else:
                tx_type = "sale"
        elif tx_code == "E":
            tx_type = "exchange"
        else:
            tx_type = "unknown"

        # Amount parsing
        amount_str = (raw.get("amount_str") or "").strip()
        amount_low, amount_high = AMOUNT_RANGES.get(amount_str, (None, None))

        # Owner / filer type
        owner = (raw.get("owner") or "").lower().strip()
        if "spouse" in owner:
            filer_type = "spouse"
        elif "dependent" in owner or "child" in owner:
            filer_type = "dependent"
        elif "joint" in owner:
            filer_type = "joint"
        else:
            filer_type = "member"

        member_name = (raw.get("member_name") or "").strip()
        # Clean up "Hon.." prefix
        member_name = re.sub(r"^Hon\.\.?\s*", "", member_name)
        if not member_name:
            return None

        return {
            "member_name": member_name,
            "filer_type": filer_type,
            "ticker": ticker,
            "asset_name": (raw.get("asset_name") or "").strip(),
            "asset_type": "Stock" if ticker else None,
            "transaction_type": tx_type,
            "transaction_date": transaction_date,
            "disclosure_date": disclosure_date,
            "amount_range_low": amount_low,
            "amount_range_high": amount_high,
            "chamber": "house",
            "source": self.source_name,
            "filing_url": raw.get("pdf_url"),
            "raw_data": raw,
        }

    @staticmethod
    def _parse_date(date_str: str) -> date | None:
        if not date_str:
            return None
        for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"):
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue
        return None
