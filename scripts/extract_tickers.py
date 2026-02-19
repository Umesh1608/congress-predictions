"""Extract tickers from untickered trade_disclosure asset_name descriptions.

Strategies:
1. Bare tickers: asset_name is already a valid ticker (e.g. "WMT", "GE")
2. Parenthetical tickers: "(BRK.B)" or "(TICKER)" embedded in name
3. Known company → ticker mapping (manual + pattern matching)
4. Prefix/suffix stripping: "DIGITAL REALTY TRUST INC" → look up via known trades

Run: python -m scripts.extract_tickers [--dry-run]
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys

from sqlalchemy import text

from src.db.postgres import async_session_factory

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Manual company name → ticker mappings for common untickered assets
_MANUAL_MAPPINGS: dict[str, str] = {
    # Bare-ish names that aren't valid tickers
    "Apple, Inc.": "AAPL",
    "Amazon": "AMZN",
    "Amazon.com": "AMZN",
    "AMAZON COM INC": "AMZN",
    "Google": "GOOG",
    "Alphabet Inc": "GOOG",
    "Facebook": "META",
    "Meta Platforms": "META",
    "Microsoft": "MSFT",
    "MICROSOFT CORP": "MSFT",
    "Tesla": "TSLA",
    "TESLA INC": "TSLA",
    "NVIDIA": "NVDA",
    "NVIDIA CORP": "NVDA",

    # Common ones from our data
    "UNIVAR INC.": "UNVR",
    "POST HOLDINGS INC.": "POST",
    "DIGITAL REALTY TRUST INC": "DLR",
    "ANTERO MIDSTREAM GP LP": "AM",
    "DOWDUPONT INC.": "DD",
    "Cabela's Inc CMN": "CAB",
    "Kite Pharma, Inc (stock) NASDAQ": "KITE",
    "Centurylink Inc. (stock) NYSE": "CTL",
    "OCI NV Sponsored ADR (stock) NYSE": "OCI",
    "Wynn Resorts LTD (stock) NASDAQ": "WYNN",
    "Univar Inc. (stock)": "UNVR",
    "REALNETWORKS INC (stock) NASDAQ": "RNWK",
    "Esterline Technologies Corporation (stock) NYSE": "ESL",
    "CSX Corporation (stock) NASDAQ": "CSX",
    "Devon Energy Corp. (stock) NYSE": "DVN",
    "CEMEX - SAB DE CV sponsored ADR Part CER (stock) NYSE": "CX",
    "Mohawk Industries Inc. (stock) NYSE": "MHK",
    "Post Holdings Inc. (stock)": "POST",
    "Applied Materials Incorporated (stock) NASDAQ": "AMAT",
    "Zimmer Biomet Holdings, Inc. (stock) NYSE": "ZBH",
    "Lowes Corporation (stock) NYSE": "LOW",
    "Park Hotels and Resorts Inc. (stock) NYSE": "PK",
    "Ingersoll Rand PLC (stock) NYSE": "IR",
    "Energy Transfer Equity (ETE)": "ET",
    "Tesoro Logistics LP CMN": "ANDX",
    "Advanced Disposal Services Inc., Del common stock": "ADSW",

    # Additional well-known companies
    "APPLE INC": "AAPL",
    "BOEING CO": "BA",
    "BOEING COMPANY": "BA",
    "WALT DISNEY CO": "DIS",
    "DISNEY": "DIS",
    "EXXON MOBIL CORP": "XOM",
    "CHEVRON CORP": "CVX",
    "JOHNSON & JOHNSON": "JNJ",
    "PROCTER & GAMBLE CO": "PG",
    "JPMORGAN CHASE": "JPM",
    "BANK OF AMERICA": "BAC",
    "WELLS FARGO": "WFC",
    "GOLDMAN SACHS": "GS",
    "MORGAN STANLEY": "MS",
    "INTEL CORP": "INTC",
    "CISCO SYSTEMS": "CSCO",
    "PFIZER INC": "PFE",
    "MERCK & CO": "MRK",
    "ABBVIE INC": "ABBV",
    "COMCAST CORP": "CMCSA",
    "AT&T INC": "T",
    "VERIZON": "VZ",
    "HOME DEPOT": "HD",
    "WALMART INC": "WMT",
    "COSTCO": "COST",
    "BERKSHIRE HATHAWAY": "BRK.B",
    "SALESFORCE": "CRM",
    "ADOBE INC": "ADBE",
    "NETFLIX INC": "NFLX",
    "PAYPAL": "PYPL",
    "VISA INC": "V",
    "MASTERCARD": "MA",
    "UNITEDHEALTH": "UNH",
}

# Regex to find parenthetical tickers like "(BRK.B)" or "(AAPL)"
_PAREN_TICKER_RE = re.compile(r"\(([A-Z]{1,5}(?:\.[A-Z])?)\)")

# Regex for bare ticker: 1-5 uppercase letters only
_BARE_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")

# Suffixes to strip when matching company names
_COMPANY_SUFFIXES = re.compile(
    r",?\s*(?:Inc\.?|Corp\.?|Corporation|Company|Co\.?|Ltd\.?|PLC|LP|LLC|CMN|N\.?V\.?|SA|AG|SE|"
    r"common\s+stock|sponsored\s+ADR|Del|New|Class\s+[A-Z])\s*\.?\s*",
    re.IGNORECASE,
)

# Words that indicate this is a fund/trust/bond, not a stock
_FUND_WORDS = {
    "fund", "trust", "portfolio", "bond", "tiaa", "vanguard", "fidelity",
    "schwab", "morgan stanley institutional", "age-based", "lifecycle",
    "retirement", "savings plan", "municipal", "treasury", "money market",
    "certificate", "annuity", "reit", "index", "etf",
}


def _is_fund(name: str) -> bool:
    """Check if asset_name looks like a fund/bond/trust rather than a stock."""
    lower = name.lower()
    return any(w in lower for w in _FUND_WORDS)


def _extract_ticker(
    asset_name: str,
    known_tickers: set[str],
    name_to_ticker: dict[str, str],
) -> str | None:
    """Try to extract a ticker from an asset_name.

    Returns ticker string or None if not matchable.
    """
    name = asset_name.strip()

    # 0. Skip funds/trusts/bonds early
    if _is_fund(name):
        return None

    # 1. Check manual mappings first (case-insensitive keys)
    if name in _MANUAL_MAPPINGS:
        return _MANUAL_MAPPINGS[name]

    # Also check case-insensitive
    name_upper = name.upper()
    for k, v in _MANUAL_MAPPINGS.items():
        if k.upper() == name_upper:
            return v

    # 2. Bare ticker: "WMT", "GE", "NFLX"
    if _BARE_TICKER_RE.match(name):
        if name in known_tickers:
            return name
        # Even if not in our stock_daily, it's likely a valid ticker
        return name

    # 3. Parenthetical ticker: "Energy Transfer Equity (ETE)"
    m = _PAREN_TICKER_RE.search(name)
    if m:
        ticker = m.group(1)
        # Must be 2+ chars to avoid false positives like "(A)" in fund names
        if len(ticker) >= 2 and ticker in known_tickers:
            return ticker
        if len(ticker) >= 2:
            return ticker

    # 4. Check name_to_ticker lookup (built from existing trades)
    # Normalize: strip suffixes, uppercase
    cleaned = _COMPANY_SUFFIXES.sub("", name).strip().upper()
    cleaned = re.sub(r"\s*\(stock\)\s*(NYSE|NASDAQ)?\s*$", "", cleaned, flags=re.IGNORECASE).strip()

    # Skip if cleaned result is too short or looks like garbage
    if len(cleaned) < 2 or cleaned == ".":
        return None

    if cleaned in name_to_ticker:
        result = name_to_ticker[cleaned]
        # Validate the result
        if len(result) >= 1 and result != ".":
            return result

    # Try without trailing punctuation
    cleaned2 = cleaned.rstrip(".,; ")
    if cleaned2 and cleaned2 in name_to_ticker:
        result = name_to_ticker[cleaned2]
        if len(result) >= 1 and result != ".":
            return result

    return None


async def extract_and_update(dry_run: bool = False) -> None:
    """Extract tickers from untickered trades and update the database."""
    async with async_session_factory() as session:
        # Build name→ticker lookup from existing tickered trades
        r = await session.execute(text("""
            SELECT DISTINCT asset_name, ticker
            FROM trade_disclosure
            WHERE ticker IS NOT NULL AND asset_name IS NOT NULL
        """))
        # Build reverse lookup: cleaned asset_name → ticker
        name_to_ticker: dict[str, str] = {}
        for asset_name, ticker in r.fetchall():
            cleaned = _COMPANY_SUFFIXES.sub("", asset_name).strip().upper()
            cleaned = re.sub(r"\s*\(stock\)\s*(NYSE|NASDAQ)?\s*$", "", cleaned, flags=re.IGNORECASE).strip()
            name_to_ticker[cleaned] = ticker
            # Also store raw uppercase
            name_to_ticker[asset_name.strip().upper()] = ticker

        logger.info("Built name→ticker lookup: %d entries", len(name_to_ticker))

        # Get known tickers
        r = await session.execute(text("SELECT DISTINCT ticker FROM stock_daily"))
        known_tickers = {row[0] for row in r.fetchall()}

        r = await session.execute(text("SELECT DISTINCT ticker FROM trade_disclosure WHERE ticker IS NOT NULL"))
        known_tickers.update(row[0] for row in r.fetchall())
        logger.info("Known tickers: %d", len(known_tickers))

        # Get all untickered asset names
        r = await session.execute(text("""
            SELECT DISTINCT asset_name
            FROM trade_disclosure
            WHERE ticker IS NULL AND asset_name IS NOT NULL
        """))
        untickered = [row[0] for row in r.fetchall()]
        logger.info("Distinct untickered asset names: %d", len(untickered))

        # Try to extract tickers
        matched: dict[str, str] = {}
        unmatched: list[str] = []

        for name in untickered:
            ticker = _extract_ticker(name, known_tickers, name_to_ticker)
            if ticker:
                matched[name] = ticker
            else:
                unmatched.append(name)

        logger.info("Matched: %d / %d asset names", len(matched), len(untickered))
        logger.info("Unmatched: %d (funds, trusts, other)", len(unmatched))

        # Show matches
        logger.info("\nMatched asset_name → ticker:")
        for name, ticker in sorted(matched.items(), key=lambda x: x[1]):
            logger.info("  %-60s → %s", name[:60], ticker)

        if dry_run:
            logger.info("\n[DRY RUN] Would update trades. Run without --dry-run to apply.")

            # Count how many trades would be updated
            total = 0
            for name in matched:
                r = await session.execute(
                    text("SELECT COUNT(*) FROM trade_disclosure WHERE asset_name = :name AND ticker IS NULL"),
                    {"name": name},
                )
                total += r.scalar()
            logger.info("[DRY RUN] Total trades that would gain a ticker: %d", total)
            return

        # Update trades — use per-row savepoints to skip dedup conflicts
        total_updated = 0
        total_skipped = 0
        conn = await session.connection()

        for name, ticker in matched.items():
            r = await session.execute(
                text("SELECT id FROM trade_disclosure WHERE asset_name = :name AND ticker IS NULL"),
                {"name": name},
            )
            trade_ids = [row[0] for row in r.fetchall()]

            for tid in trade_ids:
                savepoint = await conn.begin_nested()
                try:
                    await session.execute(
                        text("UPDATE trade_disclosure SET ticker = :ticker WHERE id = :tid"),
                        {"ticker": ticker, "tid": tid},
                    )
                    await savepoint.commit()
                    total_updated += 1
                except Exception:
                    await savepoint.rollback()
                    total_skipped += 1

        await session.commit()
        logger.info("\nUpdated %d trades with extracted tickers (skipped %d conflicts)",
                    total_updated, total_skipped)

        # Final stats
        r = await session.execute(text("SELECT COUNT(*) FROM trade_disclosure WHERE ticker IS NOT NULL"))
        logger.info("Total trades with ticker: %d", r.scalar())
        r = await session.execute(text("SELECT COUNT(*) FROM trade_disclosure WHERE ticker IS NULL"))
        logger.info("Total trades still without ticker: %d", r.scalar())


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    asyncio.run(extract_and_update(dry_run=dry_run))
