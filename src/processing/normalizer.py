"""Entity resolution: match lobbying clients and employers to stock tickers.

Uses a multi-strategy approach:
1. Manual override table (highest confidence)
2. Exact name matching against known company-ticker pairs
3. Fuzzy string matching (thefuzz/rapidfuzz)
4. Common abbreviation/suffix normalization

The resolved matches are stored on LobbyingClient.matched_ticker and
CampaignContribution.matched_ticker for use in the network graph.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from thefuzz import fuzz

from src.models.campaign_finance import CampaignContribution
from src.models.lobbying import LobbyingClient
from src.models.trade import TradeDisclosure

logger = logging.getLogger(__name__)

# Confidence thresholds
EXACT_MATCH_CONFIDENCE = 1.0
MANUAL_MATCH_CONFIDENCE = 0.99
FUZZY_HIGH_CONFIDENCE = 0.90
FUZZY_MIN_CONFIDENCE = 0.80

# Common corporate suffixes to strip for matching
CORPORATE_SUFFIXES = re.compile(
    r"\b(inc\.?|corp\.?|corporation|company|co\.?|ltd\.?|llc|lp|plc|"
    r"group|holdings?|enterprises?|international|intl\.?|"
    r"technologies|technology|tech|systems?|solutions?|"
    r"pharmaceuticals?|pharma|therapeutics?|biosciences?|"
    r"industries|industrial|services?|financial|bancorp|"
    r"communications?|networks?|software|platforms?|"
    r"the|of|and|&)\s*",
    re.IGNORECASE,
)

# Manual overrides for tricky mappings (client_name -> ticker)
# These are cases where fuzzy matching fails or is ambiguous
MANUAL_OVERRIDES: dict[str, str] = {
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta platforms": "META",
    "facebook": "META",
    "amazon.com": "AMZN",
    "amazon": "AMZN",
    "microsoft": "MSFT",
    "apple": "AAPL",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "berkshire hathaway": "BRK-B",
    "johnson & johnson": "JNJ",
    "jpmorgan chase": "JPM",
    "jp morgan": "JPM",
    "jpmorgan": "JPM",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
    "pfizer": "PFE",
    "unitedhealth": "UNH",
    "exxon mobil": "XOM",
    "exxonmobil": "XOM",
    "chevron": "CVX",
    "at&t": "T",
    "verizon": "VZ",
    "comcast": "CMCSA",
    "walt disney": "DIS",
    "disney": "DIS",
    "boeing": "BA",
    "lockheed martin": "LMT",
    "raytheon": "RTX",
    "general dynamics": "GD",
    "northrop grumman": "NOC",
    "general electric": "GE",
    "honeywell": "HON",
    "3m": "MMM",
    "caterpillar": "CAT",
    "deere": "DE",
    "john deere": "DE",
    "procter & gamble": "PG",
    "coca-cola": "KO",
    "pepsico": "PEP",
    "walmart": "WMT",
    "home depot": "HD",
    "costco": "COST",
    "salesforce": "CRM",
    "oracle": "ORCL",
    "intel": "INTC",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "broadcom": "AVGO",
    "qualcomm": "QCOM",
    "texas instruments": "TXN",
    "cisco": "CSCO",
    "cisco systems": "CSCO",
    "ibm": "IBM",
    "adobe": "ADBE",
    "uber": "UBER",
    "lyft": "LYFT",
    "airbnb": "ABNB",
    "robinhood": "HOOD",
    "coinbase": "COIN",
    "palantir": "PLTR",
}


def normalize_company_name(name: str) -> str:
    """Normalize a company name for matching.

    Strips corporate suffixes, punctuation, and extra whitespace.
    """
    if not name:
        return ""
    normalized = name.strip().lower()
    normalized = CORPORATE_SUFFIXES.sub(" ", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def match_name_to_ticker(
    name: str,
    known_tickers: dict[str, str],
) -> tuple[str | None, float, str]:
    """Match a company name to a ticker symbol.

    Args:
        name: The company/client/employer name to match.
        known_tickers: Dict mapping normalized company names to tickers,
                       built from TradeDisclosure.asset_name.

    Returns:
        (ticker, confidence, method) or (None, 0.0, "none")
    """
    if not name:
        return None, 0.0, "none"

    name_lower = name.strip().lower()
    normalized = normalize_company_name(name)

    # Strategy 1: Manual override (check against both raw and normalized forms)
    for override_name, ticker in MANUAL_OVERRIDES.items():
        if (
            normalized == override_name
            or override_name in normalized
            or name_lower == override_name
            or override_name in name_lower
        ):
            return ticker, MANUAL_MATCH_CONFIDENCE, "manual"

    # Strategy 2: Exact match against known tickers
    if normalized in known_tickers:
        return known_tickers[normalized], EXACT_MATCH_CONFIDENCE, "exact"

    # Strategy 3: Fuzzy matching
    best_score = 0
    best_ticker = None
    for known_name, ticker in known_tickers.items():
        # Use token_sort_ratio which handles word order differences
        score = fuzz.token_sort_ratio(normalized, known_name) / 100.0
        if score > best_score:
            best_score = score
            best_ticker = ticker

    if best_score >= FUZZY_HIGH_CONFIDENCE and best_ticker:
        return best_ticker, best_score, "fuzzy"

    if best_score >= FUZZY_MIN_CONFIDENCE and best_ticker:
        return best_ticker, best_score, "fuzzy_low"

    return None, 0.0, "none"


async def build_ticker_lookup(session: AsyncSession) -> dict[str, str]:
    """Build a lookup dict: normalized_asset_name -> ticker.

    Uses TradeDisclosure records to build the known company universe.
    """
    result = await session.execute(
        select(TradeDisclosure.ticker, TradeDisclosure.asset_name)
        .where(TradeDisclosure.ticker.is_not(None))
        .where(TradeDisclosure.asset_name.is_not(None))
        .distinct()
    )

    lookup: dict[str, str] = {}
    for ticker, asset_name in result.all():
        normalized = normalize_company_name(asset_name)
        if normalized and ticker:
            lookup[normalized] = ticker

    logger.info("Built ticker lookup with %d entries", len(lookup))
    return lookup


async def resolve_lobbying_clients(session: AsyncSession) -> int:
    """Match unresolved LobbyingClient records to stock tickers."""
    ticker_lookup = await build_ticker_lookup(session)

    # Get unresolved clients
    result = await session.execute(
        select(LobbyingClient).where(LobbyingClient.matched_ticker.is_(None))
    )
    clients = result.scalars().all()

    resolved = 0
    for client in clients:
        ticker, confidence, method = match_name_to_ticker(client.name, ticker_lookup)
        if ticker and confidence >= FUZZY_MIN_CONFIDENCE:
            await session.execute(
                update(LobbyingClient)
                .where(LobbyingClient.id == client.id)
                .values(
                    matched_ticker=ticker,
                    match_confidence=confidence,
                    match_method=method,
                    normalized_name=normalize_company_name(client.name),
                )
            )
            resolved += 1

    await session.commit()
    logger.info("Resolved %d out of %d lobbying clients", resolved, len(clients))
    return resolved


async def resolve_campaign_employers(session: AsyncSession) -> int:
    """Match unresolved campaign contribution employers to stock tickers."""
    ticker_lookup = await build_ticker_lookup(session)

    # Get unresolved contributions with employer info
    result = await session.execute(
        select(CampaignContribution)
        .where(CampaignContribution.matched_ticker.is_(None))
        .where(CampaignContribution.contributor_employer.is_not(None))
    )
    contributions = result.scalars().all()

    # Cache resolved employers to avoid re-matching
    employer_cache: dict[str, tuple[str | None, float]] = {}
    resolved = 0

    for contrib in contributions:
        employer = contrib.contributor_employer
        if not employer:
            continue

        if employer not in employer_cache:
            ticker, confidence, _method = match_name_to_ticker(employer, ticker_lookup)
            employer_cache[employer] = (ticker, confidence)

        ticker, confidence = employer_cache[employer]
        if ticker and confidence >= FUZZY_MIN_CONFIDENCE:
            await session.execute(
                update(CampaignContribution)
                .where(CampaignContribution.id == contrib.id)
                .values(
                    matched_ticker=ticker,
                    match_confidence=confidence,
                )
            )
            resolved += 1

    await session.commit()
    logger.info("Resolved %d out of %d contribution employers", resolved, len(contributions))
    return resolved
