"""NLP pipeline for media content analysis.

Provides sentiment analysis (FinBERT), named entity recognition (spaCy),
zero-shot sector classification, and ticker mention extraction.

Heavy ML models are lazy-loaded on first use to avoid slow imports
in processes that don't need NLP (API server, non-NLP workers).

Requires the [ml] optional dependency group:
  pip install -e ".[ml]"
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-loaded model singletons
_finbert_pipeline: Any = None
_spacy_nlp: Any = None
_zero_shot_pipeline: Any = None

# FinBERT max input length (tokens)
FINBERT_MAX_TOKENS = 512

# Sectors for zero-shot classification
SECTOR_LABELS = [
    "Technology",
    "Healthcare",
    "Finance",
    "Energy",
    "Defense",
    "Consumer",
    "Industrial",
    "Real Estate",
    "Telecommunications",
    "Transportation",
    "Agriculture",
    "Education",
]

# Regex to find ticker-like mentions ($NVDA, NVDA, etc.)
TICKER_PATTERN = re.compile(r"\$?([A-Z]{1,5})\b")


def get_finbert() -> Any:
    """Get or initialize the FinBERT sentiment analysis pipeline."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline

            _finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                truncation=True,
                max_length=FINBERT_MAX_TOKENS,
            )
            logger.info("Loaded FinBERT sentiment model")
        except ImportError:
            logger.error(
                "transformers not installed. Install with: pip install -e '.[ml]'"
            )
            raise
    return _finbert_pipeline


def get_spacy() -> Any:
    """Get or initialize the spaCy NLP pipeline."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy

            _spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy en_core_web_sm model")
        except ImportError:
            logger.error("spacy not installed. Install with: pip install -e '.[ml]'")
            raise
        except OSError:
            logger.error(
                "spaCy model not found. Install with: python -m spacy download en_core_web_sm"
            )
            raise
    return _spacy_nlp


def get_zero_shot() -> Any:
    """Get or initialize the zero-shot classification pipeline."""
    global _zero_shot_pipeline
    if _zero_shot_pipeline is None:
        try:
            from transformers import pipeline

            _zero_shot_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
            )
            logger.info("Loaded zero-shot classification model")
        except ImportError:
            logger.error(
                "transformers not installed. Install with: pip install -e '.[ml]'"
            )
            raise
    return _zero_shot_pipeline


def analyze_sentiment(text: str) -> dict[str, Any]:
    """Run FinBERT sentiment analysis on text.

    Returns:
        {label: "positive"|"negative"|"neutral", score: float, confidence: float}
    """
    if not text or not text.strip():
        return {"label": "neutral", "score": 0.0, "confidence": 0.0}

    # Truncate for FinBERT
    truncated = text[:2000]

    pipe = get_finbert()
    result = pipe(truncated)[0]

    # FinBERT returns: {label: "positive"/"negative"/"neutral", score: 0-1}
    label = result["label"].lower()
    confidence = result["score"]

    # Convert to signed score: positive=+score, negative=-score, neutral=0
    if label == "positive":
        score = confidence
    elif label == "negative":
        score = -confidence
    else:
        score = 0.0

    return {"label": label, "score": score, "confidence": confidence}


def extract_entities(text: str) -> list[dict[str, Any]]:
    """Extract named entities using spaCy NER.

    Returns:
        [{text: str, label: str, start: int, end: int}, ...]
    """
    if not text or not text.strip():
        return []

    nlp = get_spacy()
    # Process only first 100K chars to avoid memory issues
    doc = nlp(text[:100_000])

    entities = []
    seen = set()
    for ent in doc.ents:
        key = (ent.text, ent.label_)
        if key not in seen:
            seen.add(key)
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })

    return entities


def classify_sectors(
    text: str, candidate_labels: list[str] | None = None
) -> list[dict[str, Any]]:
    """Classify text into sectors using zero-shot classification.

    Returns:
        [{sector: str, score: float}, ...] sorted by score descending
    """
    if not text or not text.strip():
        return []

    labels = candidate_labels or SECTOR_LABELS

    pipe = get_zero_shot()
    result = pipe(text[:1000], labels, multi_label=True)

    sectors = []
    for label, score in zip(result["labels"], result["scores"]):
        if score >= 0.1:  # Only include sectors with some relevance
            sectors.append({"sector": label, "score": round(score, 4)})

    return sectors


def extract_ticker_mentions(
    text: str, known_tickers: set[str] | None = None
) -> list[str]:
    """Extract stock ticker mentions from text.

    Finds patterns like $NVDA, NVDA, and validates against known tickers
    to reduce false positives.

    Args:
        text: The text to search for ticker mentions.
        known_tickers: Set of valid ticker symbols. If provided, only
                      returns tickers that exist in this set.

    Returns:
        List of unique ticker symbols found.
    """
    if not text:
        return []

    # Common English words that look like tickers
    false_positives = {
        "A", "I", "AM", "PM", "AN", "AS", "AT", "BE", "BY", "DO",
        "GO", "HE", "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF",
        "OK", "ON", "OR", "SO", "TO", "UP", "US", "WE", "CEO", "CFO",
        "CTO", "VP", "HR", "PR", "TV", "UK", "EU", "UN", "GDP", "FBI",
        "CIA", "SEC", "FEC", "IRS", "DOJ", "EPA", "FDA", "FCC", "FTC",
        "NASA", "NATO", "NYSE", "IPO", "ETF", "LLC", "INC", "CORP",
        "THE", "FOR", "AND", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
        "WAS", "ONE", "OUR", "OUT", "HAS", "HAD", "HOW", "NEW", "NOW",
        "OLD", "SEE", "WAY", "WHO", "DID", "GET", "HIS", "LET", "SAY",
        "SHE", "TOO", "USE", "ACT", "MAY", "USA", "RSS",
    }

    matches = TICKER_PATTERN.findall(text)
    tickers: list[str] = []
    seen: set[str] = set()

    for match in matches:
        ticker = match.upper()
        if ticker in seen:
            continue
        if ticker in false_positives:
            continue
        if known_tickers and ticker not in known_tickers:
            continue
        seen.add(ticker)
        tickers.append(ticker)

    return tickers


def strip_html(text: str) -> str:
    """Remove HTML tags, entities, and collapse whitespace."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"&[a-zA-Z]+;", " ", clean)
    clean = re.sub(r"&#?\w+;", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def truncate_for_model(text: str, max_chars: int = 2000) -> str:
    """Truncate text to approximate token limit for transformer models."""
    if not text:
        return ""
    return text[:max_chars]
