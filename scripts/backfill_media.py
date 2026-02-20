"""Backfill media content for trading members.

Collects news articles and other media to build sentiment features
for the ML models. Rate-limited by API quotas:
- GNews: 100 requests/day
- Congress RSS: unlimited (free)
- Press releases: unlimited (free RSS)

Run: python -m scripts.backfill_media [--gnews] [--rss] [--press] [--all]
"""

from __future__ import annotations

import asyncio
import logging
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.config import settings
from src.ingestion.loader import upsert_media_content

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Top trading members to backfill media for
TOP_TRADING_MEMBERS = [
    "Nancy Pelosi", "Tommy Tuberville", "Dan Crenshaw", "Josh Gottheimer",
    "Michael McCaul", "Mark Green", "Marjorie Taylor Greene", "Pat Fallon",
    "John Curtis", "French Hill", "Ro Khanna", "Kevin Hern",
    "Virginia Foxx", "Debbie Wasserman Schultz", "Tom Malinowski",
    "David Perdue", "Kelly Loeffler", "Richard Burr",
]


async def backfill_gnews():
    """Backfill GNews articles for trading members."""
    from src.ingestion.media.gnews import GNewsCollector

    if not settings.gnews_api_key:
        logger.warning("No GNEWS_API_KEY configured — skipping GNews backfill")
        return

    engine = create_async_engine(settings.database_url, pool_size=3)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    total = 0
    # Each member query uses ~1 API call, with 100/day limit
    queries = [f'"{name}" stock trade' for name in TOP_TRADING_MEMBERS[:10]]
    queries += [
        "congress stock trading disclosure",
        "congressional insider trading 2024",
        "senator stock trade 2024",
        "representative stock trade 2024",
        "STOCK Act violation",
    ]

    for query in queries:
        logger.info("GNews query: %s", query)
        try:
            collector = GNewsCollector(queries=[query], max_articles=10)
            records = await collector.run()
            if records:
                async with factory() as session:
                    inserted = await upsert_media_content(session, records)
                    total += inserted
                    logger.info("  Inserted %d articles (total: %d)", inserted, total)
        except Exception as e:
            logger.error("  GNews error for '%s': %s", query, e)

        # Respect rate limit
        await asyncio.sleep(1.0)

    logger.info("GNews backfill complete: %d total articles", total)
    await engine.dispose()


async def backfill_rss():
    """Backfill Congress RSS feeds."""
    from src.ingestion.media.congress_rss import CongressRSSCollector

    engine = create_async_engine(settings.database_url, pool_size=3)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    collector = CongressRSSCollector()
    try:
        records = await collector.run()
        if records:
            async with factory() as session:
                inserted = await upsert_media_content(session, records)
                logger.info("Congress RSS: inserted %d articles", inserted)
    except Exception as e:
        logger.error("RSS error: %s", e)

    await engine.dispose()


async def backfill_press_releases():
    """Backfill press releases from member websites using curated RSS feeds."""
    from src.ingestion.media.press_releases import PressReleaseCollector

    engine = create_async_engine(settings.database_url, pool_size=3)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    # Use the default curated MEMBER_RSS_FEEDS dict from the collector
    collector = PressReleaseCollector()
    try:
        records = await collector.run()
        if records:
            async with factory() as session:
                inserted = await upsert_media_content(session, records)
                logger.info("Press releases: inserted %d articles", inserted)
        else:
            logger.info("Press releases: no new articles found")
    except Exception as e:
        logger.error("Press release error: %s", e)

    await engine.dispose()


async def backfill_youtube():
    """Backfill YouTube transcripts from congressional channels."""
    from src.ingestion.media.youtube import YouTubeTranscriptCollector

    engine = create_async_engine(settings.database_url, pool_size=3)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    collector = YouTubeTranscriptCollector()
    try:
        records = await collector.run()
        if records:
            async with factory() as session:
                inserted = await upsert_media_content(session, records)
                logger.info("YouTube: inserted %d transcripts", inserted)
        else:
            logger.info("YouTube: no new transcripts found")
    except Exception as e:
        logger.error("YouTube error: %s", e)

    await engine.dispose()


async def backfill_newsdata():
    """Backfill news from NewsData.io API."""
    from src.ingestion.media.newsdata import NewsDataCollector

    if not settings.newsdata_api_key:
        logger.warning("No NEWSDATA_API_KEY configured — skipping NewsData backfill")
        return

    engine = create_async_engine(settings.database_url, pool_size=3)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    queries = [
        "congress stock trading",
        "congressional insider trading",
        "senator stock disclosure",
    ]
    total = 0
    for query in queries:
        logger.info("NewsData query: %s", query)
        try:
            collector = NewsDataCollector(queries=[query])
            records = await collector.run()
            if records:
                async with factory() as session:
                    inserted = await upsert_media_content(session, records)
                    total += inserted
                    logger.info("  Inserted %d articles (total: %d)", inserted, total)
        except Exception as e:
            logger.error("  NewsData error for '%s': %s", query, e)
        await asyncio.sleep(1.0)

    logger.info("NewsData backfill complete: %d total articles", total)
    await engine.dispose()


async def run_nlp_on_new_content():
    """Run NLP analysis on media content that hasn't been analyzed yet."""
    from src.processing.text_processing import analyze_sentiment, extract_entities, extract_ticker_mentions
    from src.ingestion.loader import upsert_sentiment_analyses

    engine = create_async_engine(settings.database_url, pool_size=3)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        r = await session.execute(text("""
            SELECT mc.id, mc.content, mc.title
            FROM media_content mc
            LEFT JOIN sentiment_analysis sa ON sa.media_content_id = mc.id
            WHERE sa.id IS NULL AND mc.content IS NOT NULL AND mc.content != ''
            LIMIT 500
        """))
        rows = r.fetchall()

    if not rows:
        logger.info("NLP: no new content to analyze")
        await engine.dispose()
        return

    logger.info("NLP: analyzing %d new media items...", len(rows))
    analyzed = 0
    for media_id, content, title in rows:
        text_to_analyze = content or title or ""
        if not text_to_analyze.strip():
            continue
        try:
            sentiment = analyze_sentiment(text_to_analyze)
            entities = extract_entities(text_to_analyze)
            tickers = extract_ticker_mentions(text_to_analyze)
            record = {
                "media_content_id": media_id,
                "model_name": "finbert",
                "sentiment_label": sentiment["label"],
                "sentiment_score": sentiment["score"],
                "confidence": sentiment["confidence"],
                "entities": [{"text": e["text"], "label": e["label"]} for e in entities],
                "tickers_extracted": tickers,
            }
            async with factory() as session:
                await upsert_sentiment_analyses(session, [record])
            analyzed += 1
        except Exception as e:
            logger.error("NLP error for media_id=%d: %s", media_id, e)

    logger.info("NLP: analyzed %d / %d items", analyzed, len(rows))
    await engine.dispose()


async def main():
    args = set(sys.argv[1:])
    run_all = "--all" in args or not args

    if run_all or "--rss" in args:
        logger.info("=== Backfilling Congress RSS ===")
        await backfill_rss()

    if run_all or "--press" in args:
        logger.info("=== Backfilling Press Releases ===")
        await backfill_press_releases()

    if run_all or "--youtube" in args:
        logger.info("=== Backfilling YouTube ===")
        await backfill_youtube()

    if run_all or "--newsdata" in args:
        logger.info("=== Backfilling NewsData ===")
        await backfill_newsdata()

    if run_all or "--gnews" in args:
        logger.info("=== Backfilling GNews ===")
        await backfill_gnews()

    if run_all or "--nlp" in args:
        logger.info("=== Running NLP Analysis ===")
        await run_nlp_on_new_content()


if __name__ == "__main__":
    asyncio.run(main())
