"""Celery tasks for media content collection and NLP analysis."""

from __future__ import annotations

import asyncio
import logging

from src.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


async def _collect_and_store(collector_cls, **kwargs):
    """Helper: instantiate collector, run, and store results."""
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from src.config import settings
    from src.ingestion.loader import upsert_media_content

    engine = create_async_engine(settings.database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        collector = collector_cls(**kwargs)
        records = await collector.run()
        await collector.close()

        if records:
            async with async_session() as session:
                count = await upsert_media_content(session, records)
                logger.info("Stored %d records from %s", count, collector.source_name)
        else:
            logger.info("No records from %s", collector_cls.source_name)
    finally:
        await engine.dispose()


@celery_app.task(name="src.tasks.media_tasks.collect_hearing_transcripts")
def collect_hearing_transcripts():
    """Collect committee hearing transcripts from GovInfo."""
    from src.ingestion.media.govinfo_hearings import GovInfoHearingCollector

    asyncio.run(_collect_and_store(GovInfoHearingCollector))


@celery_app.task(name="src.tasks.media_tasks.collect_youtube_transcripts")
def collect_youtube_transcripts():
    """Collect YouTube interview transcripts from congressional channels."""
    from src.ingestion.media.youtube import YouTubeTranscriptCollector

    asyncio.run(_collect_and_store(YouTubeTranscriptCollector))


@celery_app.task(name="src.tasks.media_tasks.collect_news_articles")
def collect_news_articles():
    """Collect news articles from GNews and NewsData."""
    from src.ingestion.media.gnews import GNewsCollector
    from src.ingestion.media.newsdata import NewsDataCollector

    asyncio.run(_collect_and_store(GNewsCollector))
    asyncio.run(_collect_and_store(NewsDataCollector))


@celery_app.task(name="src.tasks.media_tasks.collect_congress_rss")
def collect_congress_rss():
    """Collect Congress.gov RSS feed updates."""
    from src.ingestion.media.congress_rss import CongressRSSCollector

    asyncio.run(_collect_and_store(CongressRSSCollector))


@celery_app.task(name="src.tasks.media_tasks.collect_press_releases")
def collect_press_releases():
    """Collect press releases from member websites."""
    from src.ingestion.media.press_releases import PressReleaseCollector

    asyncio.run(_collect_and_store(PressReleaseCollector))


@celery_app.task(name="src.tasks.media_tasks.collect_tweets")
def collect_tweets():
    """Collect tweets from congress member accounts.

    Only runs if TWITTER_BEARER_TOKEN is configured.
    """
    from src.ingestion.media.twitter import TwitterCollector

    asyncio.run(_collect_and_store(TwitterCollector))


@celery_app.task(name="src.tasks.media_tasks.run_nlp_analysis")
def run_nlp_analysis():
    """Process unanalyzed media content through the NLP pipeline.

    Finds media content without sentiment analysis records and runs
    FinBERT sentiment, spaCy NER, and ticker extraction on each.
    """
    asyncio.run(_run_nlp())


async def _run_nlp():
    """Run NLP analysis on unprocessed media content."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from src.config import settings
    from src.ingestion.loader import upsert_sentiment_analyses
    from src.models.media import MediaContent, SentimentAnalysis
    from src.processing.text_processing import (
        analyze_sentiment,
        extract_entities,
        extract_ticker_mentions,
    )

    engine = create_async_engine(settings.database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as session:
            # Find content without sentiment analysis
            subquery = (
                select(SentimentAnalysis.media_content_id)
                .where(SentimentAnalysis.model_name == "finbert")
            )
            result = await session.execute(
                select(MediaContent)
                .where(MediaContent.id.not_in(subquery))
                .where(MediaContent.content.is_not(None))
                .where(MediaContent.content != "")
                .limit(100)  # Process in batches
            )
            unprocessed = result.scalars().all()

            if not unprocessed:
                logger.info("No unprocessed media content found")
                return

            logger.info("Processing %d media content items through NLP", len(unprocessed))

            analyses = []
            for content in unprocessed:
                text = content.content or ""
                if not text.strip():
                    continue

                try:
                    sentiment = analyze_sentiment(text)
                    entities = extract_entities(text)
                    tickers = extract_ticker_mentions(text)

                    analyses.append({
                        "media_content_id": content.id,
                        "model_name": "finbert",
                        "sentiment_label": sentiment["label"],
                        "sentiment_score": sentiment["score"],
                        "confidence": sentiment["confidence"],
                        "entities": entities,
                        "sectors": [],
                        "tickers_extracted": tickers,
                    })
                except Exception:
                    logger.exception(
                        "Failed to analyze content id=%d", content.id
                    )

            if analyses:
                count = await upsert_sentiment_analyses(session, analyses)
                logger.info("Created %d sentiment analyses", count)

    finally:
        await engine.dispose()
