"""Media content and sentiment analysis API endpoints."""

from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.models.media import MediaContent, MemberMediaMention, SentimentAnalysis
from src.models.member import CongressMember
from src.schemas.media import (
    MediaContentDetailResponse,
    MediaContentResponse,
    MediaStatsResponse,
    MemberSentimentPoint,
    MemberSentimentTimelineResponse,
    SentimentResponse,
)

router = APIRouter(tags=["media"])


@router.get("/media", response_model=list[MediaContentResponse])
async def list_media_content(
    source_type: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    member_bioguide_id: str | None = None,
    ticker: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    query = select(MediaContent).order_by(MediaContent.published_date.desc())

    if source_type:
        query = query.where(MediaContent.source_type == source_type)
    if date_from:
        query = query.where(MediaContent.published_date >= date_from)
    if date_to:
        query = query.where(MediaContent.published_date <= date_to)
    if ticker:
        query = query.where(
            MediaContent.tickers_mentioned.contains([ticker])
        )
    if member_bioguide_id:
        query = query.where(
            MediaContent.member_bioguide_ids.contains([member_bioguide_id])
        )

    query = query.offset(offset).limit(limit)
    result = await db.execute(query)
    items = result.scalars().all()

    responses = []
    for item in items:
        resp = {
            "id": item.id,
            "source_type": item.source_type,
            "title": item.title,
            "summary": item.summary,
            "url": item.url,
            "author": item.author,
            "published_date": item.published_date,
            "tickers_mentioned": item.tickers_mentioned or [],
            "member_bioguide_ids": item.member_bioguide_ids or [],
            "created_at": item.created_at,
            "sentiment": None,
        }

        # Attach primary sentiment if available
        sent_result = await db.execute(
            select(SentimentAnalysis)
            .where(SentimentAnalysis.media_content_id == item.id)
            .limit(1)
        )
        sentiment = sent_result.scalar_one_or_none()
        if sentiment:
            resp["sentiment"] = {
                "model_name": sentiment.model_name,
                "sentiment_label": sentiment.sentiment_label,
                "sentiment_score": float(sentiment.sentiment_score) if sentiment.sentiment_score else None,
                "confidence": float(sentiment.confidence) if sentiment.confidence else None,
                "entities": sentiment.entities or [],
                "sectors": sentiment.sectors or [],
                "tickers_extracted": sentiment.tickers_extracted or [],
            }

        responses.append(resp)

    return responses


@router.get("/media/stats", response_model=MediaStatsResponse)
async def media_stats(db: AsyncSession = Depends(get_db)) -> dict:
    # Total count
    total_result = await db.execute(select(func.count(MediaContent.id)))
    total = total_result.scalar() or 0

    # Count by source type
    type_result = await db.execute(
        select(MediaContent.source_type, func.count(MediaContent.id))
        .group_by(MediaContent.source_type)
    )
    by_source = {row[0]: row[1] for row in type_result.all()}

    # Last 7 days count
    week_ago = date.today() - timedelta(days=7)
    recent_result = await db.execute(
        select(func.count(MediaContent.id))
        .where(MediaContent.published_date >= week_ago)
    )
    content_7d = recent_result.scalar() or 0

    # Recent sentiment averages
    sent_result = await db.execute(
        select(
            SentimentAnalysis.sentiment_label,
            func.count(SentimentAnalysis.id),
        )
        .join(MediaContent, SentimentAnalysis.media_content_id == MediaContent.id)
        .where(MediaContent.published_date >= week_ago)
        .group_by(SentimentAnalysis.sentiment_label)
    )
    recent_sentiment = {row[0]: row[1] for row in sent_result.all() if row[0]}

    return {
        "total_content": total,
        "by_source_type": by_source,
        "content_last_7_days": content_7d,
        "recent_sentiment": recent_sentiment,
    }


@router.get("/media/{media_id}", response_model=MediaContentDetailResponse)
async def get_media_content(
    media_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    result = await db.execute(
        select(MediaContent).where(MediaContent.id == media_id)
    )
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Media content not found")

    # Get all sentiment analyses
    sent_result = await db.execute(
        select(SentimentAnalysis)
        .where(SentimentAnalysis.media_content_id == media_id)
    )
    analyses = sent_result.scalars().all()

    return {
        "id": item.id,
        "source_type": item.source_type,
        "title": item.title,
        "content": item.content,
        "summary": item.summary,
        "url": item.url,
        "author": item.author,
        "published_date": item.published_date,
        "tickers_mentioned": item.tickers_mentioned or [],
        "member_bioguide_ids": item.member_bioguide_ids or [],
        "raw_metadata": item.raw_metadata or {},
        "created_at": item.created_at,
        "sentiment": None,
        "sentiment_analyses": [
            {
                "model_name": a.model_name,
                "sentiment_label": a.sentiment_label,
                "sentiment_score": float(a.sentiment_score) if a.sentiment_score else None,
                "confidence": float(a.confidence) if a.confidence else None,
                "entities": a.entities or [],
                "sectors": a.sectors or [],
                "tickers_extracted": a.tickers_extracted or [],
            }
            for a in analyses
        ],
    }


@router.get(
    "/members/{bioguide_id}/media",
    response_model=list[MediaContentResponse],
)
async def member_media(
    bioguide_id: str,
    source_type: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    query = (
        select(MediaContent)
        .where(MediaContent.member_bioguide_ids.contains([bioguide_id]))
        .order_by(MediaContent.published_date.desc())
    )
    if source_type:
        query = query.where(MediaContent.source_type == source_type)
    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    items = result.scalars().all()

    return [
        {
            "id": item.id,
            "source_type": item.source_type,
            "title": item.title,
            "summary": item.summary,
            "url": item.url,
            "author": item.author,
            "published_date": item.published_date,
            "tickers_mentioned": item.tickers_mentioned or [],
            "member_bioguide_ids": item.member_bioguide_ids or [],
            "created_at": item.created_at,
            "sentiment": None,
        }
        for item in items
    ]


@router.get(
    "/members/{bioguide_id}/sentiment-timeline",
    response_model=MemberSentimentTimelineResponse,
)
async def member_sentiment_timeline(
    bioguide_id: str,
    days: int = Query(default=90, le=365),
    db: AsyncSession = Depends(get_db),
) -> dict:
    # Get member name
    member_result = await db.execute(
        select(CongressMember.full_name)
        .where(CongressMember.bioguide_id == bioguide_id)
    )
    member_name = member_result.scalar_one_or_none()

    start_date = date.today() - timedelta(days=days)

    # Get daily sentiment averages for this member
    result = await db.execute(
        select(
            MediaContent.published_date,
            func.avg(SentimentAnalysis.sentiment_score),
            func.count(SentimentAnalysis.id),
        )
        .join(SentimentAnalysis, SentimentAnalysis.media_content_id == MediaContent.id)
        .where(MediaContent.member_bioguide_ids.contains([bioguide_id]))
        .where(MediaContent.published_date >= start_date)
        .where(MediaContent.published_date.is_not(None))
        .group_by(MediaContent.published_date)
        .order_by(MediaContent.published_date)
    )

    timeline = []
    for row in result.all():
        if row[0] is not None:
            timeline.append({
                "date": row[0],
                "avg_score": round(float(row[1] or 0), 4),
                "count": row[2],
                "sources": [],
            })

    return {
        "bioguide_id": bioguide_id,
        "member_name": member_name,
        "timeline": timeline,
    }
