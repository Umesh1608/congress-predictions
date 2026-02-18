"""YouTube transcript collector.

Fetches transcripts from congressional-related YouTube channels using
the youtube-transcript-api library (no API key required). Discovers
recent videos via public RSS feeds from YouTube channels.

Key channels: C-SPAN, committee channels, member channels.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any
from xml.etree import ElementTree

from src.ingestion.base import BaseCollector, RateLimiter

logger = logging.getLogger(__name__)

# YouTube RSS feed template
YT_RSS_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

# Curated list of congressional YouTube channels
# Format: {channel_id: channel_name}
DEFAULT_CHANNELS: dict[str, str] = {
    "UCb--64Gl51jIEVE-GLDAVTg": "C-SPAN",
    "UCY4sMEDJqVFBJP6n1jUvV6A": "Senate Judiciary Committee",
    "UCPBr2CJOrmLOsMfN-vIpTcg": "House Financial Services Committee",
    "UC6Z4cMDm-6xSdXfYEvMZfuA": "Senate Finance Committee",
    "UCMJTbGLl3u_gp6mPeHSK-yg": "Senate Banking Committee",
    "UCsSCVWFT9_mWRiNHqLbZp_A": "House Ways and Means Committee",
}

_yt_rate_limiter = RateLimiter(max_calls=2, period_seconds=1.0)


class YouTubeTranscriptCollector(BaseCollector):
    """Collect transcripts from congressional YouTube channels.

    Uses youtube-transcript-api (free, no API key) for transcripts
    and YouTube RSS feeds to discover recent video IDs.
    """

    source_name = "youtube"
    rate_limiter = _yt_rate_limiter

    def __init__(
        self,
        channels: dict[str, str] | None = None,
        max_videos_per_channel: int = 5,
    ) -> None:
        super().__init__()
        self.channels = channels or DEFAULT_CHANNELS
        self.max_videos_per_channel = max_videos_per_channel

    async def collect(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for channel_id, channel_name in self.channels.items():
            try:
                videos = await self._fetch_channel_videos(channel_id, channel_name)
                results.extend(videos[: self.max_videos_per_channel])
            except Exception:
                logger.exception(
                    "[%s] Failed to fetch videos for channel %s",
                    self.source_name,
                    channel_name,
                )

        logger.info("[%s] Collected %d video entries", self.source_name, len(results))
        return results

    async def _fetch_channel_videos(
        self, channel_id: str, channel_name: str
    ) -> list[dict[str, Any]]:
        """Fetch recent video metadata from a channel's RSS feed."""
        url = YT_RSS_URL.format(channel_id=channel_id)
        try:
            response = await self.client.get(url)
            response.raise_for_status()
        except Exception:
            logger.warning("[%s] Failed to fetch RSS for %s", self.source_name, channel_name)
            return []

        videos = []
        try:
            root = ElementTree.fromstring(response.text)
            ns = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}

            for entry in root.findall("atom:entry", ns):
                video_id_el = entry.find("yt:videoId", ns)
                title_el = entry.find("atom:title", ns)
                published_el = entry.find("atom:published", ns)

                if video_id_el is None or video_id_el.text is None:
                    continue

                video = {
                    "video_id": video_id_el.text,
                    "title": title_el.text if title_el is not None else "",
                    "published": published_el.text if published_el is not None else "",
                    "channel_id": channel_id,
                    "channel_name": channel_name,
                }

                # Try to fetch transcript
                transcript = await self._fetch_transcript(video_id_el.text)
                video["transcript"] = transcript
                videos.append(video)

        except ElementTree.ParseError:
            logger.warning("[%s] Failed to parse RSS for %s", self.source_name, channel_name)

        return videos

    async def _fetch_transcript(self, video_id: str) -> str:
        """Fetch transcript for a video using youtube-transcript-api."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join(segment["text"] for segment in transcript_list)
        except Exception:
            logger.debug("No transcript available for video %s", video_id)
            return ""

    def transform(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        video_id = raw.get("video_id", "")
        if not video_id:
            return None

        title = raw.get("title", "")
        transcript = raw.get("transcript", "")

        # Parse published date
        published_date = None
        pub_str = raw.get("published", "")
        if pub_str:
            try:
                published_date = datetime.fromisoformat(
                    pub_str.replace("Z", "+00:00")
                ).date()
            except (ValueError, TypeError):
                pass

        return {
            "source_type": "youtube",
            "source_id": video_id,
            "title": title,
            "content": transcript,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "published_date": published_date,
            "raw_metadata": {
                "channel_id": raw.get("channel_id", ""),
                "channel_name": raw.get("channel_name", ""),
                "has_transcript": bool(transcript),
            },
        }
