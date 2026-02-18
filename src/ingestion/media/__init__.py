"""Media content collectors for Phase 4.

Collects hearing transcripts, YouTube interviews, news articles,
press releases, and (when API key is configured) tweets.
"""

from src.ingestion.media.congress_rss import CongressRSSCollector
from src.ingestion.media.gnews import GNewsCollector
from src.ingestion.media.govinfo_hearings import GovInfoHearingCollector
from src.ingestion.media.newsdata import NewsDataCollector
from src.ingestion.media.press_releases import PressReleaseCollector
from src.ingestion.media.twitter import TwitterCollector
from src.ingestion.media.youtube import YouTubeTranscriptCollector

__all__ = [
    "CongressRSSCollector",
    "GNewsCollector",
    "GovInfoHearingCollector",
    "NewsDataCollector",
    "PressReleaseCollector",
    "TwitterCollector",
    "YouTubeTranscriptCollector",
]
