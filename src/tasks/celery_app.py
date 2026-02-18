from celery import Celery
from celery.schedules import crontab

from src.config import settings

celery_app = Celery(
    "congress_predictions",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="US/Eastern",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

celery_app.conf.beat_schedule = {
    "collect-house-trades-every-6h": {
        "task": "src.tasks.ingestion_tasks.collect_house_trades",
        "schedule": crontab(minute=0, hour="*/6"),
    },
    "collect-senate-trades-every-6h": {
        "task": "src.tasks.ingestion_tasks.collect_senate_trades",
        "schedule": crontab(minute=15, hour="*/6"),
    },
    "collect-fmp-house-every-6h": {
        "task": "src.tasks.ingestion_tasks.collect_fmp_house_trades",
        "schedule": crontab(minute=30, hour="*/6"),
    },
    "collect-fmp-senate-every-6h": {
        "task": "src.tasks.ingestion_tasks.collect_fmp_senate_trades",
        "schedule": crontab(minute=45, hour="*/6"),
    },
    "collect-market-data-daily": {
        "task": "src.tasks.ingestion_tasks.collect_market_data",
        "schedule": crontab(minute=30, hour=16),  # 4:30 PM ET
    },
    # Phase 2: Legislative data
    "collect-members-daily": {
        "task": "src.tasks.legislation_tasks.collect_members",
        "schedule": crontab(minute=0, hour=6),  # 6 AM ET daily
    },
    "collect-bills-daily": {
        "task": "src.tasks.legislation_tasks.collect_bills",
        "schedule": crontab(minute=15, hour=6),
    },
    "collect-committees-weekly": {
        "task": "src.tasks.legislation_tasks.collect_committees",
        "schedule": crontab(minute=0, hour=5, day_of_week="sunday"),
    },
    "collect-hearings-weekly": {
        "task": "src.tasks.legislation_tasks.collect_hearings",
        "schedule": crontab(minute=30, hour=5, day_of_week="sunday"),
    },
    "collect-voteview-weekly": {
        "task": "src.tasks.legislation_tasks.collect_voteview_scores",
        "schedule": crontab(minute=0, hour=4, day_of_week="sunday"),
    },
    # Phase 3: Network data
    "collect-lobbying-weekly": {
        "task": "src.tasks.network_tasks.collect_lobbying_filings",
        "schedule": crontab(minute=0, hour=3, day_of_week="sunday"),
    },
    "collect-campaign-committees-weekly": {
        "task": "src.tasks.network_tasks.collect_campaign_committees",
        "schedule": crontab(minute=30, hour=3, day_of_week="sunday"),
    },
    "resolve-entities-daily": {
        "task": "src.tasks.network_tasks.resolve_entities",
        "schedule": crontab(minute=0, hour=7),  # 7 AM ET daily, after data collection
    },
    "sync-graph-daily": {
        "task": "src.tasks.network_tasks.sync_graph",
        "schedule": crontab(minute=30, hour=7),  # 7:30 AM ET daily, after entity resolution
    },
    # Phase 4: Media content
    "collect-hearing-transcripts-weekly": {
        "task": "src.tasks.media_tasks.collect_hearing_transcripts",
        "schedule": crontab(minute=0, hour=8, day_of_week="sunday"),  # Sunday 8 AM ET
    },
    "collect-youtube-transcripts-daily": {
        "task": "src.tasks.media_tasks.collect_youtube_transcripts",
        "schedule": crontab(minute=0, hour=9),  # 9 AM ET daily
    },
    "collect-news-articles-every-6h": {
        "task": "src.tasks.media_tasks.collect_news_articles",
        "schedule": crontab(minute=0, hour="*/6"),  # Every 6h
    },
    "collect-congress-rss-every-3h": {
        "task": "src.tasks.media_tasks.collect_congress_rss",
        "schedule": crontab(minute=0, hour="*/3"),  # Every 3h
    },
    "collect-press-releases-daily": {
        "task": "src.tasks.media_tasks.collect_press_releases",
        "schedule": crontab(minute=0, hour=10),  # 10 AM ET daily
    },
    "run-nlp-analysis-daily": {
        "task": "src.tasks.media_tasks.run_nlp_analysis",
        "schedule": crontab(minute=0, hour=11),  # 11 AM ET daily, after collections
    },
    # Phase 5: ML prediction engine
    "run-batch-predictions-daily": {
        "task": "src.tasks.ml_tasks.run_batch_predictions",
        "schedule": crontab(minute=0, hour=12),  # 12 PM ET daily, after NLP
    },
    "train-all-models-weekly": {
        "task": "src.tasks.ml_tasks.train_all_models",
        "schedule": crontab(minute=0, hour=14, day_of_week="sunday"),  # Sunday 2 PM ET
    },
    "backfill-actual-returns-daily": {
        "task": "src.tasks.ml_tasks.backfill_actual_returns",
        "schedule": crontab(minute=0, hour=17),  # 5 PM ET, after market close
    },
    # Phase 6: Signal generation
    "generate-signals-daily": {
        "task": "src.tasks.signal_tasks.generate_signals",
        "schedule": crontab(minute=0, hour=13),  # 1 PM ET, after predictions
    },
    "expire-signals-daily": {
        "task": "src.tasks.signal_tasks.expire_signals",
        "schedule": crontab(minute=0, hour=3),  # 3 AM ET
    },
    "dispatch-alerts-every-30min": {
        "task": "src.tasks.signal_tasks.dispatch_alerts",
        "schedule": crontab(minute="*/30"),  # Every 30 minutes
    },
}

celery_app.autodiscover_tasks(["src.tasks"])
