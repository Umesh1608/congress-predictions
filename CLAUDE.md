# Congress Predictions

Congressional trade prediction system that tracks trades by congress/senate members, their families, staff, and associates — then fuses trade data, legislative activity, social media sentiment, and relationship networks to generate ML-powered trading signals.

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m virtualenv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"        # Core + dev tools
pip install -e ".[ml]"         # Add ML dependencies (Phase 5+)
pip install -e ".[dashboard]"  # Add dashboard dependencies (Phase 7+)

# 3. Set up environment
cp .env.example .env
# Edit .env — add your FMP_API_KEY at minimum

# 4. Start infrastructure
docker compose up -d postgres redis neo4j

# 5. Create database tables and seed historical data
python -m scripts.seed_initial_data              # Full backfill (trades + market data)
python -m scripts.seed_initial_data --skip-market-data  # Trades only (faster)

# 5b. Backfill House Clerk data (multi-year, runs in background)
nohup python3 -m scripts.backfill_house_clerk --start 2020 > backfill.log 2>&1 &
# Or specific years:
python -m scripts.backfill_house_clerk --years 2023 2024

# 6. Start the API
uvicorn src.main:app --reload
# API docs at http://localhost:8000/docs
# Prometheus metrics at http://localhost:8000/metrics

# 7. Start background workers (optional, for scheduled collection)
celery -A src.tasks.celery_app worker --loglevel=info --concurrency=4
celery -A src.tasks.celery_app beat --loglevel=info

# 8. Start the dashboard
streamlit run dashboard/app.py
# Dashboard at http://localhost:8501

# Full Docker deployment (all services including dashboard)
docker compose up
```

## Commands

```bash
# Tests
python -m pytest tests/ -v                      # All tests
python -m pytest tests/unit/ -v                  # Unit tests only
python -m pytest tests/unit/test_ingestion/ -v   # Ingestion tests only

# Linting
ruff check src/ tests/
ruff format src/ tests/

# Type checking
mypy src/

# Database migrations
alembic revision --autogenerate -m "description"
alembic upgrade head
```

---

## Architecture

```
[Data Sources] → [Celery Ingestion Workers] → [PostgreSQL + Neo4j]
                                                       ↓
                                              [Processing Layer]
                                              (features, NLP, graph)
                                                       ↓
                                              [ML Prediction Layer]
                                              (GNN, LightGBM, ensemble)
                                                       ↓
                                              [Signal Generation]
                                                       ↓
                                              [FastAPI + Streamlit Dashboard]
```

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy 2.0 (async), PostgreSQL 16, Neo4j 5, Redis 7, Celery 5, LightGBM, PyTorch Geometric, HuggingFace Transformers, Streamlit, Docker

---

## Project Structure

```
congress_predictions/
├── docker-compose.yml              # postgres, redis, neo4j, api, celery-worker, celery-beat, dashboard
├── Dockerfile                      # Python 3.11-slim multi-stage build
├── pyproject.toml                  # hatchling build, deps, tool config
├── alembic.ini                     # Migration config (uses DATABASE_URL from settings)
├── .env.example                    # All env vars with defaults
│
├── alembic/
│   ├── env.py                      # Async Alembic env (imports all models via src/models/__init__)
│   ├── script.py.mako              # Migration template
│   └── versions/                   # Generated migrations
│
├── src/
│   ├── config.py                   # Settings (pydantic-settings, reads .env)
│   ├── main.py                     # FastAPI app factory, CORS, middleware, Prometheus, router registration
│   ├── logging_config.py           # Phase 8: JSON structured logging, correlation IDs
│   ├── metrics.py                  # Phase 8: Prometheus counters, histograms, gauges
│   │
│   ├── db/
│   │   ├── postgres.py             # engine, async_session_factory, Base, get_session()
│   │   ├── redis.py                # redis_pool, get_redis()
│   │   └── neo4j.py                # async Neo4j driver, get_neo4j_session(), verify_connectivity()
│   │
│   ├── models/                     # SQLAlchemy ORM models
│   │   ├── __init__.py             # Re-exports all models (important for Alembic)
│   │   ├── member.py               # CongressMember (+nominate_dim1/2), MemberFamily, MemberStaff, CommitteeAssignment
│   │   ├── trade.py                # TradeDisclosure (dedup index on member+ticker+date+type+amount)
│   │   ├── financial.py            # StockDaily (composite PK: ticker+date)
│   │   ├── legislation.py          # Bill, BillCosponsor, Committee, CommitteeHearing, VoteRecord
│   │   ├── lobbying.py             # LobbyingFiling, LobbyingRegistrant, LobbyingClient, LobbyingLobbyist
│   │   ├── campaign_finance.py     # CampaignCommittee, CampaignContribution
│   │   ├── media.py                # MediaContent, SentimentAnalysis, MemberMediaMention
│   │   ├── ml.py                   # Phase 5: MLModelArtifact, TradePrediction
│   │   └── signal.py               # Phase 6: Signal, AlertConfig
│   │
│   ├── schemas/                    # Pydantic response/request models
│   │   ├── member.py               # MemberResponse, MemberDetailResponse
│   │   ├── trade.py                # TradeResponse, TradeListParams, TradeStatsResponse
│   │   ├── legislation.py          # BillResponse, CommitteeResponse, etc.
│   │   ├── network.py              # MemberNetworkResponse, PathsResponse, SuspiciousTriangleResponse
│   │   ├── media.py                # MediaContentResponse, SentimentResponse, MediaStatsResponse
│   │   ├── predictions.py          # Phase 5: PredictionResponse, ModelPerformanceResponse, LeaderboardEntry
│   │   └── signals.py              # Phase 6: SignalResponse, AlertConfigCreate/Response, SignalStatsResponse
│   │
│   ├── api/
│   │   ├── deps.py                 # get_db() dependency
│   │   ├── middleware.py           # Phase 8: CorrelationId, RateLimit, ApiKey middleware
│   │   ├── cache.py                # Phase 8: In-memory TTL cache with @cached decorator
│   │   └── v1/
│   │       ├── members.py          # /members endpoints
│   │       ├── trades.py           # /trades endpoints
│   │       ├── legislation.py      # /bills, /committees, /hearings endpoints
│   │       ├── network.py          # /network/* endpoints (Neo4j graph queries)
│   │       ├── media.py            # /media/* endpoints (content + sentiment)
│   │       ├── predictions.py      # Phase 5: /predictions endpoints
│   │       ├── signals.py          # Phase 6: /signals, /alerts/configs endpoints
│   │       └── health.py           # Phase 8: /health/detailed, /health/legal
│   │
│   ├── ingestion/
│   │   ├── base.py                 # RateLimiter (asyncio.Lock for concurrency), BaseCollector (abstract: collect/transform/run)
│   │   ├── loader.py               # All upsert functions (trades, stock, members, bills, lobbying, campaign finance, media)
│   │   ├── trades/
│   │   │   ├── house_watcher.py    # HouseWatcherCollector (free S3 JSON — currently 403)
│   │   │   ├── senate_watcher.py   # SenateWatcherCollector (free S3 JSON — currently 403)
│   │   │   ├── house_clerk.py      # HouseClerkCollector (free, scrapes PTR PDFs from clerk.house.gov)
│   │   │   ├── github_senate.py    # GitHubSenateCollector (free, CSV from jeremiak GitHub dataset)
│   │   │   └── fmp_client.py       # FMPHouseCollector, FMPSenateCollector (paid API)
│   │   ├── market/
│   │   │   └── yahoo_finance.py    # fetch_stock_history() via yfinance
│   │   ├── legislation/
│   │   │   ├── congress_gov.py     # Members, Bills, Committees, Hearings collectors
│   │   │   └── voteview.py         # DW-NOMINATE ideology scores
│   │   ├── network/
│   │   │   ├── lobbying.py         # LobbyingFilingCollector (Senate LDA API)
│   │   │   └── campaign_finance.py # FECCommitteeCollector, FECContributionCollector
│   │   └── media/                  # Phase 4: Media collectors
│   │       ├── govinfo_hearings.py # GovInfoHearingCollector (free, hearing transcripts)
│   │       ├── youtube.py          # YouTubeTranscriptCollector (free, no API key)
│   │       ├── gnews.py            # GNewsCollector (free 100 req/day)
│   │       ├── newsdata.py         # NewsDataCollector (free 2000 articles/day)
│   │       ├── congress_rss.py     # CongressRSSCollector (free, no key)
│   │       ├── press_releases.py   # PressReleaseCollector (free, member RSS feeds)
│   │       └── twitter.py          # TwitterCollector (stub, skip-if-no-key)
│   │
│   ├── tasks/
│   │   ├── celery_app.py           # Celery app, beat schedule config (27 scheduled tasks), explicit conf.include
│   │   ├── ingestion_tasks.py      # Trade + market data tasks (with S3→fallback logic)
│   │   ├── legislation_tasks.py    # Legislative data tasks
│   │   ├── network_tasks.py        # Lobbying, campaign finance, entity resolution, graph sync tasks
│   │   ├── media_tasks.py          # Media collection + NLP analysis tasks
│   │   ├── ml_tasks.py             # Phase 5: ML training, batch prediction, return backfill
│   │   └── signal_tasks.py         # Phase 6: Signal generation, expiration, alert dispatch
│   │
│   ├── graph/
│   │   ├── schema.py               # Neo4j constraints, indexes, node/relationship type definitions
│   │   ├── sync.py                 # PostgreSQL → Neo4j full sync (members, trades, bills, lobbying, campaign finance)
│   │   └── queries.py              # Cypher queries: member_network, paths, suspicious_triangles, overlap
│   ├── processing/
│   │   ├── timing_analysis.py      # Trade-legislation correlation analysis
│   │   ├── normalizer.py           # Entity resolution: fuzzy match lobbying clients/employers to tickers
│   │   └── text_processing.py      # NLP pipeline: FinBERT sentiment, spaCy NER, ticker extraction
│   ├── ml/                         # Phase 5: ML prediction engine
│   │   ├── features.py             # FeatureBuilder: 6 feature groups (trade, member, market, legislative, sentiment, network)
│   │   ├── dataset.py              # TemporalSplitter (walk-forward CV) + legacy DatasetBuilder (per-trade)
│   │   ├── dataset_fast.py         # Bulk SQL dataset builder (build_dataset_fast) — 100x faster, 68 features
│   │   ├── training.py             # ModelTrainer orchestrator (temporal CV, CatBoost, Optuna tuning, artifact saving)
│   │   ├── predictor.py            # PredictionService (lazy-loaded singleton, batch predict)
│   │   ├── evaluation.py           # Classifier/regressor/profit metrics
│   │   └── models/
│   │       ├── base.py             # BasePredictor ABC (train/predict/predict_proba/save/load)
│   │       ├── trade_predictor.py  # LightGBM classifier (profitable at horizon)
│   │       ├── return_predictor.py # XGBoost regressor (expected return)
│   │       ├── anomaly_model.py    # Isolation Forest (unusual patterns, market features excluded)
│   │       └── ensemble.py         # Logistic Regression meta-learner (stacking)
│   └── signals/                    # Phase 6: Signal generation
│       ├── generator.py            # SignalGenerator: 4 signal types (trade_follow, anomaly, sentiment_divergence, insider_cluster)
│       ├── scorer.py               # Composite signal scoring (freshness, lag, corroboration, cluster bonuses)
│       └── alerting.py             # Alert dispatch + rate limiting (10/hour/config) + webhook
│
├── scripts/
│   ├── seed_initial_data.py        # Historical backfill: creates tables, loads trades + market data
│   └── backfill_house_clerk.py     # Multi-year House Clerk backfill (streaming inserts, 10 req/sec)
│
├── tests/
│   ├── conftest.py                 # Fixtures: sample_house_trade_raw, sample_senate_trade_raw
│   ├── unit/
│   │   ├── test_ingestion/
│   │   │   ├── test_house_watcher.py   # 8 tests
│   │   │   ├── test_senate_watcher.py  # 3 tests
│   │   │   ├── test_congress_gov.py    # 14 tests
│   │   │   └── test_voteview.py        # 5 tests
│   │   ├── test_trades/
│   │   │   ├── test_house_clerk.py     # 16 tests (HTML parsing, PDF text parsing, transform, concurrency)
│   │   │   └── test_github_senate.py   # 11 tests (transform, fallback, edge cases)
│   │   ├── test_processing/
│   │   │   └── test_timing_analysis.py # 8 tests
│   │   ├── test_network/
│   │   │   ├── test_normalizer.py      # 12 tests: entity resolution, fuzzy matching
│   │   │   ├── test_lobbying.py        # 20 tests: filing transform, bill extraction, revolving door
│   │   │   └── test_campaign_finance.py # 8 tests: FEC committee/contribution transforms
│   │   ├── test_media/
│   │   │   ├── test_govinfo_hearings.py # 12 tests
│   │   │   ├── test_youtube.py          # 6 tests
│   │   │   ├── test_gnews.py            # 5 tests
│   │   │   ├── test_newsdata.py         # 5 tests
│   │   │   ├── test_congress_rss.py     # 10 tests
│   │   │   ├── test_press_releases.py   # 6 tests
│   │   │   ├── test_twitter.py          # 5 tests
│   │   │   └── test_text_processing.py  # 20 tests
│   │   ├── test_ml/                     # Phase 5: 40 tests
│   │   │   ├── test_features.py         # 10 tests: feature groups, missing values
│   │   │   ├── test_dataset.py          # 5 tests: temporal splitter, label computation
│   │   │   ├── test_models.py           # 15 tests: each model train/predict with synthetic data
│   │   │   ├── test_evaluation.py       # 6 tests: metric computation
│   │   │   └── test_predictor.py        # 4 tests: prediction service with mocked models
│   │   ├── test_signals/                # Phase 6: 19 tests
│   │   │   ├── test_scorer.py           # 7 tests: composite scoring, lag penalty, cap
│   │   │   ├── test_alerting.py         # 7 tests: config matching, rate limiting
│   │   │   └── test_models.py           # 3 tests: signal/alert model creation
│   │   ├── test_dashboard/              # Phase 7: 16 tests
│   │   │   ├── test_api_client.py       # 5 tests: cache behavior, URLs
│   │   │   └── test_charts.py           # 11 tests: chart functions return valid Plotly figures
│   │   └── test_production/             # Phase 8: 24 tests
│   │       ├── test_middleware.py        # 8 tests: rate limiting, API key validation
│   │       ├── test_cache.py            # 9 tests: cache hit/miss, TTL, invalidation, decorator
│   │       └── test_health.py           # 7 tests: service health checks
│   └── integration/                # Empty placeholder
│
├── notebooks/                      # Jupyter exploration (empty)
└── dashboard/                      # Phase 7: Streamlit dashboard
    ├── app.py                      # Main entry + page navigation
    ├── api_client.py               # Sync httpx wrapper calling FastAPI backend
    ├── charts.py                   # Reusable Plotly chart functions (5 chart types)
    └── pages/
        ├── overview.py             # KPI cards, recent trades, signal strength
        ├── member_explorer.py      # Member drill-down: trades, sentiment, predictions
        ├── trade_feed.py           # Filterable trade list with legislative context
        ├── network_graph.py        # Interactive pyvis graph visualization
        ├── signals.py              # Active signals + alert config management
        └── backtesting.py          # Model performance + leaderboard
```

---

## Database Schema

### congress_member
| Column | Type | Notes |
|---|---|---|
| bioguide_id | String(10) | **PK** |
| full_name | String(200) | |
| first_name | String(100) | nullable |
| last_name | String(100) | nullable |
| chamber | String(10) | "house" / "senate" |
| state | String(2) | nullable |
| district | String(10) | nullable |
| party | String(50) | nullable |
| in_office | Boolean | default: true |
| first_elected | Integer | nullable |
| social_accounts | JSONB | nullable, default: {} |
| created_at | DateTime(tz) | server default: now() |
| updated_at | DateTime(tz) | auto-updates |

### trade_disclosure
| Column | Type | Notes |
|---|---|---|
| id | Integer | **PK** auto |
| member_bioguide_id | String(10) | FK → congress_member, nullable, indexed |
| member_name | String(200) | always set (for records without bioguide match) |
| filer_type | String(50) | "member" / "spouse" / "dependent" / "joint" |
| ticker | String(20) | nullable, indexed |
| asset_name | String(500) | |
| asset_type | String(100) | nullable |
| transaction_type | String(50) | "purchase" / "sale" / "sale_full" / "sale_partial" / "exchange" |
| transaction_date | Date | indexed, **when the trade actually happened** |
| disclosure_date | Date | nullable, indexed, **when the filing was published** |
| amount_range_low | Numeric(15,2) | nullable |
| amount_range_high | Numeric(15,2) | nullable |
| chamber | String(10) | "house" / "senate" |
| source | String(50) | "house_watcher" / "senate_watcher" / "house_clerk" / "github_senate" / "fmp_house" / "fmp_senate" |
| filing_url | Text | nullable |
| raw_data | JSONB | original source record |
| created_at | DateTime(tz) | server default: now() |
| **Unique index** | ix_trade_dedup | (member_name, ticker, transaction_date, transaction_type, amount_range_low) |

### stock_daily
| Column | Type | Notes |
|---|---|---|
| ticker | String(20) | **Composite PK** |
| date | Date | **Composite PK** |
| open, high, low, close, adj_close | Numeric(12,4) | nullable |
| volume | Integer | nullable |
| created_at | DateTime(tz) | |

### member_family
| Column | Type | Notes |
|---|---|---|
| id | Integer | PK auto |
| member_bioguide_id | String(10) | FK, indexed |
| name | String(200) | |
| relationship_type | String(50) | "spouse" / "dependent" |
| known_employers | JSONB | nullable |

### member_staff
| Column | Type | Notes |
|---|---|---|
| id | Integer | PK auto |
| member_bioguide_id | String(10) | FK, indexed |
| name, title | String | |
| start_date, end_date | Date | nullable |
| subsequent_employer | String(300) | for revolving door tracking |

### committee_assignment
| Column | Type | Notes |
|---|---|---|
| id | Integer | PK auto |
| member_bioguide_id | String(10) | FK, indexed |
| committee_code | String(20) | |
| committee_name | String(300) | |
| role | String(50) | "chair" / "ranking" / "member" |
| congress_number | Integer | nullable |

### lobbying_filing (Phase 3)
| Column | Type | Notes |
|---|---|---|
| id | Integer | PK auto |
| filing_uuid | String(50) | **unique**, indexed |
| filing_type | String(50) | "registration" / "report" |
| filing_year | Integer | |
| filing_period | String(20) | Q1/Q2/H1/H2 |
| filing_date | Date | nullable |
| amount | Numeric(15,2) | nullable |
| registrant_id | Integer | FK → lobbying_registrant |
| client_id | Integer | FK → lobbying_client |
| specific_issues, general_issue_codes, government_entities, lobbied_bills | JSONB | |

### lobbying_registrant (Phase 3)
| Column | Type | Notes |
|---|---|---|
| id | Integer | PK auto |
| senate_id | String(50) | **unique**, indexed |
| name | String(300) | indexed |
| description | Text | nullable |

### lobbying_client (Phase 3)
| Column | Type | Notes |
|---|---|---|
| id | Integer | PK auto |
| name | String(300) | indexed |
| normalized_name | String(300) | indexed, for fuzzy matching |
| matched_ticker | String(20) | indexed, entity resolution result |
| match_confidence | Numeric(5,4) | 0-1 confidence |
| match_method | String(50) | "manual" / "exact" / "fuzzy" / "edgar" |

### lobbying_lobbyist (Phase 3)
| Column | Type | Notes |
|---|---|---|
| id | Integer | PK auto |
| filing_id | Integer | FK → lobbying_filing |
| name | String(200) | |
| covered_position | Text | former government position (revolving door) |
| is_former_congress | Boolean | |
| is_former_executive | Boolean | |

### campaign_committee (Phase 3)
| Column | Type | Notes |
|---|---|---|
| id | Integer | PK auto |
| fec_committee_id | String(20) | **unique**, indexed |
| name | String(300) | |
| committee_type | String(10) | H/S/P |
| candidate_fec_id | String(20) | indexed |
| member_bioguide_id | String(10) | indexed, links to congress_member |

### campaign_contribution (Phase 3)
| Column | Type | Notes |
|---|---|---|
| id | Integer | PK auto |
| fec_transaction_id | String(50) | **unique constraint** |
| committee_id | Integer | FK → campaign_committee |
| contributor_name | String(300) | |
| contributor_employer | String(300) | indexed, key for entity resolution |
| contributor_occupation | String(300) | |
| amount | Numeric(12,2) | |
| contribution_date | Date | indexed |
| matched_ticker | String(20) | indexed, entity resolution result |
| match_confidence | Numeric(5,4) | |

### media_content (Phase 4)
| Column | Type | Notes |
|---|---|---|
| id | Integer | PK auto |
| source_type | String(50) | indexed, `"hearing_transcript"/"youtube"/"gnews"/"newsdata"/"press_release"/"congress_rss"/"twitter"` |
| source_id | String(500) | unique per source_type |
| title | Text | nullable |
| content | Text | nullable, full text/transcript |
| summary | Text | nullable |
| url | Text | nullable |
| author | String(300) | nullable |
| published_date | Date | indexed |
| member_bioguide_ids | JSONB | list of associated member bioguide IDs |
| tickers_mentioned | JSONB | extracted stock tickers |
| raw_metadata | JSONB | source-specific metadata |
| created_at | DateTime(tz) | server default |
| **Unique constraint** | uq_media_content_source | `(source_type, source_id)` |
| **Index** | ix_media_source_type_date | `(source_type, published_date)` |

### sentiment_analysis (Phase 4)
| Column | Type | Notes |
|---|---|---|
| id | Integer | PK auto |
| media_content_id | Integer | FK → media_content, indexed |
| model_name | String(100) | `"finbert"/"vader"` etc. |
| sentiment_label | String(20) | `"positive"/"negative"/"neutral"` |
| sentiment_score | Numeric(5,4) | -1.0 to 1.0 |
| confidence | Numeric(5,4) | 0-1 |
| entities | JSONB | spaCy NER results |
| sectors | JSONB | zero-shot classification |
| tickers_extracted | JSONB | company→ticker mappings |
| created_at | DateTime(tz) | server default |
| **Unique constraint** | uq_sentiment_content_model | `(media_content_id, model_name)` |

### member_media_mention (Phase 4)
| Column | Type | Notes |
|---|---|---|
| id | Integer | PK auto |
| member_bioguide_id | String(10) | FK → congress_member, indexed |
| media_content_id | Integer | FK → media_content, indexed |
| mention_type | String(50) | `"speaker"/"mentioned"/"author"/"subject"` |
| context_snippet | Text | surrounding text of mention |

### ml_model_artifact (Phase 5)
| Column | Type | Notes |
|---|---|---|
| id | Integer | PK auto |
| model_name | String(100) | `"trade_predictor"/"return_predictor"/"anomaly_detector"/"ensemble"` |
| model_version | String(50) | semver or timestamp |
| artifact_path | Text | filesystem path to saved model |
| metrics | JSONB | `{accuracy, precision, recall, f1, auc}` |
| feature_columns | JSONB | list of feature names used |
| training_config | JSONB | hyperparams, data window |
| trained_at | DateTime(tz) | |
| is_active | Boolean | default false, only one active per model_name |
| created_at | DateTime(tz) | server default |
| **Unique constraint** | uq_model_name_version | `(model_name, model_version)` |

### trade_prediction (Phase 5)
| Column | Type | Notes |
|---|---|---|
| id | Integer | PK auto |
| trade_id | Integer | FK → trade_disclosure, indexed |
| model_artifact_id | Integer | FK → ml_model_artifact |
| prediction_type | String(50) | `"will_trade"/"return_direction"/"anomaly"` |
| predicted_value | Numeric(8,4) | probability or score |
| predicted_label | String(50) | `"buy"/"sell"/"anomalous"/"normal"` |
| confidence | Numeric(5,4) | 0-1 |
| feature_vector | JSONB | input features snapshot |
| actual_return_5d | Numeric(8,4) | nullable, backfilled |
| actual_return_21d | Numeric(8,4) | nullable, backfilled |
| created_at | DateTime(tz) | server default |
| **Unique constraint** | uq_trade_model_type | `(trade_id, model_artifact_id, prediction_type)` |

### signal (Phase 6)
| Column | Type | Notes |
|---|---|---|
| id | Integer | PK auto |
| signal_type | String(50) | `"trade_follow"/"anomaly_alert"/"sentiment_divergence"/"insider_cluster"/"network_signal"` |
| member_bioguide_id | String(10) | FK, nullable, indexed |
| ticker | String(20) | nullable, indexed |
| direction | String(10) | `"bullish"/"bearish"/"neutral"` |
| strength | Numeric(5,4) | 0-1 composite score |
| confidence | Numeric(5,4) | 0-1 |
| evidence | JSONB | `{sources: [{type, id, detail}], trade_ids: [], prediction_ids: []}` |
| expires_at | DateTime(tz) | signal validity window |
| is_active | Boolean | default true |
| created_at | DateTime(tz) | server default |
| **Index** | ix_signal_type_active_created | `(signal_type, is_active, created_at DESC)` |

### alert_config (Phase 6)
| Column | Type | Notes |
|---|---|---|
| id | Integer | PK auto |
| name | String(200) | |
| signal_types | JSONB | list of signal_type strings to watch |
| min_strength | Numeric(5,4) | threshold for alerting |
| tickers | JSONB | nullable, filter to specific tickers |
| members | JSONB | nullable, filter to specific members |
| webhook_url | Text | nullable, for webhook dispatch |
| is_active | Boolean | default true |
| created_at | DateTime(tz) | |

---

## Neo4j Graph Schema (Phase 3)

### Node Types
| Label | Key Property | Other Properties |
|---|---|---|
| Member | bioguide_id | full_name, chamber, state, party, in_office, nominate_dim1/2 |
| Company | ticker | name |
| Committee | system_code | name, chamber |
| Bill | bill_id | title, bill_type, congress, policy_area, introduced_date |
| LobbyingFirm | senate_id | name |
| Lobbyist | name | covered_position, is_former_congress |

### Relationship Types
| Relationship | From → To | Properties |
|---|---|---|
| TRADED | Member → Company | trade_id, transaction_type, transaction_date, amount_range_low/high |
| SITS_ON | Member → Committee | role, committee_name |
| SPONSORED | Member → Bill | |
| COSPONSORED | Member → Bill | |
| REFERRED_TO | Bill → Committee | |
| LOBBIED_FOR | LobbyingFirm → Company | amount, filing_count, year |
| LOBBIED | LobbyingFirm → Member | issues |
| EMPLOYED_BY | Lobbyist → LobbyingFirm | |
| FORMERLY_AT | Lobbyist → Member | position (revolving door) |
| DONATED_TO | Company → Member | total_amount, contribution_count, employer_name |
| HEARING_ON | Committee → Bill | date, title |

---

## API Endpoints

Base URL: `/api/v1`

### Members
| Method | Path | Description | Key Params |
|---|---|---|---|
| GET | `/members` | List members | `chamber`, `party`, `state`, `in_office`, `limit`, `offset` |
| GET | `/members/{bioguide_id}` | Member detail | |
| GET | `/members/{bioguide_id}/trades` | Member's trades | `limit`, `offset` |

### Trades
| Method | Path | Description | Key Params |
|---|---|---|---|
| GET | `/trades` | List trades (filterable) | `member_name`, `ticker`, `chamber`, `transaction_type`, `filer_type`, `date_from`, `date_to`, `limit`, `offset` |
| GET | `/trades/stats` | Aggregate stats | returns top tickers, most active members, 7-day count |
| GET | `/trades/{trade_id}/legislative-context` | Legislative context for a trade | `hearing_window` (days ±, default 30), `bill_window` (days ±, default 90) |

### Legislation (Phase 2)
| Method | Path | Description | Key Params |
|---|---|---|---|
| GET | `/bills` | List bills | `congress`, `bill_type`, `policy_area`, `sponsor_bioguide_id`, `date_from`, `date_to` |
| GET | `/bills/{bill_id}` | Bill detail (e.g. `hr1234-118`) | |
| GET | `/committees` | List committees | `chamber`, `current_only` |
| GET | `/committees/{system_code}` | Committee detail (e.g. `HSBA00`) | |
| GET | `/committees/{system_code}/trades` | Trades by committee members | `limit`, `offset` |
| GET | `/committees/{system_code}/hearings` | Committee hearings | `limit`, `offset` |

### Network (Phase 3 — Neo4j)
| Method | Path | Description | Key Params |
|---|---|---|---|
| GET | `/network/stats` | Graph database statistics | |
| GET | `/network/member/{bioguide_id}` | Member's full relationship network | `max_depth` (1-3, default 2) |
| GET | `/network/member/{bioguide_id}/paths-to/{ticker}` | All connection paths to a company | `max_depth` (2-6, default 4) |
| GET | `/network/member/{bioguide_id}/connections` | Summary of trading-related connections | |
| GET | `/network/suspicious-triangles` | Member traded company that lobbied them | `limit` |
| GET | `/network/committee-company-overlap` | Companies lobbied on bills in member's committee | `limit` |

### Media (Phase 4)
| Method | Path | Description | Key Params |
|---|---|---|---|
| GET | `/media` | List media content | `source_type`, `date_from`, `date_to`, `member_bioguide_id`, `ticker`, `limit`, `offset` |
| GET | `/media/stats` | Content counts + sentiment aggregates | |
| GET | `/media/{media_id}` | Full media content with all NLP results | |
| GET | `/members/{bioguide_id}/media` | Member's media timeline | `source_type`, `limit`, `offset` |
| GET | `/members/{bioguide_id}/sentiment-timeline` | Rolling sentiment over time | `days` (default 90, max 365) |

### Predictions (Phase 5)
| Method | Path | Description | Key Params |
|---|---|---|---|
| GET | `/predictions` | List predictions | `member_bioguide_id`, `ticker`, `prediction_type`, `date_from`, `date_to`, `limit`, `offset` |
| GET | `/predictions/model-performance` | Active model metrics | |
| GET | `/predictions/stats` | Prediction counts + accuracy | |
| GET | `/predictions/leaderboard` | Members ranked by accuracy | `limit` |
| GET | `/predictions/{trade_id}` | All predictions for a trade | |

### Signals (Phase 6)
| Method | Path | Description | Key Params |
|---|---|---|---|
| GET | `/signals` | List active signals | `signal_type`, `ticker`, `member_bioguide_id`, `min_strength`, `limit`, `offset` |
| GET | `/signals/stats` | Signal counts by type | |
| GET | `/signals/{signal_id}` | Signal detail with evidence | |
| POST | `/alerts/configs` | Create alert configuration | body: `AlertConfigCreate` |
| GET | `/alerts/configs` | List alert configs | |

### Health (Phase 8 enhanced)
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns `{"status": "ok"}` |
| GET | `/health/detailed` | Checks PostgreSQL, Redis, Neo4j, data freshness |
| GET | `/health/legal` | Financial disclaimer |
| GET | `/metrics` | Prometheus scrape endpoint |

---

## Data Ingestion

### Collector Pattern
All collectors extend `BaseCollector` (in `src/ingestion/base.py`):
- `collect()` → fetches raw records from source
- `transform(raw)` → converts to dict for DB upsert (returns None to skip)
- `run()` → orchestrates collect → transform pipeline
- Built-in: rate limiting (`RateLimiter` with `asyncio.Lock` for concurrent safety), retry with backoff, HTTP 429 handling

### Data Sources (Phase 1)

| Source | Class | URL / API | Cost | Schedule | Status |
|---|---|---|---|---|---|
| House Stock Watcher | `HouseWatcherCollector` | S3 JSON: `house-stock-watcher-data.s3-us-west-2.amazonaws.com` | Free | Every 6h | **403 — down since ~2025** |
| Senate Stock Watcher | `SenateWatcherCollector` | S3 JSON: `senate-stock-watcher-data.s3-us-west-2.amazonaws.com` | Free | Every 6h | **403 — down since ~2025** |
| House Clerk (fallback) | `HouseClerkCollector` | Scrapes PTR PDFs from `disclosures-clerk.house.gov` | Free | Every 6h at :10 | **Active** |
| GitHub Senate CSV (fallback) | `GitHubSenateCollector` | `github.com/jeremiak/us-senate-financial-disclosure-data` | Free | Fallback for Senate | **Active** (2012-2024 data) |
| FMP House | `FMPHouseCollector` | `financialmodelingprep.com/stable/house-disclosure` | $15-30/mo | Every 6h | Supplement (needs API key) |
| FMP Senate | `FMPSenateCollector` | `financialmodelingprep.com/stable/senate-trading` | $15-30/mo | Every 6h | Supplement (needs API key) |
| Yahoo Finance | `fetch_stock_history()` | yfinance library | Free | Daily 4:30 PM ET | **Active** |

**Fallback strategy:** Celery tasks `collect_house_trades` and `collect_senate_trades` try the S3 watchers first. On failure (403), they automatically fall back to `HouseClerkCollector` and `GitHubSenateCollector` respectively. The House Clerk task also runs independently every 6h at :10 for direct scraping.

### Data Sources (Phase 2)

| Source | Class | URL / API | Cost | Schedule |
|---|---|---|---|---|
| Congress.gov Members | `CongressMemberCollector` | `api.congress.gov/v3/member` | Free (5K req/hr) | Daily 6:00 AM |
| Congress.gov Bills | `CongressBillCollector` | `api.congress.gov/v3/bill/{congress}/{type}` | Free | Daily 6:15 AM |
| Congress.gov Committees | `CongressCommitteeCollector` | `api.congress.gov/v3/committee/{chamber}` | Free | Weekly Sunday 5:00 AM |
| Congress.gov Hearings | `CongressHearingCollector` | `api.congress.gov/v3/hearing/{congress}` | Free | Weekly Sunday 5:30 AM |
| Voteview Ideology | `VoteviewCollector` | `voteview.com/static/data/out/members/HSall_members.csv` | Free | Weekly Sunday 4:00 AM |

All Congress.gov collectors share a rate limiter (1 req/sec, well under the 5K/hr limit). API key required: sign up at https://api.data.gov/signup/.

### Data Sources (Phase 3)

| Source | Class | URL / API | Cost | Schedule |
|---|---|---|---|---|
| Senate LDA Lobbying | `LobbyingFilingCollector` | `lda.senate.gov/api/v1/filings/` | Free | Weekly Sunday 3:00 AM |
| FEC Committees | `FECCommitteeCollector` | `api.open.fec.gov/v1/committees/` | Free (API key required) | Weekly Sunday 3:30 AM |
| FEC Contributions | `FECContributionCollector` | `api.open.fec.gov/v1/schedules/schedule_a/` | Free | Via Celery task |

FEC API key: free at https://api.data.gov/signup/ (1000 req/hr). Senate LDA rate limited to 2 req/sec.

### Data Sources (Phase 4)

| Source | Class | URL / API | Cost | Schedule |
|---|---|---|---|---|
| GovInfo Hearings | `GovInfoHearingCollector` | `api.govinfo.gov/search` (CHRG collection) | Free (API key) | Weekly Sunday 8:00 AM |
| YouTube Transcripts | `YouTubeTranscriptCollector` | youtube-transcript-api + YT RSS feeds | Free | Daily 9:00 AM |
| GNews | `GNewsCollector` | `gnews.io/api/v4/search` | Free (100 req/day) | Every 6h |
| NewsData.io | `NewsDataCollector` | `newsdata.io/api/1/news` | Free (2000/day) | Every 6h |
| Congress RSS | `CongressRSSCollector` | `congress.gov/rss/...` (5 feeds) | Free | Every 3h |
| Press Releases | `PressReleaseCollector` | Member website RSS feeds | Free | Daily 10:00 AM |
| X/Twitter | `TwitterCollector` | `api.twitter.com/v2` (stub) | ~$200/mo | Not scheduled (activate with API key) |

All media collectors use skip-if-no-key pattern. GovInfo/GNews/NewsData keys: free at respective sites. YouTube and Congress RSS: no key needed. Twitter: requires paid API ($200+/mo).

### Upsert Logic (`src/ingestion/loader.py`)
- `upsert_trades()` — PostgreSQL `INSERT ON CONFLICT DO NOTHING` using the dedup index
- `upsert_stock_daily()` — same pattern on (ticker, date) composite key
- `get_unique_tickers()` — distinct tickers from trade_disclosure for market data backfill
- `get_existing_filing_urls(session, source)` — returns set of already-processed filing URLs for incremental collection
- `upsert_members()` — `ON CONFLICT DO UPDATE` on bioguide_id (merges new fields)
- `update_member_ideology()` — updates only nominate_dim1/dim2 on existing members
- `upsert_bills()` — `ON CONFLICT DO UPDATE` on bill_id (updates status, actions, subjects)
- `upsert_committees()` — `ON CONFLICT DO UPDATE` on system_code
- `upsert_hearings()` — `ON CONFLICT DO NOTHING` on (committee_code, title, hearing_date)
- `upsert_lobbying_filings()` — inserts filing + nested registrant, client, lobbyists
- `upsert_campaign_committees()` — `ON CONFLICT DO UPDATE` on fec_committee_id
- `upsert_campaign_contributions()` — `ON CONFLICT DO NOTHING` on fec_transaction_id
- `upsert_media_content()` — `ON CONFLICT DO UPDATE` on (source_type, source_id)
- `upsert_sentiment_analyses()` — `ON CONFLICT DO UPDATE` on (media_content_id, model_name)
- `upsert_member_media_mentions()` — `ON CONFLICT DO NOTHING`

### Amount Range Parsing
House/Senate watchers report ranges like `"$15,001 - $50,000"`. These are parsed to `(Decimal("15001"), Decimal("50000"))` via the `AMOUNT_RANGES` lookup dict in `house_watcher.py` (also used by `senate_watcher.py`).

---

## Celery Tasks & Schedule

**Broker/Backend:** Redis (from `REDIS_URL`)
**Timezone:** US/Eastern
**Concurrency:** 4 workers

| Beat Name | Task | Schedule | Notes |
|---|---|---|---|
| `collect-house-trades-every-6h` | `collect_house_trades` | Every 6h at :00 | S3 → House Clerk fallback |
| `collect-house-clerk-trades-every-6h` | `collect_house_clerk_trades` | Every 6h at :10 | Direct clerk.house.gov scrape |
| `collect-senate-trades-every-6h` | `collect_senate_trades` | Every 6h at :15 | S3 → GitHub CSV fallback |
| `collect-fmp-house-every-6h` | `collect_fmp_house_trades` | Every 6h at :30 | Needs FMP_API_KEY |
| `collect-fmp-senate-every-6h` | `collect_fmp_senate_trades` | Every 6h at :45 | Needs FMP_API_KEY |
| `collect-market-data-daily` | `collect_market_data` | Daily at 4:30 PM ET | |
| `collect-members-daily` | `collect_members` | Daily at 6:00 AM ET |
| `collect-bills-daily` | `collect_bills` | Daily at 6:15 AM ET |
| `collect-committees-weekly` | `collect_committees` | Sunday 5:00 AM ET |
| `collect-hearings-weekly` | `collect_hearings` | Sunday 5:30 AM ET |
| `collect-voteview-weekly` | `collect_voteview_scores` | Sunday 4:00 AM ET |
| `collect-lobbying-weekly` | `collect_lobbying_filings` | Sunday 3:00 AM ET |
| `collect-campaign-committees-weekly` | `collect_campaign_committees` | Sunday 3:30 AM ET |
| `resolve-entities-daily` | `resolve_entities` | Daily 7:00 AM ET |
| `sync-graph-daily` | `sync_graph` | Daily 7:30 AM ET |
| `collect-hearing-transcripts-weekly` | `collect_hearing_transcripts` | Sunday 8:00 AM ET |
| `collect-youtube-transcripts-daily` | `collect_youtube_transcripts` | Daily 9:00 AM ET |
| `collect-news-articles-every-6h` | `collect_news_articles` | Every 6h (GNews + NewsData) |
| `collect-congress-rss-every-3h` | `collect_congress_rss` | Every 3h |
| `collect-press-releases-daily` | `collect_press_releases` | Daily 10:00 AM ET |
| `run-nlp-analysis-daily` | `run_nlp_analysis` | Daily 11:00 AM ET |
| `run-batch-predictions-daily` | `run_batch_predictions` | Daily 12:00 PM ET |
| `generate-signals-daily` | `generate_signals` | Daily 1:00 PM ET |
| `train-all-models-weekly` | `train_all_models` | Weekly Sunday 2:00 PM ET |
| `dispatch-alerts-every-30min` | `dispatch_alerts` | Every 30 minutes |
| `expire-signals-daily` | `expire_signals` | Daily 3:00 AM ET |
| `backfill-actual-returns-daily` | `backfill_actual_returns` | Daily 5:00 PM ET |

Tasks use `asyncio.run()` to bridge Celery's sync workers with async collectors/DB operations. Each task creates its own engine/session factory (not shared with the FastAPI app). Task discovery uses explicit `conf.include` (not `autodiscover_tasks`, which doesn't work with non-standard module names like `ingestion_tasks.py`). Twitter task (`collect_tweets`) is defined but NOT in the beat schedule — activate by adding to beat_schedule when API key is available.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | Yes | `postgresql+asyncpg://congress:congress@localhost:5432/congress_predictions` | Async PostgreSQL |
| `DATABASE_URL_SYNC` | Yes | `postgresql://congress:...` | Sync PostgreSQL (for Alembic) |
| `REDIS_URL` | Yes | `redis://localhost:6379/0` | Redis broker |
| `FMP_API_KEY` | No | empty | Financial Modeling Prep key (skipped if empty) |
| `FEC_API_KEY` | No | empty | FEC API key (free at api.data.gov) |
| `NEO4J_URI` | Yes (P3+) | `bolt://localhost:7687` | Neo4j graph database |
| `NEO4J_USER` / `NEO4J_PASSWORD` | Yes (P3+) | `neo4j` / `congress_neo4j` | Neo4j credentials |
| `CONGRESS_GOV_API_KEY` | No | empty | Congress.gov API |
| `TWITTER_BEARER_TOKEN` | No | empty | X/Twitter API (skip if empty, ~$200/mo) |
| `GNEWS_API_KEY` | No | empty | GNews API (free 100 req/day at gnews.io) |
| `NEWSDATA_API_KEY` | No | empty | NewsData.io (free 2000/day at newsdata.io) |
| `GOVINFO_API_KEY` | No | empty | GovInfo API (free at api.data.gov) |
| `APP_ENV` | No | `development` | Enables SQL echo in dev; `production` enforces API key auth |
| `LOG_LEVEL` | No | `INFO` | |
| `RATE_LIMIT_PER_MINUTE` | No | `100` | Anonymous rate limit per IP |
| `RATE_LIMIT_AUTHENTICATED` | No | `1000` | Authenticated rate limit per API key |
| `API_KEYS` | No | empty | Comma-separated API keys (enforced only in production) |
| `API_BASE_URL` | No | `http://localhost:8000/api/v1` | Dashboard → API connection (used by Streamlit) |

---

## Docker Services

| Service | Image | Port | Depends On |
|---|---|---|---|
| `postgres` | postgres:16-alpine | 5432 | — |
| `redis` | redis:7-alpine | 6379 | — |
| `neo4j` | neo4j:5-community | 7474, 7687 | — |
| `api` | Built from Dockerfile | 8000 | postgres, redis, neo4j |
| `celery-worker` | Built from Dockerfile | — | postgres, redis, neo4j |
| `celery-beat` | Built from Dockerfile | — | postgres, redis |
| `dashboard` | Built from Dockerfile | 8501 | api |

All app services mount `.` as a volume for hot-reload during development. Docker Compose overrides `DATABASE_URL`, `REDIS_URL`, `NEO4J_URI`, and `API_BASE_URL` to use container hostnames. Neo4j has APOC plugin enabled and auth configured as `neo4j/congress_neo4j`.

---

## Tests

**Framework:** pytest + pytest-asyncio
**Run:** `python -m pytest tests/ -v`

### Current Tests (277 passing)

**tests/conftest.py** — Shared fixtures:
- `sample_house_trade_raw` — Nancy Pelosi NVDA purchase record
- `sample_senate_trade_raw` — Tommy Tuberville MSFT purchase record

**tests/unit/test_ingestion/test_house_watcher.py** — 8 tests (transform logic, dates, owners, tickers)
**tests/unit/test_ingestion/test_senate_watcher.py** — 3 tests (transform, spouse, missing date)
**tests/unit/test_ingestion/test_congress_gov.py** — 14 tests:
- `TestCongressMemberTransform` (4) — basic member, senator, missing bioguide, comma name format
- `TestCongressBillTransform` (3) — basic bill, no sponsor, missing fields
- `TestCongressCommitteeTransform` (3) — house committee, subcommittee, empty code
- `TestCongressHearingTransform` (3) — basic hearing, associated bills, no title
- `TestCurrentCongress` (1) — congress number sanity check

**tests/unit/test_ingestion/test_voteview.py** — 5 tests (basic record, party mapping, independent, missing bioguide, NaN scores)

**tests/unit/test_trades/test_house_clerk.py** — 16 tests:
- `TestSearchResultParser` (2) — basic table parsing, empty table
- `TestParsePtrPdfText` (3) — basic trade extraction, purchase trade, no trades
- `TestHouseClerkTransform` (7) — purchase/sale transform, exchange type, no date/name returns None, default/custom years
- `TestConcurrentCollect` (3) — concurrent faster than sequential, malformed PDF doesn't crash, semaphore limits concurrency
- `TestRateLimiterConcurrency` (1) — concurrent acquires respect rate limit

**tests/unit/test_trades/test_github_senate.py** — 11 tests:
- Basic transform, sale types, partial sale, no ticker, long ticker nullified, no date, no filer, dependent/joint owner, date formats, raw data preserved

**tests/unit/test_processing/test_timing_analysis.py** — 8 tests (suspicion score: zero baseline, hearing proximity 3d/7d, sector alignment, late disclosure, sponsored bill, cap at 1.0, bill proximity 30d)
**tests/unit/test_network/test_normalizer.py** — 12 tests:
- `TestNormalizeCompanyName` (6) — suffix stripping, empty/None handling, whitespace, punctuation
- `TestMatchNameToTicker` (6) — manual overrides (Google/Amazon/Meta), exact match, fuzzy match, no match, empty name, defense companies

**tests/unit/test_network/test_lobbying.py** — 20 tests:
- `TestLobbyingFilingTransform` (5) — basic filing, empty/missing UUID, expenses fallback, date parsing
- `TestBillReferenceExtraction` (4) — H.R./S. extraction, multiple bills, no bills
- `TestFormerPositionDetection` (9) — senator, representative, staff director, executive branch, edge cases
- `TestLobbyistNameNormalization` (3) — first/last, empty dict, missing key

**tests/unit/test_network/test_campaign_finance.py** — 8 tests:
- `TestFECCommitteeTransform` (3) — basic committee, no candidate, empty ID
- `TestFECContributionTransform` (5) — basic contribution, no name/amount, missing/bad date

**tests/unit/test_media/test_govinfo_hearings.py** — 12 tests:
- `TestGovInfoHearingTransform` (6) — basic hearing, empty/missing package ID, no title, no full text, URL generation
- `TestStripHtml` (3) — tag removal, empty string, no HTML
- `TestParseDate` (3) — ISO date, ISO datetime, invalid date

**tests/unit/test_media/test_youtube.py** — 6 tests:
- `TestYouTubeTranscriptTransform` (6) — basic video, missing ID, empty transcript, transcript flag, bad/missing date

**tests/unit/test_media/test_gnews.py** — 5 tests:
- `TestGNewsTransform` (5) — basic article, no title, no URL, bad date, content fallback

**tests/unit/test_media/test_newsdata.py** — 5 tests:
- `TestNewsDataTransform` (5) — basic article, no title, no source ID, alternate date format, null creator

**tests/unit/test_media/test_congress_rss.py** — 10 tests:
- `TestCongressRSSTransform` (5) — basic entry, no title, no ID, link fallback, feed name
- `TestRSSStripHtml` (2) — strips tags/entities, empty
- `TestRSSParseDate` (3) — RFC822, ISO, invalid

**tests/unit/test_media/test_press_releases.py** — 6 tests:
- `TestPressReleaseTransform` (6) — basic press release, no title, no ID, HTML stripping, content field, empty bioguide

**tests/unit/test_media/test_twitter.py** — 5 tests:
- `TestTwitterTransform` (5) — basic tweet, no ID, no text, title truncation, skip without token

**tests/unit/test_media/test_text_processing.py** — 20 tests:
- `TestExtractTickerMentions` (7) — dollar prefix, bare tickers, false positives, known ticker filter, empty, no duplicates, no filter
- `TestStripHtml` (5) — tag removal, entity removal, empty, None, whitespace collapse
- `TestTruncateForModel` (3) — short, long, empty
- `TestAnalyzeSentimentMocked` (3) — positive, negative, empty text
- `TestExtractEntitiesMocked` (2) — entity extraction, empty text

---

## Key Design Decisions

### Multi-Source Fallback Cascade
Trade collection uses a fallback strategy: primary S3 sources (House/Senate Stock Watcher) are tried first, and on failure (currently 403), tasks automatically fall back to free alternatives — `HouseClerkCollector` (scrapes PTR PDFs from clerk.house.gov with concurrent downloads via `asyncio.Semaphore(10)`) and `GitHubSenateCollector` (parses CSV from jeremiak's GitHub dataset). FMP paid API runs as a supplementary source on a separate schedule. All sources deduplicate via the same unique index, so duplicate trades from multiple sources are handled automatically.

### Dual Dating for the 30-45 Day Reporting Lag
Every trade stores both `transaction_date` (when it happened) and `disclosure_date` (when it was filed). This is critical because:
- Legislative context correlates with `transaction_date`
- Actionable follow-trading signals use `disclosure_date` (can't act on what isn't public yet)
- The lag itself is a signal — members who file quickly are more useful for follow strategies

### Multi-Source Trade Dedup
Trades from different sources (House Watcher, FMP, etc.) are deduplicated via unique index on `(member_name, ticker, transaction_date, transaction_type, amount_range_low)`. INSERT ON CONFLICT DO NOTHING means the first source to ingest a trade wins.

### Async Everywhere
- SQLAlchemy 2.0 async with asyncpg driver
- httpx async client for all external API calls
- Celery tasks bridge sync→async with `asyncio.run()`

### Build System
Uses hatchling with `packages = ["src"]` in pyproject.toml. This is required because the source code is in `src/` not at the project root.

### Unified Media Table (Phase 4)
All media content (hearing transcripts, YouTube videos, news articles, press releases, tweets) stored in a single `media_content` table with a `source_type` discriminator column. This simplifies queries across content types and avoids table proliferation. NLP results in a separate `sentiment_analysis` table linked by FK.

### Lazy-Loaded NLP Models (Phase 4)
FinBERT and spaCy models are loaded on first use, not at import time. This prevents slow startup for API/worker processes that don't need NLP. The `[ml]` optional dependency group keeps heavy ML libs separate from core dependencies.

### Extensible Media Collectors (Phase 4)
All media collectors use the skip-if-no-key pattern: if an API key is empty, `collect()` returns `[]` immediately. Adding a new data source requires: (1) adding the API key to config.py, (2) creating a new collector extending BaseCollector, (3) adding a Celery task. No existing code changes needed.

### Walk-Forward Temporal CV (Phase 5)
ML models use `TemporalSplitter` for walk-forward cross-validation that splits strictly on `transaction_date`. This prevents future data leakage — each fold trains on historical data and tests on the subsequent period. The anomaly model additionally excludes market features (price changes, volatility, volume) since those would leak the outcome.

### Stacking Ensemble (Phase 5)
The ensemble meta-learner (Logistic Regression) takes probability outputs from the trade predictor, return predictor, and anomaly model, plus raw `timing_suspicion_score` and `avg_sentiment_7d` as direct features. This lets the ensemble learn which base model signals are most predictive in combination.

### Composite Signal Scoring (Phase 6)
Signal strength is computed via a multi-factor scoring function: base model confidence, freshness bonus (+0.1 for recent disclosures), corroboration bonus (+0.15 per evidence source), cluster size bonus (for insider_cluster signals), and a lag penalty that discounts old disclosures. All scores are capped at 1.0.

### Production Middleware Stack (Phase 8)
Middleware is applied in order: CorrelationId → RateLimit → ApiKey → CORS. Rate limiting uses an in-memory sliding window (no Redis dependency at startup). API key auth is only enforced when `APP_ENV=production` and `API_KEYS` is configured. Public paths (/health, /metrics, /docs) are always accessible.

---

## Implementation Progress

### Phase 1: Foundation + Trade Data — COMPLETE
- [x] Project scaffold (pyproject.toml, Docker, config, .gitignore)
- [x] Database models (CongressMember, TradeDisclosure, StockDaily + family/staff/committee)
- [x] Alembic async migration setup
- [x] BaseCollector abstraction with rate limiting and retry
- [x] Trade collectors: HouseWatcher, SenateWatcher, FMP (House + Senate)
- [x] Market data collector (yfinance)
- [x] Database upsert logic with dedup
- [x] Celery tasks + beat schedule (6h trades, daily market data)
- [x] FastAPI endpoints: /members, /members/{id}/trades, /trades, /trades/stats
- [x] Pydantic schemas for API responses
- [x] Historical backfill script (scripts/seed_initial_data.py)
- [x] Unit tests: 11 passing

### Phase 2: Legislative Context — COMPLETE
- [x] Database models: Bill, BillCosponsor, Committee, CommitteeHearing, VoteRecord
- [x] Added nominate_dim1/dim2 to CongressMember model
- [x] Congress.gov API client — members, bills (HR/S/HJRES/SJRES), committees, hearings
- [x] Voteview integration — DW-NOMINATE ideology scores (downloads HSall_members.csv)
- [x] Timing analysis engine (`src/processing/timing_analysis.py`):
  - Finds committee hearings ±30 days, bills ±90 days of each trade
  - Committee-sector alignment detection via `COMMITTEE_SECTOR_MAP`
  - Heuristic suspicion score [0,1] based on timing proximity, sector alignment, disclosure lag, sponsor status
  - `get_trades_for_committee()` — all trades by members of a committee
- [x] API endpoints: /bills, /bills/{id}, /committees, /committees/{code}, /committees/{code}/trades, /committees/{code}/hearings, /trades/{id}/legislative-context
- [x] Celery tasks: daily members + bills, weekly committees + hearings + voteview
- [x] Loader functions: upsert_members, update_member_ideology, upsert_bills, upsert_committees, upsert_hearings
- [x] Unit tests: 27 new in Phase 2 (14 congress_gov, 5 voteview, 8 timing analysis)

### Phase 3: Network Graph — COMPLETE
- [x] Neo4j 5 Docker service with APOC plugin, healthcheck, persistent volume
- [x] Neo4j async driver wrapper (`src/db/neo4j.py`) with singleton pattern, connectivity check
- [x] Database models: LobbyingFiling, LobbyingRegistrant, LobbyingClient, LobbyingLobbyist, CampaignCommittee, CampaignContribution
- [x] Graph schema (`src/graph/schema.py`): 6 node types, 11 relationship types, uniqueness constraints + indexes
- [x] PostgreSQL → Neo4j sync (`src/graph/sync.py`): full MERGE-based sync for members, committees, assignments, trades, bills, cosponsors, lobbying, campaign finance
- [x] Lobbying data collector (`src/ingestion/network/lobbying.py`): Senate LDA API client with pagination, filing/registrant/client/lobbyist extraction, bill reference parsing, revolving door detection
- [x] Campaign finance collector (`src/ingestion/network/campaign_finance.py`): FEC API for committees + individual contributions with employer info
- [x] Entity resolution (`src/processing/normalizer.py`): multi-strategy matching (manual overrides for 60+ companies, exact match, fuzzy via thefuzz/rapidfuzz), corporate suffix normalization, async batch resolution for lobbying clients and campaign employers
- [x] Graph queries (`src/graph/queries.py`): member network, all paths between member↔ticker, suspicious triangles (member traded company that lobbied them), committee-company overlap, member trading connections summary, graph stats
- [x] API endpoints: /network/stats, /network/member/{id}, /network/member/{id}/paths-to/{ticker}, /network/member/{id}/connections, /network/suspicious-triangles, /network/committee-company-overlap
- [x] Celery tasks: weekly lobbying + campaign committees, daily entity resolution + graph sync
- [x] Loader functions: upsert_lobbying_filings (nested registrant/client/lobbyists), upsert_campaign_committees, upsert_campaign_contributions
- [x] Unit tests: 44 new (12 normalizer, 20 lobbying, 8 campaign finance + 4 from pattern) — 82 total passing

### Phase 4: Media Intelligence + NLP — COMPLETE
- [x] Database models: MediaContent (unified table with source_type discriminator), SentimentAnalysis, MemberMediaMention
- [x] GovInfo hearing transcript collector — full text from CHRG collection, speaker extraction
- [x] YouTube transcript collector — youtube-transcript-api + RSS feed video discovery, curated channel list (C-SPAN, committees)
- [x] GNews API collector — congressional trading news search, 100 free req/day, skip-if-no-key
- [x] NewsData.io collector — politics/business news, 2000 free articles/day, skip-if-no-key
- [x] Congress.gov RSS collector — 5 feeds (house/senate floor, bills, committee activity), no key needed
- [x] Member press release collector — curated RSS feeds for active traders (Pelosi, Tuberville, etc.)
- [x] X/Twitter collector — full stub implementation, skip-if-no-key, ready for ~$200/mo API when needed
- [x] NLP pipeline (`src/processing/text_processing.py`): FinBERT sentiment (lazy-loaded), spaCy NER (lazy-loaded), ticker mention extraction with false positive filtering, HTML stripping, text truncation
- [x] API endpoints: /media, /media/stats, /media/{id}, /members/{id}/media, /members/{id}/sentiment-timeline
- [x] Celery tasks: weekly hearing transcripts, daily YouTube + press releases, every 6h news, every 3h RSS, daily NLP analysis
- [x] Loader functions: upsert_media_content, upsert_sentiment_analyses, upsert_member_media_mentions
- [x] Unit tests: 69 new — 151 total passing

### Phase 5: ML Prediction Engine — COMPLETE
- [x] Database models: MLModelArtifact (trained model tracking), TradePrediction (per-trade predictions with actual return backfill)
- [x] Feature engineering (`src/ml/features.py`): 6 feature groups — trade, member, market (RSI, volatility, volume ratio), legislative (reuses timing_analysis), sentiment (7d/30d averages + momentum), network (lobbying/campaign connections)
- [x] Dataset builder + TemporalSplitter: walk-forward cross-validation with configurable train/test windows, splits on transaction_date — no future data leakage
- [x] Trade Predictor (LightGBM classifier): predicts probability of profitable 5d trade, class_weight="balanced"
- [x] Return Predictor (XGBoost regressor): predicts expected 5d return magnitude
- [x] Anomaly Detector (Isolation Forest): unusual trading patterns, market features excluded to prevent leakage
- [x] Ensemble (Logistic Regression meta-learner): stacking outputs from 3 base models + timing_suspicion_score + avg_sentiment_7d
- [x] ModelTrainer orchestrator: trains all models, saves artifacts to `data/models/`, records in DB, manages active/inactive
- [x] PredictionService (lazy-loaded singleton): predict single trade or batch, loads active artifacts from DB
- [x] Evaluation metrics: classifier (accuracy, precision, recall, F1, AUC), regressor (MAE, RMSE, R²), profit (profit_factor, win_rate)
- [x] Celery tasks: weekly training (Sunday 2 PM), daily batch prediction (12 PM), daily return backfill (5 PM)
- [x] API endpoints: /predictions (list, filter, stats, leaderboard, per-trade)
- [x] Unit tests: 40 new — 191 total passing

### Phase 6: Signal Generation — COMPLETE
- [x] Database models: Signal (5 types, strength/confidence/evidence, expiry), AlertConfig (signal_types, filters, webhook)
- [x] SignalGenerator: 4 signal types implemented:
  - trade_follow: high-confidence ML predictions (>0.7), disclosure lag penalty, 7d expiry
  - anomaly_alert: anomalous trade patterns, 21d expiry
  - sentiment_divergence: trade direction contradicts 30d sentiment (contrarian/insider signals), 14d expiry
  - insider_cluster: 3+ members trading same stock same direction within 7 days, 10d expiry
- [x] Composite signal scoring: freshness bonus (+0.1 if lag <7d), lag penalty (max(0.3, 1-lag/45)), corroboration bonus (+0.15/source), cluster size bonus, cap at 1.0
- [x] Alert dispatch: matches signals to AlertConfig, rate limiting (10/hour/config), webhook POST
- [x] Celery tasks: daily signal generation (1 PM), daily expiration (3 AM), 30-min alert dispatch
- [x] API endpoints: /signals (list, stats, detail), /alerts/configs (CRUD)
- [x] Unit tests: 19 new — 210 total passing

### Phase 7: Streamlit Dashboard — COMPLETE
- [x] CongressAPI sync httpx client with in-memory TTL cache (60s stats, 300s lists)
- [x] 5 reusable Plotly chart functions: trade timeline, sentiment timeline, signal strength, model performance, top tickers
- [x] 6 dashboard pages:
  - Overview: KPI cards, top tickers chart, signal strength, recent trades, data sources
  - Member Explorer: member selector, trade timeline + table, sentiment chart, ML predictions
  - Trade Feed: filterable table, legislative context with suspicion score
  - Network Graph: interactive pyvis visualization, depth control, suspicious triangles
  - Signals: signal stats, filterable list with evidence expanders, alert config form
  - Backtesting: model performance chart, prediction stats, member leaderboard
- [x] Docker service: port 8501, connects to API at http://api:8000/api/v1
- [x] Unit tests: 16 new — 226 total passing

### Phase 8: Production Hardening — COMPLETE
- [x] Structured JSON logging (`src/logging_config.py`): python-json-logger, correlation ID via ContextVar, CorrelationFilter
- [x] Prometheus metrics (`src/metrics.py`): counters (trades_ingested, predictions_generated, signals_generated), histograms (prediction_latency, feature_computation), gauges (active_signals, model_accuracy)
- [x] prometheus-fastapi-instrumentator: automatic HTTP request metrics at /metrics endpoint
- [x] Correlation ID middleware: generates/propagates X-Correlation-ID header across requests
- [x] Rate limiting middleware: in-memory sliding window (100 req/min anonymous, 1000 authenticated via X-API-Key)
- [x] API key middleware: optional enforcement in production mode (APP_ENV=production)
- [x] Response cache (`src/api/cache.py`): in-memory TTL cache with @cached decorator, prefix invalidation
- [x] Enhanced health checks: /health/detailed (PostgreSQL, Redis, Neo4j, data freshness), /health/legal (disclaimer)
- [x] Config: rate_limit_per_minute, rate_limit_authenticated, api_keys settings
- [x] Unit tests: 24 new — 250 total passing (277 after trade source additions)

---

## Data Sources & Costs

| Source | Data | Cost | Phase | Status |
|---|---|---|---|---|
| House/Senate Stock Watcher | Trade filings | Free | 1 | **S3 returning 403** — fallbacks active |
| House Clerk PTR Scraper | House trade PDFs | Free | 1 | **Active** — scrapes disclosures-clerk.house.gov |
| GitHub Senate CSV | Senate trades (2012-2024) | Free | 1 | **Active** — jeremiak/us-senate-financial-disclosure-data |
| yfinance | Stock prices | Free | 1 | Active |
| Financial Modeling Prep | Structured trade API | ~$15-30/mo | 1 | Supplement (needs API key) |
| Congress.gov API | Bills, members, committees, votes, hearings | Free | 2 (done) |
| Voteview CSV | DW-NOMINATE ideology scores | Free | 2 (done) |
| Senate LDA API | Lobbying filings (registrants, clients, lobbyists, issues) | Free | 3 (done) |
| FEC API | Campaign committees + individual contributions | Free (API key) | 3 (done) |
| GovInfo API | Committee hearing transcripts | Free (API key) | 4 (done) |
| YouTube transcripts | Congressional channel interviews | Free | 4 (done) |
| GNews API | Congressional trading news | Free (100 req/day) | 4 (done) |
| NewsData.io | Politics/business news | Free (2000/day) | 4 (done) |
| Congress.gov RSS | Floor updates, bill activity | Free | 4 (done) |
| Member press releases | RSS from member websites | Free | 4 (done) |
| X/Twitter API | Social media posts | ~$200/mo | 4 (stub, skip-if-no-key) |

**Current cost: $0/mo** with free fallback sources. FMP optional at ~$15-30/mo for structured API supplement. All Phase 4 media sources are free. X/Twitter: ~$200/mo when activated.

---

## Key Differentiators vs. Existing Platforms

| Capability | Capitol Trades / Quiver / Unusual Whales | Our System |
|---|---|---|
| Trade filing aggregation | Yes | Yes (multi-source) |
| Network/relationship graph | No | **Neo4j graph: members ↔ families ↔ lobbyists ↔ donors ↔ staff** |
| Predictive ML models | No | **Trade prediction, return forecasting, anomaly detection** |
| Social media NLP | No | **Sentiment analysis of X posts, press releases, hearing transcripts** |
| Trade-legislation timing analysis | No | **Correlate trades with committee hearings and bill activity** |
| Multi-signal fusion | No | **Ensemble combining all data streams into unified signals** |
| Extended network tracking | Partial (spouse/dependent only) | **Staff, revolving door lobbyists, campaign donors** |
