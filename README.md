# Congress Predictions

**ML-powered trading signals from congressional disclosures, legislative activity, network analysis, and media sentiment.**

Track trades by Congress and Senate members (and their families, staff, and associates), fuse multiple data streams, and generate predictive signals using machine learning.

---

## What This Does

Members of Congress are required to disclose stock trades within 30-45 days. This system:

1. **Ingests** trade disclosures from 4 sources, market data, legislative activity, lobbying filings, campaign finance, and media content from 7 sources
2. **Builds** a relationship graph (Neo4j) connecting members to companies via lobbying, campaign donations, committee assignments, and revolving-door lobbyists
3. **Analyzes** trade-legislation timing correlations, media sentiment (FinBERT), and network connections
4. **Predicts** trade profitability using an ensemble of LightGBM, XGBoost, and Isolation Forest models
5. **Generates** composite trading signals with configurable alerts
6. **Visualizes** everything in an interactive Streamlit dashboard

---

## Architecture

```
Data Sources (17)        Celery Workers (26 tasks)       Storage
 ├─ Trade Filings  ───>  ├─ Ingestion Tasks  ──────────> PostgreSQL 16
 ├─ Market Data    ───>  ├─ Legislative Tasks             (SQLAlchemy async)
 ├─ Congress.gov   ───>  ├─ Network Tasks    ──────────> Neo4j 5
 ├─ Lobbying/FEC   ───>  ├─ Media + NLP Tasks             (relationship graph)
 ├─ News/Media     ───>  ├─ ML Tasks         ──────────> Redis 7
 └─ RSS Feeds      ───>  └─ Signal Tasks                   (broker + cache)
                                │
                                v
                    ┌─────────────────────┐
                    │  ML Prediction Engine │
                    │  ├─ LightGBM         │
                    │  ├─ XGBoost          │
                    │  ├─ Isolation Forest  │
                    │  └─ Ensemble Stacking │
                    └─────────┬───────────┘
                              │
                              v
                    ┌─────────────────────┐
                    │  Signal Generation   │
                    │  (4 signal types)    │
                    └─────────┬───────────┘
                              │
                    ┌─────────┴───────────┐
                    │                     │
              FastAPI REST API     Streamlit Dashboard
              (port 8000)          (port 8501)
```

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy 2.0 (async), PostgreSQL 16, Neo4j 5, Redis 7, Celery 5, LightGBM, XGBoost, HuggingFace Transformers, Streamlit, Plotly, Docker

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose

### Option 1: Docker (all services)

```bash
# Clone and configure
git clone https://github.com/Umesh1608/congress-predictions.git
cd congress-predictions
cp .env.example .env
# Edit .env — add API keys (most are free, see Data Sources below)

# Start everything
docker compose up
```

- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Prometheus metrics: http://localhost:8000/metrics

### Option 2: Local development

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"        # Core + dev tools
pip install -e ".[ml]"         # ML models (LightGBM, XGBoost, transformers)
pip install -e ".[dashboard]"  # Streamlit dashboard

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Start infrastructure
docker compose up -d postgres redis neo4j

# 5. Seed historical data
python -m scripts.seed_initial_data

# 6. Start the API
uvicorn src.main:app --reload

# 7. Start background workers
celery -A src.tasks.celery_app worker --loglevel=info --concurrency=4
celery -A src.tasks.celery_app beat --loglevel=info

# 8. Start the dashboard
streamlit run dashboard/app.py
```

---

## Data Sources

| Source | Data | Cost | API Key |
|---|---|---|---|
| House/Senate Stock Watcher | Trade filings (S3 JSON) | Free | None |
| Financial Modeling Prep | Structured trade API | ~$15-30/mo | [fmp](https://financialmodelingprep.com/) |
| Yahoo Finance | Stock prices (yfinance) | Free | None |
| Congress.gov | Members, bills, committees, hearings | Free | [api.data.gov](https://api.data.gov/signup/) |
| Voteview | DW-NOMINATE ideology scores | Free | None |
| Senate LDA | Lobbying filings | Free | None |
| FEC | Campaign committees + contributions | Free | [api.data.gov](https://api.data.gov/signup/) |
| GovInfo | Hearing transcripts | Free | [api.data.gov](https://api.data.gov/signup/) |
| YouTube | Congressional channel transcripts | Free | None |
| GNews | Trading news articles | Free (100/day) | [gnews.io](https://gnews.io/) |
| NewsData.io | Politics/business news | Free (2000/day) | [newsdata.io](https://newsdata.io/) |
| Congress.gov RSS | Floor updates, bill activity | Free | None |
| Member press releases | RSS from member websites | Free | None |
| X/Twitter | Social media (stub) | ~$200/mo | Optional |

**Minimum cost: $0/mo** (free sources only). With FMP for structured trade data: **~$15-30/mo**.

All collectors use a skip-if-no-key pattern — missing API keys are silently skipped, not errors.

> **Optional paid upgrades:** The system is pre-built to support two paid data sources that can be activated at any time by adding their API keys to `.env`:
> - **X/Twitter API (~$200/mo)** — Real-time social media sentiment from congressional accounts. The collector, transform logic, and Celery task are fully implemented; just set `TWITTER_BEARER_TOKEN` and add the task to the beat schedule.
> - **Financial Modeling Prep (~$15-30/mo)** — Structured, cleaned trade disclosure data as a complement to the free House/Senate Stock Watcher sources. Set `FMP_API_KEY` to activate. Already included in the default beat schedule.

---

## ML Pipeline

### Feature Engineering (6 groups, 30+ features)

| Group | Features |
|---|---|
| **Trade** | Amount midpoint, is_purchase, filer_type, disclosure lag days |
| **Member** | Party, chamber, DW-NOMINATE ideology scores, years in office, committee count |
| **Market** | 5d/21d price change, 21d volatility, volume ratio, RSI-14 |
| **Legislative** | Hearing proximity, bill proximity, committee-sector alignment, suspicion score |
| **Sentiment** | 7d/30d average sentiment, sentiment momentum, media mention counts |
| **Network** | Lobbying connections, campaign donor connections, network degree, lobbying triangles |

### Models

| Model | Type | Purpose |
|---|---|---|
| **Trade Predictor** | LightGBM Classifier | Probability trade is profitable at 5 days |
| **Return Predictor** | XGBoost Regressor | Expected 5-day return magnitude |
| **Anomaly Detector** | Isolation Forest | Unusual trading patterns (market features excluded) |
| **Ensemble** | Logistic Regression (stacking) | Unified signal score combining all models |

Walk-forward temporal cross-validation prevents future data leakage. Models retrain weekly.

### Signal Types

| Signal | Trigger | Expiry |
|---|---|---|
| **Trade Follow** | High-confidence ML prediction (>0.7) | 7 days |
| **Anomaly Alert** | Anomalous trade pattern detected | 21 days |
| **Sentiment Divergence** | Trade contradicts media sentiment | 14 days |
| **Insider Cluster** | 3+ members trading same stock, same direction, within 7 days | 10 days |

Signals are scored with freshness bonus, disclosure lag penalty, corroboration bonus, and cluster size bonus, capped at 1.0.

---

## API Endpoints

Base URL: `/api/v1` | Full docs at `/docs` (Swagger UI)

| Category | Endpoints |
|---|---|
| **Members** | `GET /members`, `GET /members/{id}`, `GET /members/{id}/trades` |
| **Trades** | `GET /trades`, `GET /trades/stats`, `GET /trades/{id}/legislative-context` |
| **Legislation** | `GET /bills`, `GET /bills/{id}`, `GET /committees`, `GET /committees/{code}/hearings` |
| **Network** | `GET /network/member/{id}`, `GET /network/suspicious-triangles`, `GET /network/stats` |
| **Media** | `GET /media`, `GET /media/stats`, `GET /members/{id}/sentiment-timeline` |
| **Predictions** | `GET /predictions`, `GET /predictions/model-performance`, `GET /predictions/leaderboard` |
| **Signals** | `GET /signals`, `GET /signals/stats`, `POST /alerts/configs` |
| **Health** | `GET /health`, `GET /health/detailed`, `GET /health/legal`, `GET /metrics` |

---

## Dashboard Pages

| Page | Description |
|---|---|
| **Overview** | KPI cards, top traded tickers, signal strength chart, recent trades |
| **Member Explorer** | Deep dive into any member: trades, committees, sentiment, predictions |
| **Trade Feed** | Filterable trade list with ML predictions and legislative context |
| **Network Graph** | Interactive pyvis visualization of member-company relationships |
| **Signals** | Active signals sorted by strength, alert configuration management |
| **Backtesting** | Model performance metrics, walk-forward results, member leaderboard |

---

## Background Tasks (26 scheduled)

All tasks run via Celery Beat on US/Eastern timezone:

| Frequency | Tasks |
|---|---|
| **Every 3h** | Congress RSS feeds |
| **Every 6h** | Trade collection (4 sources), news articles (GNews + NewsData) |
| **Every 30min** | Alert dispatch |
| **Daily** | Members, bills, market data, YouTube, press releases, NLP analysis, batch predictions, signal generation, return backfill, signal expiration |
| **Weekly (Sunday)** | Committees, hearings, Voteview scores, lobbying filings, campaign finance, entity resolution, graph sync, ML model retraining |

---

## Production Features

- **Structured JSON logging** with correlation IDs propagated across requests
- **Prometheus metrics** — HTTP request metrics (auto-instrumented), custom counters/histograms/gauges at `/metrics`
- **Rate limiting** — sliding window: 100 req/min anonymous, 1000 req/min with API key
- **API key authentication** — optional, enforced only when `APP_ENV=production`
- **Response caching** — in-memory TTL cache with `@cached` decorator and prefix invalidation
- **Health checks** — `/health/detailed` verifies PostgreSQL, Redis, Neo4j, and data freshness

---

## Development

```bash
# Run all tests (250 passing)
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/unit/test_ml/ -v          # ML tests
python -m pytest tests/unit/test_signals/ -v      # Signal tests
python -m pytest tests/unit/test_production/ -v   # Production tests

# Lint & format
ruff check src/ tests/
ruff format src/ tests/

# Type checking
mypy src/

# Database migrations
alembic revision --autogenerate -m "description"
alembic upgrade head
```

### Docker Services

| Service | Image | Port |
|---|---|---|
| `postgres` | postgres:16-alpine | 5432 |
| `redis` | redis:7-alpine | 6379 |
| `neo4j` | neo4j:5-community | 7474, 7687 |
| `api` | Built from Dockerfile | 8000 |
| `celery-worker` | Built from Dockerfile | — |
| `celery-beat` | Built from Dockerfile | — |
| `dashboard` | Built from Dockerfile | 8501 |

---

## Key Design Decisions

- **Dual dating** — Every trade stores `transaction_date` (when it happened) and `disclosure_date` (when filed). The 30-45 day lag is itself a signal.
- **Multi-source dedup** — Trades from different sources are deduplicated via unique index on `(member_name, ticker, transaction_date, transaction_type, amount_range_low)`.
- **Walk-forward CV** — ML models use temporal cross-validation split on `transaction_date` to prevent future data leakage. The anomaly model additionally excludes market features.
- **Stacking ensemble** — Meta-learner combines probability outputs from 3 base models + direct features (timing suspicion score, sentiment) to learn optimal signal weighting.
- **Composite signal scoring** — Multi-factor scoring with freshness bonus, lag penalty, corroboration bonus, and cluster size bonus. All scores capped at 1.0.
- **Skip-if-no-key pattern** — All data collectors gracefully skip when their API key is missing. No configuration errors, just reduced data coverage.
- **Async everywhere** — SQLAlchemy 2.0 async with asyncpg. Celery tasks bridge sync-to-async with `asyncio.run()`.

---

## Differentiators vs. Existing Platforms

| Capability | Capitol Trades / Quiver / Unusual Whales | This System |
|---|---|---|
| Trade filing aggregation | Yes | Yes (4 sources, multi-source dedup) |
| Network/relationship graph | No | Neo4j graph: members, lobbyists, donors, staff, companies |
| Predictive ML models | No | LightGBM + XGBoost + Isolation Forest + ensemble |
| NLP sentiment analysis | No | FinBERT on news, press releases, hearing transcripts |
| Trade-legislation timing | No | Correlates trades with hearings, bills, sector alignment |
| Multi-signal fusion | No | Ensemble combining all data streams into scored signals |
| Extended network tracking | Partial (spouse only) | Staff, revolving door lobbyists, campaign donors |
| Alerting system | No | Configurable alerts with webhook dispatch |

---

## Disclaimer

This system is for **informational and research purposes only**. It does NOT constitute financial advice, investment recommendations, or solicitation to buy or sell securities. Past congressional trading patterns do not guarantee future results. Always consult a qualified financial advisor before making investment decisions.

---

## License

This project is provided as-is for educational and research purposes.
