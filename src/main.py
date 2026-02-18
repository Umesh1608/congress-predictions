from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.config import settings
from src.db.postgres import engine
from src.db.neo4j import close_driver as close_neo4j
from src.logging_config import setup_logging
from src.api.middleware import CorrelationIdMiddleware, RateLimitMiddleware, ApiKeyMiddleware
from src.api.v1.members import router as members_router
from src.api.v1.trades import router as trades_router
from src.api.v1.legislation import router as legislation_router
from src.api.v1.network import router as network_router
from src.api.v1.media import router as media_router
from src.api.v1.predictions import router as predictions_router
from src.api.v1.signals import router as signals_router
from src.api.v1.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logging(settings.log_level)
    yield
    await engine.dispose()
    await close_neo4j()


app = FastAPI(
    title="Congress Predictions API",
    description="Congressional trade tracking and prediction system",
    version="0.1.0",
    lifespan=lifespan,
)

# Middleware (order matters — outermost first)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(ApiKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

app.include_router(members_router, prefix="/api/v1")
app.include_router(trades_router, prefix="/api/v1")
app.include_router(legislation_router, prefix="/api/v1")
app.include_router(network_router, prefix="/api/v1")
app.include_router(media_router, prefix="/api/v1")
app.include_router(predictions_router, prefix="/api/v1")
app.include_router(signals_router, prefix="/api/v1")
app.include_router(health_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
