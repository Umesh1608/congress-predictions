"""Production middleware: rate limiting, API key auth, correlation IDs."""

from __future__ import annotations

import time
import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.config import settings
from src.logging_config import correlation_id_var, generate_correlation_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Correlation ID middleware
# ---------------------------------------------------------------------------

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach a correlation ID to every request."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        cid = request.headers.get("X-Correlation-ID") or generate_correlation_id()
        correlation_id_var.set(cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response


# ---------------------------------------------------------------------------
# In-memory sliding-window rate limiter (no Redis dependency for startup)
# ---------------------------------------------------------------------------

class _SlidingWindow:
    """Simple in-memory sliding-window counter."""

    def __init__(self) -> None:
        self._windows: dict[str, list[float]] = {}

    def is_allowed(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        now = time.monotonic()
        timestamps = self._windows.get(key, [])
        # Prune old entries
        cutoff = now - window_seconds
        timestamps = [t for t in timestamps if t > cutoff]
        if len(timestamps) >= limit:
            self._windows[key] = timestamps
            return False
        timestamps.append(now)
        self._windows[key] = timestamps
        return True


_rate_limiter = _SlidingWindow()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limit requests per client IP (or API key)."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip rate limiting for health checks and metrics
        if request.url.path in ("/health", "/health/detailed", "/health/legal", "/metrics"):
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if api_key and _is_valid_api_key(api_key):
            limit = settings.rate_limit_authenticated
            key = f"apikey:{api_key}"
        else:
            limit = settings.rate_limit_per_minute
            client_ip = request.client.host if request.client else "unknown"
            key = f"ip:{client_ip}"

        if not _rate_limiter.is_allowed(key, limit):
            logger.warning("Rate limit exceeded for %s", key)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )

        return await call_next(request)


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------

def _is_valid_api_key(key: str) -> bool:
    """Check if an API key is in the configured set."""
    if not settings.api_keys:
        return False
    valid_keys = {k.strip() for k in settings.api_keys.split(",") if k.strip()}
    return key in valid_keys


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Optional API key requirement for production.

    When api_keys is configured and the request path is under /api/,
    an API key header is required. Public paths are always accessible.
    """

    PUBLIC_PATHS = {"/health", "/health/detailed", "/health/legal", "/metrics", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Only enforce if API keys are configured and env is production
        if settings.app_env != "production" or not settings.api_keys:
            return await call_next(request)

        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if not api_key or not _is_valid_api_key(api_key):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key."},
            )

        return await call_next(request)
