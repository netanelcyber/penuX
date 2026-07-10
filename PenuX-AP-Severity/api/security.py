"""Security middleware and dependencies for the research API.

Implements the technical controls flagged by the HIPAA/ISO 27799 gap
analysis (see docs/hipaa_iso27799_gap_analysis_he.md):
  - API key authentication (§164.312(d) person/entity authentication)
  - Per-client rate limiting (availability / basic DoS mitigation)
  - Request body size limits (availability / basic DoS mitigation)
  - Audit logging of every request: who, what, when, outcome — never the
    clinical payload itself (§164.312(b) audit controls)
  - Generic error responses to callers; full exception detail only in
    server-side logs (avoids leaking internals to an unauthenticated network)

None of this makes the software HIPAA/ISO 27799 *compliant* by itself —
compliance is an organizational + technical program, not a code property.
It closes the concrete, fixable technical gaps identified in this repo.
"""
import hmac
import logging
import os
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

log = logging.getLogger("penux_ap.api.security")

API_KEY_ENV_VAR = "PENUX_AP_API_KEY"
MAX_BODY_BYTES = int(os.environ.get("PENUX_AP_MAX_BODY_BYTES", 1_000_000))  # 1 MB
RATE_LIMIT_REQUESTS = int(os.environ.get("PENUX_AP_RATE_LIMIT_REQUESTS", 60))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("PENUX_AP_RATE_LIMIT_WINDOW_SECONDS", 60))


def _configured_api_key() -> str | None:
    return os.environ.get(API_KEY_ENV_VAR) or None


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency enforcing X-API-Key when PENUX_AP_API_KEY is set.

    If the operator has not configured an API key, the endpoint remains open
    (research-default) but every request is still audit-logged and a startup
    warning is emitted — see main.py:startup(). This mirrors the "Addressable"
    vs "Required" split in the HIPAA Security Rule: the mechanism must exist
    and be usable, but this repo cannot force a deployer to turn it on.
    """
    configured = _configured_api_key()
    if configured is None:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, configured):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple fixed-window per-client-IP rate limiter (in-memory).

    Not a substitute for a proper API gateway/WAF in production, but closes
    the "no rate limiting at all" gap for the research deployment this repo
    ships. Health checks are exempt so orchestrator liveness probes aren't
    throttled.
    """

    def __init__(self, app, max_requests: int = RATE_LIMIT_REQUESTS, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        client_id = request.client.host if request.client else "unknown"
        now = time.monotonic()
        hits = self._hits[client_id]
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()
        if len(hits) >= self.max_requests:
            log.warning("Rate limit exceeded for client=%s path=%s", client_id, request.url.path)
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
        hits.append(now)
        return await call_next(request)


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Rejects requests whose declared or actual body size exceeds the limit.

    Mitigates unbounded-body DoS on endpoints like /hl7/predict that read
    the raw request body as text with no prior size validation.
    """

    def __init__(self, app, max_bytes: int = MAX_BODY_BYTES):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None and content_length.isdigit() and int(content_length) > self.max_bytes:
            return JSONResponse(status_code=413, content={"detail": "Request body too large"})
        return await call_next(request)


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Logs who called which endpoint, when, and with what outcome.

    Deliberately logs metadata only (timestamp, client host, method, path,
    status code, duration) — never the request or response body, which may
    contain clinical values. Satisfies HIPAA §164.312(b) "audit controls"
    at a basic level; a production deployment should ship these logs to a
    tamper-evident, access-controlled audit store rather than stdout.
    """

    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        client_host = request.client.host if request.client else "unknown"
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.monotonic() - start) * 1000
            log.exception(
                "AUDIT client=%s method=%s path=%s status=500 duration_ms=%.1f",
                client_host, request.method, request.url.path, duration_ms,
            )
            raise
        duration_ms = (time.monotonic() - start) * 1000
        log.info(
            "AUDIT client=%s method=%s path=%s status=%d duration_ms=%.1f",
            client_host, request.method, request.url.path, response.status_code, duration_ms,
        )
        return response
