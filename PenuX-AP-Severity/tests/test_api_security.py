"""Tests for API-key auth, rate limiting, and body-size limits in api/security.py."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


def test_health_open_without_api_key(client, monkeypatch):
    monkeypatch.delenv("PENUX_AP_API_KEY", raising=False)
    resp = client.get("/health")
    assert resp.status_code == 200


def test_predict_requires_api_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("PENUX_AP_API_KEY", "secret123")
    resp = client.post("/predict", json={"age": 55, "wbc": 12.0})
    assert resp.status_code == 401


def test_predict_accepts_correct_api_key(client, monkeypatch):
    monkeypatch.setenv("PENUX_AP_API_KEY", "secret123")
    monkeypatch.delenv("PENUX_AP_MODEL_PATH", raising=False)
    import api.main as m
    m._model = None
    resp = client.post("/predict", json={"age": 55, "wbc": 12.0}, headers={"X-API-Key": "secret123"})
    assert resp.status_code == 200


def test_predict_rejects_wrong_api_key(client, monkeypatch):
    monkeypatch.setenv("PENUX_AP_API_KEY", "secret123")
    resp = client.post("/predict", json={"age": 55, "wbc": 12.0}, headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_oversized_body_rejected(client):
    from api.security import MAX_BODY_BYTES
    resp = client.post(
        "/predict",
        json={"age": 55},
        headers={"content-length": str(MAX_BODY_BYTES + 1)},
    )
    assert resp.status_code == 413


def test_rate_limit_blocks_after_threshold():
    """Exercises RateLimitMiddleware directly on a minimal app, independent
    of api.main, to avoid brittle module-reload interactions with other
    tests that also import api.main."""
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient as StarletteTestClient

    from api.security import RateLimitMiddleware

    async def homepage(request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/thing", homepage)])
    app.add_middleware(RateLimitMiddleware, max_requests=3, window_seconds=60)
    client = StarletteTestClient(app)

    statuses = [client.get("/thing").status_code for _ in range(5)]
    assert statuses.count(429) == 2
    assert statuses[:3] == [200, 200, 200]
