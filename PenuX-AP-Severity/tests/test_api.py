"""Tests for FastAPI research endpoint."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "warning" in data
    assert "clinical" in data["warning"].lower()


def test_predict_no_model_configured(client, monkeypatch):
    monkeypatch.delenv("PENUX_AP_MODEL_PATH", raising=False)
    import api.main as m
    m._model = None
    resp = client.post("/predict", json={"age": 55, "wbc": 12.0})
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] is not None
    assert "PENUX_AP_MODEL_PATH" in data["error"]
