from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.observability.metrics import instrument


def test_metrics_endpoint_exposed_when_enabled():
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    instrument(app, enabled=True)
    client = TestClient(app)
    client.get("/ping")  # generate one request so counters exist
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "http_request" in resp.text or "http_requests" in resp.text


def test_metrics_disabled_returns_404():
    app = FastAPI()
    instrument(app, enabled=False)
    client = TestClient(app)
    assert client.get("/metrics").status_code == 404
