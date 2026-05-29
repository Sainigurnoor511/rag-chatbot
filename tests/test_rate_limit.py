import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import app.routes.rag_routes as routes_mod
from app.middleware.rate_limit import limiter, rate_limit_handler
from slowapi.errors import RateLimitExceeded


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(routes_mod.settings, "RATE_LIMIT_CHAT", "2/minute")
    monkeypatch.setattr(routes_mod.RAGController, "__init__", lambda self: None)
    monkeypatch.setattr(routes_mod.RAGController, "chat_with_document",
                        lambda self, request: {"success": True, "message": "ok", "data": {}, "error": None})
    monkeypatch.setattr(routes_mod.settings, "API_KEYS", "")
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
    app.include_router(routes_mod.router)
    return TestClient(app)


def test_chat_rate_limited_after_threshold(client):
    limiter.reset()
    payload = {"channel_id": "c1", "message": "hi"}
    assert client.post("/chat", json=payload).status_code == 200
    assert client.post("/chat", json=payload).status_code == 200
    assert client.post("/chat", json=payload).status_code == 429
