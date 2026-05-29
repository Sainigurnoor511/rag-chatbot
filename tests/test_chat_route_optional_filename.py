import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import app.routes.rag_routes as routes_mod


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(routes_mod.RAGController, "__init__", lambda self: None)
    monkeypatch.setattr(
        routes_mod.RAGController, "chat_with_document",
        lambda self, request: {"success": True, "message": "ok",
                               "data": {"user_input": request.get("message"),
                                        "bot_output": "hi", "filename_seen": request.get("filename")},
                               "error": None},
    )
    app = FastAPI()
    app.include_router(routes_mod.router)
    return TestClient(app)


def test_chat_works_without_filename(client):
    resp = client.post("/chat", json={"channel_id": "chan-1", "message": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["filename_seen"] is None


def test_chat_accepts_optional_filename(client):
    resp = client.post("/chat", json={"channel_id": "chan-1", "message": "hello", "filename": "a.pdf"})
    assert resp.status_code == 200
    assert resp.json()["data"]["filename_seen"] == "a.pdf"
