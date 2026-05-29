import io
import pytest
from fastapi.testclient import TestClient

import app.routes.rag_routes as routes_mod


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Stub embedding creation and manifest so the route runs without GPU/Redis.
    monkeypatch.setattr(routes_mod, "PROJECT_UPLOAD_DIRECTORY", str(tmp_path / "uploads"))
    monkeypatch.setattr(routes_mod, "PROJECT_EMBEDDING_DIRECTORY", str(tmp_path / "emb"))

    def fake_create(self, channel_id, file_path):
        return {"doc_id": "doc-x", "chunks": 3}

    monkeypatch.setattr(routes_mod.RAGController, "__init__", lambda self: None)
    monkeypatch.setattr(routes_mod.RAGController, "create_document_embeddings", fake_create)

    registered = {}
    monkeypatch.setattr(
        routes_mod, "register_document",
        lambda channel_id, doc_id, filename: registered.update(
            {"channel_id": channel_id, "doc_id": doc_id, "filename": filename}) or True,
    )

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(routes_mod.router)
    client = TestClient(app)
    client.registered = registered
    return client


def test_upload_requires_channel_id(client):
    resp = client.post("/upload", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")})
    assert resp.status_code == 422  # missing channel_id form field


def test_upload_registers_document(client):
    resp = client.post(
        "/upload",
        data={"channel_id": "chan-1"},
        files={"file": ("a.pdf", b"%PDF-1.4 data", "application/pdf")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["channel_id"] == "chan-1"
    assert client.registered["channel_id"] == "chan-1"
    assert client.registered["doc_id"] == "doc-x"
