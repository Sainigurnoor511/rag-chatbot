import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
import app.middleware.auth as auth_mod


def _app():
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(auth_mod.require_api_key)])
    def protected():
        return {"ok": True}

    return TestClient(app)


def test_no_keys_configured_allows_all(monkeypatch):
    monkeypatch.setattr(auth_mod.settings, "API_KEYS", "")
    client = _app()
    assert client.get("/protected").status_code == 200


def test_valid_key_allowed(monkeypatch):
    monkeypatch.setattr(auth_mod.settings, "API_KEYS", "secret1,secret2")
    client = _app()
    assert client.get("/protected", headers={"X-API-Key": "secret2"}).status_code == 200


def test_missing_or_wrong_key_rejected(monkeypatch):
    monkeypatch.setattr(auth_mod.settings, "API_KEYS", "secret1")
    client = _app()
    assert client.get("/protected").status_code == 401
    assert client.get("/protected", headers={"X-API-Key": "nope"}).status_code == 401
