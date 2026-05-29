from app.config.settings import settings


def test_phase3_settings_defaults():
    assert settings.API_KEYS == ""
    assert settings.RATE_LIMIT_CHAT == "30/minute"
    assert settings.RATE_LIMIT_UPLOAD == "10/minute"
    assert settings.ENABLE_QUERY_CACHE is False
    assert settings.QUERY_CACHE_TTL == 300
    assert settings.METRICS_ENABLED is True


def test_api_keys_list_parses_csv(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", "k1, k2 ,k3")
    assert settings.api_keys_list() == ["k1", "k2", "k3"]


def test_api_keys_list_empty(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", "")
    assert settings.api_keys_list() == []
