import fakeredis
import pytest


@pytest.fixture
def fake_redis(monkeypatch):
    """Replace the global redis_client with an in-memory fake."""
    client = fakeredis.FakeStrictRedis(decode_responses=False)
    monkeypatch.setattr("app.database.redis.redis_client", client)
    # Re-point modules that imported the client by name.
    import app.repository.channel_repository as repo
    monkeypatch.setattr(repo, "redis_client", client, raising=False)
    try:
        import app.repository.query_cache as qcache
        monkeypatch.setattr(qcache, "redis_client", client, raising=False)
    except Exception:
        pass
    try:
        import app.repository.channel_sweeper as csweeper
        monkeypatch.setattr(csweeper, "redis_client", client, raising=False)
    except Exception:
        pass
    return client
