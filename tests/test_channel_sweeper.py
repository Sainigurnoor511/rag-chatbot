import os
import time
import app.repository.channel_sweeper as sweeper


def test_sweep_removes_orphaned_old_dirs(monkeypatch, tmp_path, fake_redis):
    monkeypatch.setattr(sweeper.settings, "EMBEDDING_DIR", str(tmp_path))
    monkeypatch.setattr(sweeper.settings, "CHANNEL_TTL_SECONDS", 1)

    live = tmp_path / "live-chan"
    live.mkdir()
    fake_redis.hset("channel:live-chan:docs", "doc", "{}")

    orphan = tmp_path / "orphan-chan"
    orphan.mkdir()
    old = time.time() - 3600
    os.utime(orphan, (old, old))

    removed = sweeper.sweep_once()
    assert "orphan-chan" in removed
    assert not orphan.exists()
    assert live.exists()


def test_sweep_keeps_recent_orphans(monkeypatch, tmp_path, fake_redis):
    monkeypatch.setattr(sweeper.settings, "EMBEDDING_DIR", str(tmp_path))
    monkeypatch.setattr(sweeper.settings, "CHANNEL_TTL_SECONDS", 3600)
    recent = tmp_path / "recent-chan"
    recent.mkdir()
    removed = sweeper.sweep_once()
    assert removed == []
    assert recent.exists()
