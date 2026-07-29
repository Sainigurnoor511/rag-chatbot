from unittest.mock import patch

from app.repository import crawl_jobs


def test_create_and_get_job(fake_redis):
    with patch("app.repository.crawl_jobs.redis_client", fake_redis):
        ok = crawl_jobs.create_job("job1", channel_id="chan1", base_url="https://example.com")
        assert ok is True

        job = crawl_jobs.get_job("job1")

    assert job["channel_id"] == "chan1"
    assert job["base_url"] == "https://example.com"
    assert job["status"] == "queued"
    assert job["pages_found"] == 0
    assert job["pages_processed"] == 0


def test_update_job_merges_fields(fake_redis):
    with patch("app.repository.crawl_jobs.redis_client", fake_redis):
        crawl_jobs.create_job("job2", channel_id="chan1", base_url="https://example.com")
        crawl_jobs.update_job("job2", status="crawling", pages_found=5)

        job = crawl_jobs.get_job("job2")

    assert job["status"] == "crawling"
    assert job["pages_found"] == 5
    assert job["base_url"] == "https://example.com"  # untouched fields survive


def test_get_job_returns_none_when_missing(fake_redis):
    with patch("app.repository.crawl_jobs.redis_client", fake_redis):
        job = crawl_jobs.get_job("does-not-exist")

    assert job is None


def test_create_job_returns_false_when_redis_unavailable():
    with patch("app.repository.crawl_jobs.redis_client", None):
        ok = crawl_jobs.create_job("job3", channel_id="chan1", base_url="https://example.com")

    assert ok is False
