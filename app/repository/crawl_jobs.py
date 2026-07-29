import json

from app.database.redis import redis_client
from app.config.logger import logger
from app.config.settings import settings


def _job_key(job_id: str) -> str:
    return f"crawl_job:{job_id}"


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def create_job(job_id: str, channel_id: str, base_url: str) -> bool:
    """Create a new crawl job record with status=queued."""
    if redis_client is None:
        logger.warning("Redis unavailable; cannot create crawl job")
        return False
    job = {
        "channel_id": channel_id,
        "base_url": base_url,
        "status": "queued",
        "pages_found": 0,
        "pages_processed": 0,
        "error": None,
    }
    try:
        redis_client.setex(_job_key(job_id), settings.CRAWL_JOB_TTL_SECONDS, json.dumps(job))
        return True
    except Exception as e:
        logger.error(f"create_job failed: {e}")
        return False


def update_job(job_id: str, **fields) -> bool:
    """Merge fields into an existing job record, refreshing its TTL."""
    if redis_client is None:
        logger.warning("Redis unavailable; cannot update crawl job")
        return False
    try:
        existing = get_job(job_id) or {}
        existing.update(fields)
        redis_client.setex(_job_key(job_id), settings.CRAWL_JOB_TTL_SECONDS, json.dumps(existing))
        return True
    except Exception as e:
        logger.error(f"update_job failed: {e}")
        return False


def get_job(job_id: str) -> dict | None:
    """Return the job record dict, or None if missing/unavailable."""
    if redis_client is None:
        return None
    try:
        raw = redis_client.get(_job_key(job_id))
        if raw is None:
            return None
        return json.loads(_decode(raw))
    except Exception as e:
        logger.error(f"get_job failed: {e}")
        return None
