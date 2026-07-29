import hashlib

from app.database.redis import redis_client
from app.config.logger import logger


def make_key(channel_id: str, message: str, filename: str | None) -> str:
    raw = f"{channel_id}|{(filename or '').lower()}|{message.strip().lower()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"qcache:{digest}"


def get_cached(channel_id: str, message: str, filename: str | None) -> str | None:
    if redis_client is None:
        return None
    try:
        value = redis_client.get(make_key(channel_id, message, filename))
        if value is None:
            return None
        return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value
    except Exception as e:
        logger.error(f"query_cache get failed: {e}")
        return None


def set_cached(channel_id: str, message: str, filename: str | None, answer: str, ttl: int) -> None:
    if redis_client is None:
        return
    try:
        redis_client.setex(make_key(channel_id, message, filename), ttl, answer)
    except Exception as e:
        logger.error(f"query_cache set failed: {e}")
