import json

from app.database.redis import redis_client
from app.config.logger import logger
from app.config.settings import settings


def _manifest_key(channel_id: str) -> str:
    return f"channel:{channel_id}:docs"


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def register_document(channel_id: str, doc_id: str, filename: str) -> bool:
    """Add/replace a document entry in the channel manifest and refresh TTL."""
    if redis_client is None:
        logger.warning("Redis unavailable; cannot register document")
        return False
    key = _manifest_key(channel_id)
    try:
        redis_client.hset(key, doc_id, json.dumps({"filename": filename}))
        redis_client.expire(key, settings.CHANNEL_TTL_SECONDS)
        return True
    except Exception as e:
        logger.error(f"register_document failed: {e}")
        return False


def list_documents(channel_id: str) -> list[dict]:
    """Return [{doc_id, filename}, ...] for the channel (empty if none/unavailable)."""
    if redis_client is None:
        return []
    key = _manifest_key(channel_id)
    try:
        raw = redis_client.hgetall(key)
        docs = []
        for doc_id, payload in raw.items():
            meta = json.loads(_decode(payload))
            docs.append({"doc_id": _decode(doc_id), "filename": meta["filename"]})
        return docs
    except Exception as e:
        logger.error(f"list_documents failed: {e}")
        return []


def remove_channel(channel_id: str) -> bool:
    """Delete the channel's manifest key."""
    if redis_client is None:
        return False
    try:
        redis_client.delete(_manifest_key(channel_id))
        return True
    except Exception as e:
        logger.error(f"remove_channel failed: {e}")
        return False
