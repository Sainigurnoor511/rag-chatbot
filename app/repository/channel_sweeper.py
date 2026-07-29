import os
import time
import shutil
import asyncio

from app.database.redis import redis_client
from app.config.settings import settings
from app.config.logger import logger


def _manifest_exists(channel_id: str) -> bool:
    if redis_client is None:
        return False
    try:
        return bool(redis_client.exists(f"channel:{channel_id}:docs"))
    except Exception as e:
        logger.error(f"sweep manifest check failed for {channel_id}: {e}")
        return True  # err on the side of keeping data


def sweep_once() -> list[str]:
    """Delete channel dirs with no live manifest whose mtime is older than the TTL.

    Returns the list of removed channel ids.
    """
    base = settings.EMBEDDING_DIR
    removed: list[str] = []
    if not os.path.isdir(base):
        return removed
    cutoff = time.time() - settings.CHANNEL_TTL_SECONDS
    for name in os.listdir(base):
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            continue
        if _manifest_exists(name):
            continue
        try:
            if os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
                removed.append(name)
                logger.info(f"Swept expired channel dir: {name}")
        except Exception as e:
            logger.error(f"sweep failed for {name}: {e}")
    return removed


async def sweep_loop(interval_seconds: int = 300) -> None:
    """Background task: periodically sweep expired channel dirs."""
    while True:
        try:
            sweep_once()
        except Exception as e:
            logger.error(f"sweep_loop error: {e}")
        await asyncio.sleep(interval_seconds)
