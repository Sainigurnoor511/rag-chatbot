from fastapi import Header, HTTPException

from app.config.settings import settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency: enforce X-API-Key when API_KEYS is configured.

    If no keys are configured (settings.API_KEYS empty), auth is disabled (dev mode).
    """
    allowed = settings.api_keys_list()
    if not allowed:
        return
    if x_api_key not in allowed:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
