from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator


def instrument(app: FastAPI, enabled: bool = True) -> None:
    """Mount Prometheus /metrics on the app when enabled."""
    if not enabled:
        return
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
