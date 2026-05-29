# Production RAG — Phase 3: Production Cross-Cutting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the service for production: API-key auth, per-route rate limiting, Prometheus `/metrics`, an optional Redis query-result cache for chat, and a background sweep that deletes on-disk channel directories after their manifest TTL expires.

**Architecture:** New `app/middleware/auth.py` (API-key dependency) and `app/retrieval`/`app/repository` helpers stay as-is. Rate limiting via `slowapi` (limiter on `app.state`, per-route decorators). Metrics via `prometheus-fastapi-instrumentator` mounted in `main.py`. A Redis-backed query cache in `app/repository/query_cache.py` short-circuits identical chat queries. A `app/repository/channel_sweeper.py` background task (launched in the lifespan) removes orphaned channel dirs. All behavior is config-gated so dev runs stay frictionless.

**Tech Stack:** `slowapi`, `prometheus-fastapi-instrumentator`, FastAPI dependencies, Redis (fakeredis in tests), pytest.

**Environment note for implementers:** venv is uv-managed; use `.venv\Scripts\python.exe` (PowerShell: `& ".\.venv\Scripts\python.exe" -m pytest ...`), run from project root. Branch `feature/production-rag`. End commits with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Strict TDD.

---

### Task 1: Phase 3 settings + dependencies

**Files:** Modify `requirements.txt`, `app/config/settings.py`; Test `tests/test_settings_phase3.py`

- [ ] **Step 1: Add deps** — Append to `requirements.txt` (preserve UTF-16 encoding):
```
slowapi==0.1.9
prometheus-fastapi-instrumentator==7.0.0
```

- [ ] **Step 2: Install** — `& ".\.venv\Scripts\python.exe" -m pip install slowapi==0.1.9 prometheus-fastapi-instrumentator==7.0.0`

- [ ] **Step 3: Write failing test** — Create `tests/test_settings_phase3.py`:
```python
from app.config.settings import settings


def test_phase3_settings_defaults():
    assert settings.API_KEYS == ""          # empty => auth disabled (dev)
    assert settings.RATE_LIMIT_CHAT == "30/minute"
    assert settings.RATE_LIMIT_UPLOAD == "10/minute"
    assert settings.ENABLE_QUERY_CACHE is False
    assert settings.QUERY_CACHE_TTL == 300
    assert settings.METRICS_ENABLED is True


def test_api_keys_list_parses_csv(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", "k1, k2 ,k3")
    assert settings.api_keys_list() == ["k1", "k2", "k3"]


def test_api_keys_list_empty():
    monkeypatch_val = ""
    # default is empty -> no keys
    from app.config.settings import settings as s
    s.API_KEYS = ""
    assert s.api_keys_list() == []
```

- [ ] **Step 4: Run; expect FAIL.**

- [ ] **Step 5: Implement** — In `app/config/settings.py`, after the Phase 2 block add:
```python
    # Production cross-cutting (Phase 3)
    API_KEYS: str = ""  # comma-separated; empty disables auth (dev)
    RATE_LIMIT_CHAT: str = "30/minute"
    RATE_LIMIT_UPLOAD: str = "10/minute"
    ENABLE_QUERY_CACHE: bool = False
    QUERY_CACHE_TTL: int = 300
    METRICS_ENABLED: bool = True
```
And add a method to the `Settings` class:
```python
    def api_keys_list(self) -> list[str]:
        return [k.strip() for k in self.API_KEYS.split(",") if k.strip()]
```

- [ ] **Step 6: Run; expect PASS.** Then full suite.

- [ ] **Step 7: Commit**
```bash
git add requirements.txt app/config/settings.py tests/test_settings_phase3.py
git commit -m "feat: add Phase 3 cross-cutting settings and slowapi/prometheus deps"
```

---

### Task 2: API-key auth dependency

**Files:** Create `app/middleware/__init__.py`, `app/middleware/auth.py`; Test `tests/test_auth.py`

- [ ] **Step 1: Write failing test** — Create `tests/test_auth.py`:
```python
import pytest
from fastapi import FastAPI, Depends, HTTPException
from fastapi.testclient import TestClient
import app.middleware.auth as auth_mod


def _app():
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(auth_mod.require_api_key)])
    def protected():
        return {"ok": True}

    return TestClient(app)


def test_no_keys_configured_allows_all(monkeypatch):
    monkeypatch.setattr(auth_mod.settings, "API_KEYS", "")
    client = _app()
    assert client.get("/protected").status_code == 200


def test_valid_key_allowed(monkeypatch):
    monkeypatch.setattr(auth_mod.settings, "API_KEYS", "secret1,secret2")
    client = _app()
    assert client.get("/protected", headers={"X-API-Key": "secret2"}).status_code == 200


def test_missing_or_wrong_key_rejected(monkeypatch):
    monkeypatch.setattr(auth_mod.settings, "API_KEYS", "secret1")
    client = _app()
    assert client.get("/protected").status_code == 401
    assert client.get("/protected", headers={"X-API-Key": "nope"}).status_code == 401
```

- [ ] **Step 2: Run; expect FAIL (ModuleNotFoundError).**

- [ ] **Step 3: Implement** — Create `app/middleware/__init__.py` (empty). Create `app/middleware/auth.py`:
```python
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
```

- [ ] **Step 4: Run; expect PASS (3 tests).**

- [ ] **Step 5: Wire into routes** — In `app/routes/rag_routes.py`:
  - Add import: `from app.middleware.auth import require_api_key` and ensure `Depends` is imported from fastapi (add `Depends` to the `from fastapi import ...` line).
  - Add `dependencies=[Depends(require_api_key)]` to the `@router.post("/upload")` and `@router.post("/chat")` decorators. Example: `@router.post("/upload", dependencies=[Depends(require_api_key)])`.
  - Leave `/status` open. Gate `/sentry-debug`: change its body to raise 404 when in production:
    ```python
    @router.get("/sentry-debug")
    async def trigger_error():
        if settings.ENVIRONMENT == "production":
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not found")
        division_by_zero = 1 / 0
        return division_by_zero
    ```

- [ ] **Step 6: Run the full suite.** The existing `tests/test_upload_route.py` and `tests/test_chat_route_optional_filename.py` build their own bare `FastAPI()` and include the router; since `API_KEYS` defaults to empty, auth is disabled and those tests still pass. Confirm.

- [ ] **Step 7: Commit**
```bash
git add app/middleware/__init__.py app/middleware/auth.py app/routes/rag_routes.py tests/test_auth.py
git commit -m "feat: add API-key auth dependency and protect upload/chat"
```

---

### Task 3: Per-route rate limiting (slowapi)

**Files:** Create `app/middleware/rate_limit.py`; Modify `app/routes/rag_routes.py`, `main.py`; Test `tests/test_rate_limit.py`

- [ ] **Step 1: Write failing test** — Create `tests/test_rate_limit.py`:
```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import app.routes.rag_routes as routes_mod
from app.middleware.rate_limit import limiter, rate_limit_handler
from slowapi.errors import RateLimitExceeded


@pytest.fixture
def client(monkeypatch):
    # Force a tiny limit so the test is fast and deterministic.
    monkeypatch.setattr(routes_mod.settings, "RATE_LIMIT_CHAT", "2/minute")
    monkeypatch.setattr(routes_mod.RAGController, "__init__", lambda self: None)
    monkeypatch.setattr(routes_mod.RAGController, "chat_with_document",
                        lambda self, request: {"success": True, "message": "ok", "data": {}, "error": None})
    monkeypatch.setattr(routes_mod.settings, "API_KEYS", "")  # auth off
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
    app.include_router(routes_mod.router)
    return TestClient(app)


def test_chat_rate_limited_after_threshold(client):
    limiter.reset()
    payload = {"channel_id": "c1", "message": "hi"}
    assert client.post("/chat", json=payload).status_code == 200
    assert client.post("/chat", json=payload).status_code == 200
    # third call within the window exceeds 2/minute
    assert client.post("/chat", json=payload).status_code == 429
```

- [ ] **Step 2: Run; expect FAIL (ModuleNotFoundError).**

- [ ] **Step 3: Implement** — Create `app/middleware/rate_limit.py`:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse


def _key_func(request: Request) -> str:
    """Rate-limit per API key when present, else per client IP."""
    api_key = request.headers.get("X-API-Key")
    return api_key or get_remote_address(request)


limiter = Limiter(key_func=_key_func)


def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "message": "Rate limit exceeded. Please slow down.",
            "data": {},
            "error": {"code": 429, "message": str(exc.detail)},
        },
    )
```

- [ ] **Step 4: Apply limits in routes** — In `app/routes/rag_routes.py`:
  - Add imports: `from app.middleware.rate_limit import limiter` and `from fastapi import Request`.
  - The `/chat` and `/upload` endpoints must accept a `request: Request` parameter (slowapi requires it) and carry the limiter decorator. For `/chat`, rename the existing body model parameter to avoid clashing with the `Request`: the endpoint becomes:
    ```python
    @router.post("/chat", dependencies=[Depends(require_api_key)])
    @limiter.limit(settings.RATE_LIMIT_CHAT)
    async def chat(request: Request, body: ChatRequest):
        try:
            response = RAGController().chat_with_document(body.model_dump())
            return JSONResponse(content=response)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return create_error_response("Internal server error during chat processing", 500, {"details": str(e)})
    ```
    (Note: the `Request` param MUST be named `request` for slowapi to find it.)
  - For `/upload`, add `request: Request` as the FIRST parameter and the decorator `@limiter.limit(settings.RATE_LIMIT_UPLOAD)` directly under the `@router.post(...)` line. The existing `channel_id: str = Form(...)` and `file: UploadFile = File(...)` parameters stay after `request`.

- [ ] **Step 5: Wire limiter into the real app** — In `main.py`, after `app = FastAPI(...)` and before `app.include_router(...)`, add:
```python
from slowapi.errors import RateLimitExceeded
from app.middleware.rate_limit import limiter, rate_limit_handler

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
```

- [ ] **Step 6: Run the new test; expect PASS.** Then run the full suite. The pre-existing `tests/test_chat_route_optional_filename.py` and `tests/test_upload_route.py` build their own apps WITHOUT `app.state.limiter`; slowapi raises if a limited route runs without a configured limiter on app.state. To keep those tests working, they must set `app.state.limiter = limiter` and the handler too. **Update those two test files**: add the same `app.state.limiter = limiter` + `app.add_exception_handler(RateLimitExceeded, rate_limit_handler)` lines after creating their `FastAPI()` app, and add `limiter.reset()` at the start of each test to avoid cross-test limit bleed. Also update their request calls: `/chat` now takes the body as before (FastAPI still binds the `ChatRequest` from JSON via the `body` param), so no payload change is needed, but confirm the response still parses. Re-run until green.

- [ ] **Step 7: Commit**
```bash
git add app/middleware/rate_limit.py app/routes/rag_routes.py main.py tests/test_rate_limit.py tests/test_chat_route_optional_filename.py tests/test_upload_route.py
git commit -m "feat: add per-route rate limiting via slowapi"
```

---

### Task 4: Prometheus /metrics

**Files:** Modify `main.py`; Test `tests/test_metrics.py`

- [ ] **Step 1: Write failing test** — Create `tests/test_metrics.py`:
```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.observability.metrics import instrument


def test_metrics_endpoint_exposed_when_enabled():
    app = FastAPI()
    instrument(app, enabled=True)
    client = TestClient(app)
    # generate one request so counters exist
    @app.get("/ping")
    def ping():
        return {"ok": True}
    client.get("/ping")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "http_request" in resp.text or "http_requests" in resp.text


def test_metrics_disabled_returns_404():
    app = FastAPI()
    instrument(app, enabled=False)
    client = TestClient(app)
    assert client.get("/metrics").status_code == 404
```

- [ ] **Step 2: Run; expect FAIL (ModuleNotFoundError app.observability.metrics).**

- [ ] **Step 3: Implement** — Create `app/observability/__init__.py` (empty). Create `app/observability/metrics.py`:
```python
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator


def instrument(app: FastAPI, enabled: bool = True) -> None:
    """Mount Prometheus /metrics on the app when enabled."""
    if not enabled:
        return
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
```

- [ ] **Step 4: Run; expect PASS (2 tests).**

- [ ] **Step 5: Wire into main.py** — In `main.py`, after the limiter wiring and before/after `app.include_router(...)`, add:
```python
from app.observability.metrics import instrument
instrument(app, enabled=settings.METRICS_ENABLED)
```

- [ ] **Step 6: Run full suite.**

- [ ] **Step 7: Commit**
```bash
git add app/observability/__init__.py app/observability/metrics.py main.py tests/test_metrics.py
git commit -m "feat: expose Prometheus /metrics via instrumentator"
```

---

### Task 5: Optional Redis query-result cache for chat

**Files:** Create `app/repository/query_cache.py`; Modify `app/controller/rag_controller.py`; Test `tests/test_query_cache.py`

- [ ] **Step 1: Write failing test** — Create `tests/test_query_cache.py`:
```python
import app.repository.query_cache as qc


def test_cache_key_is_stable_and_scoped():
    k1 = qc.make_key("chan-1", "Hello There", "a.pdf")
    k2 = qc.make_key("chan-1", "hello there", "a.pdf")  # case-normalized
    k3 = qc.make_key("chan-2", "Hello There", "a.pdf")
    assert k1 == k2
    assert k1 != k3
    assert k1.startswith("qcache:")


def test_get_set_roundtrip(fake_redis):
    qc.set_cached("chan-1", "q", None, "the answer", ttl=300)
    assert qc.get_cached("chan-1", "q", None) == "the answer"


def test_get_missing_returns_none(fake_redis):
    assert qc.get_cached("chan-1", "absent", None) is None
```

Note: the `fake_redis` fixture (in conftest) monkeypatches `app.database.redis.redis_client`. `query_cache` must import `redis_client` by name at module top so the fixture patches it (the fixture already also patches `app.repository.channel_repository.redis_client`; extend it). **Update `tests/conftest.py`** `fake_redis` fixture to also patch `app.repository.query_cache.redis_client` (wrap in try/except import like the existing channel_repository patch).

- [ ] **Step 2: Run; expect FAIL.**

- [ ] **Step 3: Implement** — Create `app/repository/query_cache.py`:
```python
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
```

Also update `tests/conftest.py` `fake_redis` fixture, adding (after the channel_repository patch):
```python
    try:
        import app.repository.query_cache as qcache
        monkeypatch.setattr(qcache, "redis_client", client, raising=False)
    except Exception:
        pass
```

- [ ] **Step 4: Run; expect PASS (3 tests).**

- [ ] **Step 5: Wire into chat (config-gated)** — In `app/controller/rag_controller.py`, add import `from app.repository import query_cache`. In `chat_with_document`, AFTER computing `user_input` and confirming the vectorstore exists, BEFORE contextualization, add a cache check; and after generating `output`, store it. Concretely, right after the `user_input = message.strip()` and the vectorstore-None check:
```python
            if settings.ENABLE_QUERY_CACHE:
                cached = query_cache.get_cached(channel_id, user_input, filename)
                if cached is not None:
                    logger.info(f"Query cache hit for channel {channel_id}")
                    return {
                        "success": True,
                        "message": "Response generated successfully (cached)",
                        "data": {"user_input": user_input, "bot_output": cached},
                        "error": None,
                    }
```
And immediately AFTER `output = ...` is finalized (after the if/else docs guard, before appending to history), add:
```python
            if settings.ENABLE_QUERY_CACHE:
                query_cache.set_cached(channel_id, user_input, filename, output, settings.QUERY_CACHE_TTL)
```

- [ ] **Step 6: Add a chat cache test** — Append to `tests/test_chat_hybrid.py`:
```python
def test_chat_returns_cached_answer_when_enabled(patched, monkeypatch):
    controller, saved = patched
    monkeypatch.setattr(ctrl_mod.settings, "ENABLE_QUERY_CACHE", True)
    monkeypatch.setattr(ctrl_mod.query_cache, "get_cached",
                        lambda channel_id, message, filename: "CACHED ANSWER")
    resp = controller.chat_with_document({"channel_id": "chan-1", "message": "hi", "filename": None})
    assert resp["success"] is True
    assert resp["data"]["bot_output"] == "CACHED ANSWER"
    assert "(cached)" in resp["message"]
```
(`ctrl_mod.query_cache` is the imported module; `ctrl_mod.settings` is the settings singleton.)

- [ ] **Step 7: Run new tests + full suite; expect PASS.**

- [ ] **Step 8: Commit**
```bash
git add app/repository/query_cache.py app/controller/rag_controller.py tests/test_query_cache.py tests/test_chat_hybrid.py tests/conftest.py
git commit -m "feat: add optional Redis query-result cache for chat"
```

---

### Task 6: On-disk expired-channel sweep

**Files:** Create `app/repository/channel_sweeper.py`; Modify `main.py`; Test `tests/test_channel_sweeper.py`

- [ ] **Step 1: Write failing test** — Create `tests/test_channel_sweeper.py`:
```python
import os
import time
import app.repository.channel_sweeper as sweeper


def test_sweep_removes_orphaned_old_dirs(monkeypatch, tmp_path, fake_redis):
    monkeypatch.setattr(sweeper.settings, "EMBEDDING_DIR", str(tmp_path))
    monkeypatch.setattr(sweeper.settings, "CHANNEL_TTL_SECONDS", 1)

    # channel with a live manifest entry -> must be kept
    live = tmp_path / "live-chan"
    live.mkdir()
    fake_redis.hset("channel:live-chan:docs", "doc", "{}")

    # orphaned channel dir, no manifest, old mtime -> must be removed
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
    recent.mkdir()  # fresh mtime
    removed = sweeper.sweep_once()
    assert removed == []
    assert recent.exists()
```

Note: extend the `fake_redis` fixture in conftest to also patch `app.repository.channel_sweeper.redis_client` (same try/except pattern). Add that to conftest in this task.

- [ ] **Step 2: Run; expect FAIL.**

- [ ] **Step 3: Implement** — Create `app/repository/channel_sweeper.py`:
```python
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
```

Update `tests/conftest.py` `fake_redis` fixture, adding:
```python
    try:
        import app.repository.channel_sweeper as csweeper
        monkeypatch.setattr(csweeper, "redis_client", client, raising=False)
    except Exception:
        pass
```

- [ ] **Step 4: Run; expect PASS (2 tests).**

- [ ] **Step 5: Launch in lifespan** — In `main.py`, re-add `import asyncio` at the top (it was removed in Phase 1). Inside the `lifespan` startup `try` block, after `rag_utilities = RAGUtilities()`, add:
```python
        from app.repository.channel_sweeper import sweep_loop
        asyncio.create_task(sweep_loop())
```

- [ ] **Step 6: Run full suite; confirm `import main` still works** (`& ".\.venv\Scripts\python.exe" -c "import main; print(hasattr(main,'app'))"`).

- [ ] **Step 7: Commit**
```bash
git add app/repository/channel_sweeper.py main.py tests/test_channel_sweeper.py tests/conftest.py
git commit -m "feat: add background sweep for expired on-disk channel dirs"
```

---

## Self-Review

**Spec coverage (Phase 3 portion):**
- API-key auth (X-API-Key, API_KEYS env, /status open, /sentry-debug gated) → Task 2 ✓
- Rate limiting (slowapi, config-driven per-route) → Task 3 ✓
- Observability: Prometheus /metrics → Task 4 ✓ (structured retrieval traces already emitted by `hybrid_retriever` in Phase 2)
- Caching: optional Redis query-result cache → Task 5 ✓ (Chroma store already cached in `VECTOR_STORE_CACHE`; reranker/embedding class-cached in Phase 2)
- On-disk expired-channel sweep → Task 6 ✓
- Config knobs (API_KEYS, rate limits, cache flags, METRICS_ENABLED) → Task 1 ✓

**Deferred:** eval harness (Phase 4); per-channel in-process BM25 LRU (current per-query rebuild is acceptable; not adding complexity here).

**Placeholder scan:** No TBD/TODO; every code step is complete.

**Type consistency:** `settings.api_keys_list()` used in Tasks 1, 2. `limiter`/`rate_limit_handler` from `app.middleware.rate_limit` used in Task 3 + tests + main. `query_cache.make_key/get_cached/set_cached` consistent across Task 5 + tests + controller. `channel_sweeper.sweep_once()/sweep_loop()` consistent across Task 6 + main. The `fake_redis` conftest fixture is extended additively in Tasks 5 and 6.

**Known risk:** Task 3 changes the `/chat` endpoint signature (adds `request: Request`, renames body param to `body`) and `/upload` (adds `request: Request` first) — the pre-existing route tests are updated in the same task to wire `app.state.limiter`. Implementers must run the full suite after Task 3 and fix any route-test fallout before committing.
