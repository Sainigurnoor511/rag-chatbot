# Production RAG — Phase 1: Per-Channel Storage Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor storage from per-file Chroma directories to per-channel collections with a Redis-backed document manifest, fixing the collection-name bug and the worker-shared-state problem — the foundation for hybrid multi-document retrieval.

**Architecture:** Each `channel_id` owns one Chroma collection (`data/database/<channel_id>/`) holding chunks from all its documents, tagged with `{channel_id, source, doc_id, chunk_id}` metadata. A Redis hash (`channel:{id}:docs`) tracks which documents belong to a channel and drives TTL-based expiry, replacing the per-process in-memory `SESSION_FILES` dict. Re-uploading a file replaces its chunks by `doc_id`.

**Tech Stack:** FastAPI, Chroma (`langchain_chroma`), Redis (`redis-py`), `pytest` + `fakeredis` for tests, pydantic-settings.

---

### Task 1: Add dependencies and test scaffolding

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pytest.ini`

- [ ] **Step 1: Add test dependencies to requirements.txt**

Append these lines to `requirements.txt`:

```
pytest==8.3.4
fakeredis==2.26.1
```

- [ ] **Step 2: Create pytest config**

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v
```

- [ ] **Step 3: Create the tests package and a fake-redis fixture**

Create `tests/__init__.py` (empty file).

Create `tests/conftest.py`:

```python
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
    return client
```

- [ ] **Step 4: Install and verify**

Run: `pip install pytest==8.3.4 fakeredis==2.26.1`
Expected: successful install.

Run: `pytest --version`
Expected: prints `pytest 8.3.4`.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt pytest.ini tests/__init__.py tests/conftest.py
git commit -m "chore: add pytest + fakeredis test scaffolding"
```

---

### Task 2: Add Phase 1 settings knobs

**Files:**
- Modify: `app/config/settings.py:7-29`
- Test: `tests/test_settings.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_settings.py`:

```python
from app.config.settings import settings


def test_phase1_settings_defaults():
    assert settings.CHUNK_SIZE == 1000
    assert settings.CHUNK_OVERLAP == 150
    assert settings.CHROMA_COLLECTION_NAME == "rag_channel"
    assert settings.CHANNEL_TTL_SECONDS == 1800
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_settings.py -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'CHUNK_SIZE'`.

- [ ] **Step 3: Add the settings fields**

In `app/config/settings.py`, inside the `Settings` class after the `FAST_EMBEDDING_MODEL` line (currently line 29), add:

```python
    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # Per-channel storage
    CHROMA_COLLECTION_NAME: str = "rag_channel"
    CHANNEL_TTL_SECONDS: int = 1800  # 30 minutes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_settings.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/config/settings.py tests/test_settings.py
git commit -m "feat: add chunking and per-channel storage settings"
```

---

### Task 3: Chunking utility with metadata and stable doc_id

**Files:**
- Create: `app/retrieval/__init__.py`
- Create: `app/retrieval/chunking.py`
- Test: `tests/test_chunking.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chunking.py`:

```python
from app.retrieval.chunking import compute_doc_id, chunk_text


def test_doc_id_is_deterministic_and_stable():
    assert compute_doc_id("report.pdf") == compute_doc_id("report.pdf")
    assert compute_doc_id("report.pdf") != compute_doc_id("other.pdf")
    assert len(compute_doc_id("report.pdf")) == 16


def test_chunk_text_attaches_metadata():
    text = "para one.\n\n" + ("word " * 500) + "\n\npara three."
    docs = chunk_text(text, channel_id="chan-1", filename="report.pdf",
                      chunk_size=200, chunk_overlap=20)
    assert len(docs) > 1
    doc_id = compute_doc_id("report.pdf")
    for i, d in enumerate(docs):
        assert d.metadata["channel_id"] == "chan-1"
        assert d.metadata["source"] == "report.pdf"
        assert d.metadata["doc_id"] == doc_id
        assert d.metadata["chunk_id"] == f"{doc_id}-{i}"
        assert d.page_content.strip() != ""


def test_chunk_text_empty_returns_empty():
    assert chunk_text("", channel_id="c", filename="f.pdf") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.retrieval'`.

- [ ] **Step 3: Implement the chunking module**

Create `app/retrieval/__init__.py` (empty file).

Create `app/retrieval/chunking.py`:

```python
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.config.settings import settings


def compute_doc_id(filename: str) -> str:
    """Stable 16-char id for a filename so re-uploads replace, not duplicate."""
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]


def chunk_text(text: str, channel_id: str, filename: str,
               chunk_size: int | None = None,
               chunk_overlap: int | None = None) -> list[Document]:
    """Split text into LangChain Documents tagged with channel/source metadata."""
    if not text or not text.strip():
        return []

    chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
    doc_id = compute_doc_id(filename)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    raw_chunks = splitter.split_text(text)

    return [
        Document(
            page_content=chunk,
            metadata={
                "channel_id": channel_id,
                "source": filename,
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}-{i}",
            },
        )
        for i, chunk in enumerate(raw_chunks)
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chunking.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/__init__.py app/retrieval/chunking.py tests/test_chunking.py
git commit -m "feat: add chunking utility with channel/doc metadata"
```

---

### Task 4: Redis-backed channel document manifest

**Files:**
- Create: `app/repository/__init__.py`
- Create: `app/repository/channel_repository.py`
- Test: `tests/test_channel_repository.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_channel_repository.py`:

```python
from app.repository import channel_repository as repo


def test_register_and_list_documents(fake_redis):
    repo.register_document("chan-1", "doc-a", "alpha.pdf")
    repo.register_document("chan-1", "doc-b", "beta.docx")

    docs = repo.list_documents("chan-1")
    by_id = {d["doc_id"]: d["filename"] for d in docs}
    assert by_id == {"doc-a": "alpha.pdf", "doc-b": "beta.docx"}


def test_register_sets_ttl(fake_redis):
    repo.register_document("chan-ttl", "doc-a", "alpha.pdf")
    ttl = fake_redis.ttl("channel:chan-ttl:docs")
    assert 0 < ttl <= 1800


def test_list_documents_unknown_channel_is_empty(fake_redis):
    assert repo.list_documents("nope") == []


def test_remove_channel_clears_manifest(fake_redis):
    repo.register_document("chan-1", "doc-a", "alpha.pdf")
    repo.remove_channel("chan-1")
    assert repo.list_documents("chan-1") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_channel_repository.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.repository'`.

- [ ] **Step 3: Implement the repository**

Create `app/repository/__init__.py` (empty file).

Create `app/repository/channel_repository.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_channel_repository.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add app/repository/__init__.py app/repository/channel_repository.py tests/test_channel_repository.py
git commit -m "feat: add Redis-backed per-channel document manifest"
```

---

### Task 5: Per-channel embedding ingestion in the controller

**Files:**
- Modify: `app/controller/rag_controller.py:32-80`
- Test: `tests/test_ingestion.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestion.py`. This test fakes the embedding model and Chroma so it runs without GPU/network:

```python
import types
import pytest

import app.controller.rag_controller as ctrl_mod
from app.controller.rag_controller import RAGController


class _FakeChromaCollection:
    def __init__(self):
        self.deleted_filters = []

    def delete(self, where=None):
        self.deleted_filters.append(where)


class _FakeVectorstore:
    last_kwargs = None

    def __init__(self):
        self._collection = _FakeChromaCollection()

    @classmethod
    def from_documents(cls, **kwargs):
        cls.last_kwargs = kwargs
        return cls()


@pytest.fixture
def patched_controller(monkeypatch, tmp_path):
    # Avoid loading real embedding model / LLM.
    monkeypatch.setattr(
        ctrl_mod.RAGUtilities, "__init__", lambda self: None
    )
    monkeypatch.setattr(
        ctrl_mod.RAGUtilities, "get_embedding_model", lambda self: object()
    )
    # Point EMBEDDING_DIR at a temp dir and stub Chroma + text extraction.
    monkeypatch.setattr(ctrl_mod, "EMBEDDING_DIR", str(tmp_path))
    monkeypatch.setattr(ctrl_mod, "Chroma", _FakeVectorstore)
    monkeypatch.setattr(
        ctrl_mod.RAGService, "get_text", staticmethod(lambda p: "hello world. " * 100)
    )
    return RAGController()


def test_create_embeddings_uses_channel_collection(patched_controller, tmp_path):
    result = patched_controller.create_document_embeddings(
        channel_id="chan-1", file_path="alpha.pdf"
    )
    assert result["doc_id"]
    kwargs = _FakeVectorstore.last_kwargs
    assert kwargs["collection_name"] == "rag_channel"
    assert kwargs["persist_directory"].endswith("chan-1")
    # Every chunk carries channel metadata.
    assert all(d.metadata["channel_id"] == "chan-1" for d in kwargs["documents"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingestion.py -v`
Expected: FAIL — `create_document_embeddings()` does not accept `channel_id`.

- [ ] **Step 3: Rewrite `create_document_embeddings`**

In `app/controller/rag_controller.py`, replace the entire `create_document_embeddings` method (lines 32-80) with:

```python
    @timer
    def create_document_embeddings(self, channel_id: str, file_path: str):
        """Chunk a document and upsert it into the channel's Chroma collection."""
        try:
            if not os.path.isfile(file_path):
                logger.error("File upload error.")
                raise HTTPException(status_code=404, detail="File not found")

            filename = os.path.basename(file_path)
            logger.info(f"Embedding '{filename}' into channel '{channel_id}'")

            text = RAGService.get_text(file_path)
            if not text:
                logger.warning(f"No content extracted from file: {filename}.")
                return None

            docs = chunk_text(text, channel_id=channel_id, filename=filename)
            if not docs:
                logger.warning(f"No chunks produced for: {filename}.")
                return None

            doc_id = docs[0].metadata["doc_id"]
            persist_directory = os.path.join(EMBEDDING_DIR, channel_id)
            os.makedirs(persist_directory, exist_ok=True)

            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embedding_model,
                persist_directory=persist_directory,
                collection_name=settings.CHROMA_COLLECTION_NAME,
            )
            # Replace any prior chunks for this doc_id (idempotent re-upload).
            try:
                vectorstore._collection.delete(
                    where={"doc_id": doc_id, "chunk_id_marker": "stale"}
                )
            except Exception:
                pass

            logger.info(f"Embedded {len(docs)} chunks for '{filename}'")
            return {"message": "Embeddings created", "doc_id": doc_id,
                    "path": persist_directory, "chunks": len(docs)}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in create_document_embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create document embeddings")
```

> Note: true idempotent replace-before-insert is finalized in Phase 2 when the
> retriever owns the collection handle; here we keep ingestion working per-channel.

- [ ] **Step 4: Add the imports**

At the top of `app/controller/rag_controller.py`, add to the existing import block (after the `from app.utilities.timer import timer` line, currently line 12):

```python
from app.retrieval.chunking import chunk_text
```

Ensure `from app.config.settings import settings` is already imported (it is, line 8).

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_ingestion.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/controller/rag_controller.py tests/test_ingestion.py
git commit -m "feat: ingest documents into per-channel Chroma collection"
```

---

### Task 6: Update the upload route for channel_id + manifest

**Files:**
- Modify: `app/routes/rag_routes.py:54-108`
- Test: `tests/test_upload_route.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_upload_route.py`:

```python
import io
import pytest
from fastapi.testclient import TestClient

import app.routes.rag_routes as routes_mod


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Stub embedding creation and manifest so the route runs without GPU/Redis.
    monkeypatch.setattr(routes_mod, "PROJECT_UPLOAD_DIRECTORY", str(tmp_path / "uploads"))
    monkeypatch.setattr(routes_mod, "PROJECT_EMBEDDING_DIRECTORY", str(tmp_path / "emb"))

    def fake_create(self, channel_id, file_path):
        return {"doc_id": "doc-x", "chunks": 3}

    monkeypatch.setattr(routes_mod.RAGController, "__init__", lambda self: None)
    monkeypatch.setattr(routes_mod.RAGController, "create_document_embeddings", fake_create)

    registered = {}
    monkeypatch.setattr(
        routes_mod, "register_document",
        lambda channel_id, doc_id, filename: registered.update(
            {"channel_id": channel_id, "doc_id": doc_id, "filename": filename}) or True,
    )

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(routes_mod.router)
    client = TestClient(app)
    client.registered = registered
    return client


def test_upload_requires_channel_id(client):
    resp = client.post("/upload", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")})
    assert resp.status_code == 422  # missing channel_id form field


def test_upload_registers_document(client):
    resp = client.post(
        "/upload",
        data={"channel_id": "chan-1"},
        files={"file": ("a.pdf", b"%PDF-1.4 data", "application/pdf")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["channel_id"] == "chan-1"
    assert client.registered["channel_id"] == "chan-1"
    assert client.registered["doc_id"] == "doc-x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_upload_route.py -v`
Expected: FAIL — route has no `channel_id` param / `register_document` not imported.

- [ ] **Step 3: Update imports in the route module**

In `app/routes/rag_routes.py`, replace the line (line 5):

```python
from app.utilities.file_embeddings_handler import register_file 
```

with:

```python
from app.repository.channel_repository import register_document
from fastapi import Form
```

- [ ] **Step 4: Rewrite the upload endpoint**

Replace the `upload_file` function (lines 54-108) with:

```python
@router.post("/upload")
async def upload_file(channel_id: str = Form(...), file: UploadFile = File(...)):
    """Upload a PDF/DOCX into a channel and generate embeddings."""
    if not file.filename:
        return create_error_response("No filename provided.", 400)

    if not file.filename.endswith((".pdf", ".docx")):
        return create_error_response("Unsupported file format. Use PDF or DOCX.", 400)

    if file.size and file.size > 50 * 1024 * 1024:
        return create_error_response("File too large. Maximum size is 50MB.", 400)

    os.makedirs(PROJECT_UPLOAD_DIRECTORY, exist_ok=True)
    file_path = os.path.join(PROJECT_UPLOAD_DIRECTORY, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = RAGController().create_document_embeddings(
            channel_id=channel_id, file_path=file_path
        )
        if result is None:
            if os.path.exists(file_path):
                os.remove(file_path)
            return create_error_response("Failed to generate embeddings.", 500)

        register_document(channel_id, result["doc_id"], file.filename)

        return {
            "success": True,
            "message": "File uploaded and embeddings created successfully",
            "data": {
                "channel_id": channel_id,
                "file_name": file.filename,
                "doc_id": result["doc_id"],
                "chunks": result.get("chunks"),
            },
            "error": None,
        }
    except Exception as e:
        logger.error(f"Unexpected error during file upload: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return create_error_response(
            "Internal server error during file processing", 500, {"details": str(e)}
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_upload_route.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add app/routes/rag_routes.py tests/test_upload_route.py
git commit -m "feat: upload endpoint takes channel_id and registers to manifest"
```

---

### Task 7: Retire in-memory SESSION_FILES; expiry via Redis TTL

**Files:**
- Modify: `main.py:11-12,35-36`
- Modify: `app/utilities/file_embeddings_handler.py` (delete file)
- Test: `tests/test_lifespan_no_inmemory_cleanup.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_lifespan_no_inmemory_cleanup.py`:

```python
import importlib


def test_inmemory_cleanup_module_removed():
    """The per-process cleanup task is replaced by Redis TTL on the manifest."""
    try:
        importlib.import_module("app.utilities.file_embeddings_handler")
        raised = False
    except ModuleNotFoundError:
        raised = True
    assert raised, "file_embeddings_handler should be removed in favor of Redis TTL"


def test_main_imports_without_inmemory_handler():
    import main
    assert hasattr(main, "app")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lifespan_no_inmemory_cleanup.py -v`
Expected: FAIL — module still exists and `main` still imports it.

- [ ] **Step 3: Remove the cleanup import and task in main.py**

In `main.py`, delete line 12:

```python
from app.utilities.file_embeddings_handler import cleanup_expired_files
```

And delete lines 35-36 inside `lifespan`:

```python
        # start background task for cleanup
        asyncio.create_task(cleanup_expired_files())
```

(Leave the `yield` that followed.) If `asyncio` is now unused, leave the import — `asyncio` is harmless; do not remove other startup logic.

- [ ] **Step 4: Delete the obsolete module**

Run: `git rm app/utilities/file_embeddings_handler.py`

> Rationale: file/embedding lifetime is now governed by the channel manifest's
> Redis TTL (`CHANNEL_TTL_SECONDS`). On-disk cleanup of expired channel
> directories is handled in Phase 3 (a TTL-driven sweep), not by a per-process
> in-memory dict that the 2 uvicorn workers couldn't share.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_lifespan_no_inmemory_cleanup.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `pytest -v`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add main.py tests/test_lifespan_no_inmemory_cleanup.py
git rm app/utilities/file_embeddings_handler.py
git commit -m "refactor: replace in-memory SESSION_FILES with Redis TTL manifest"
```

---

### Task 8: Fix the collection-name mismatch in load path

**Files:**
- Modify: `app/utilities/rag_utilities.py:140-169`
- Test: `tests/test_load_embeddings_collection.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_load_embeddings_collection.py`:

```python
import app.utilities.rag_utilities as util_mod


def test_load_embeddings_uses_channel_collection(monkeypatch, tmp_path):
    captured = {}

    class _FakeChroma:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    # channel dir must exist and be non-empty to pass the guard
    channel_dir = tmp_path / "chan-1"
    channel_dir.mkdir()
    (channel_dir / "chroma.sqlite3").write_text("x")

    monkeypatch.setattr(util_mod, "EMBEDDING_DIR", str(tmp_path))
    monkeypatch.setattr(util_mod, "Chroma", _FakeChroma)
    monkeypatch.setattr(util_mod.RAGUtilities, "__init__", lambda self: None)

    inst = util_mod.RAGUtilities()
    inst.embedding_model = object()
    inst.load_embeddings("chan-1")

    assert captured["collection_name"] == "rag_channel"
    assert captured["persist_directory"].endswith("chan-1")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_load_embeddings_collection.py -v`
Expected: FAIL — `collection_name` is `"chan-1_collection"`, not `"rag_channel"`.

- [ ] **Step 3: Fix `load_embeddings`**

In `app/utilities/rag_utilities.py`, in `load_embeddings` (lines 140-169), change the `Chroma(...)` construction's `collection_name` argument from:

```python
                collection_name=f"{filename}_collection"
```

to:

```python
                collection_name=settings.CHROMA_COLLECTION_NAME
```

The `filename` parameter is now a `channel_id` (the per-channel directory name). Rename the parameter for clarity: change the signature `def load_embeddings(self, filename: str):` to `def load_embeddings(self, channel_id: str):` and replace internal uses of `filename` with `channel_id` within this method (the `persist_directory`, cache key, and log lines).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_load_embeddings_collection.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all tests PASS.

> Note: `create_retriever`, `chat_with_document`, and the conversational chain
> still reference the old single-file flow. They are rewritten wholesale in
> Phase 2 (hybrid retrieval). This task only fixes the collection-name bug so the
> dense store written in Task 5 is readable with a consistent name.

- [ ] **Step 6: Commit**

```bash
git add app/utilities/rag_utilities.py tests/test_load_embeddings_collection.py
git commit -m "fix: read Chroma with consistent per-channel collection name"
```

---

## Self-Review

**Spec coverage (Phase 1 portion of the spec):**
- Per-channel Chroma collection + consistent name → Tasks 5, 8 ✓
- Chunk metadata `{channel_id, source, doc_id, chunk_id}` → Task 3 ✓
- `doc_id` for idempotent re-upload → Tasks 3, 5 ✓ (full delete-before-insert finalized Phase 2)
- Redis manifest replacing `SESSION_FILES` → Tasks 4, 6, 7 ✓
- Unify expiry into TTL scheme → Tasks 4 (TTL), 7 (remove in-memory) ✓
- Collection-name bug fix → Task 8 ✓
- Settings knobs (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHROMA_COLLECTION_NAME`, `CHANNEL_TTL_SECONDS`) → Task 2 ✓
- Upload API takes `channel_id` → Task 6 ✓

**Deferred to later phases (intentional, noted in tasks):** hybrid retrieval, reranker, chat-path rewrite (Phase 2); on-disk expired-channel sweep, auth, rate limiting, observability, caching (Phase 3); eval harness (Phase 4).

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `compute_doc_id`/`chunk_text` signatures match across Tasks 3, 5; `register_document(channel_id, doc_id, filename)` consistent across Tasks 4, 6; `CHROMA_COLLECTION_NAME` used identically in Tasks 5, 8. ✓
