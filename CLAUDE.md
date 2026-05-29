# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A document-grounded RAG chatbot exposed as a FastAPI service. Users upload a PDF/DOCX, the service embeds it into a per-document Chroma vector store, and chat requests run a history-aware retrieval chain over that store using a Groq-hosted LLM. Conversation history is persisted in Redis keyed by `channel_id`.

## Commands

```powershell
# Run the API (serves on 0.0.0.0:8000, docs at /docs, /redocs)
python main.py

# Install dependencies (GPU stack — see README.md for CUDA 12.1 + PyTorch setup first)
pip install -r requirements.txt
```

There is no test runner, linter, or build step. The `test/` directory holds exploratory Jupyter notebooks (CUDA checks, ONNX export, local-model experiments), not an automated test suite.

### Required environment (`.env`)

`GROQ_API_KEY` and `GROQ_MODEL` are **required** — the app fails to start without them (`Settings` has no defaults for these). Redis defaults to `localhost:6379`; the app degrades gracefully (logs a warning, skips session persistence) if Redis is unreachable.

### Generating the local ONNX embedding model

The preferred embedding path expects an ONNX model at `app/models/bge-base-en-v1.5_ONNX`, which is **not** in the repo. Create it once with:

```python
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
OptimumEmbedding.create_and_save_optimum_model("BAAI/bge-base-en-v1.5", "app/models/bge-base-en-v1.5_ONNX")
```

(See `test/local_model.ipynb`.) If this model is missing or fails to load, the app automatically falls back to FastEmbed's `BAAI/bge-base-en-v1.5` on the CUDA execution provider — so a GPU + CUDA is effectively assumed.

## Architecture

Layered FastAPI app under `app/`, wired together in `main.py`:

> **Production upgrade in progress.** This codebase is being upgraded from naive single-document RAG to a per-channel multi-document, hybrid-retrieval system. The design spec is at `docs/superpowers/specs/2026-05-29-production-rag-design.md` and phased plans under `docs/superpowers/plans/`. **Phase 1 (per-channel storage foundation) is complete**; Phases 2–4 (hybrid BM25+dense retrieval & cross-encoder rerank; production cross-cutting; eval harness) are not yet built. The notes below reflect the post-Phase-1 state.

- **`routes/rag_routes.py`** — HTTP layer. Endpoints (all under prefix `/api/v1/rag-chatbot`): `GET /status`, `POST /upload` (multipart: `channel_id` form field + `file`), `POST /chat` (JSON `{channel_id, message, filename}`), `GET /sentry-debug`. Validates uploads (PDF/DOCX only, ≤50MB), defines the standardized JSON envelope (`success`/`message`/`data`/`error`) via `create_error_response`. On upload it registers the doc in the Redis channel manifest.
- **`controller/rag_controller.py`** — orchestration. `create_document_embeddings(channel_id, file_path)` chunks text (via `retrieval/chunking.py`) and upserts it into the channel's Chroma collection. `chat_with_document` is **still the old single-file flow** (loads Redis history, manual similarity search, builds the chain) — it is rewritten in Phase 2, so end-to-end chat is not yet consistent with the per-channel model.
- **`services/rag_service.py`** — pure text extraction from TXT/PDF (PyMuPDF/`fitz`) / DOCX (`python-docx`). Stateless static methods.
- **`retrieval/chunking.py`** — `chunk_text()` splits text into LangChain `Document`s tagged with `{channel_id, source, doc_id, chunk_id}`; `compute_doc_id()` gives a stable filename-derived id so re-uploads can replace (idempotent replace itself lands in Phase 2). (`retrieval/` is where Phase 2's hybrid retriever, BM25 index, and reranker will live.)
- **`repository/channel_repository.py`** — Redis-backed manifest of which documents belong to a channel (`channel:{id}:docs` hash, TTL `CHANNEL_TTL_SECONDS`). `register_document` / `list_documents` / `remove_channel`. Worker-shared and restart-survivable.
- **`utilities/rag_utilities.py`** — the RAG engine. `RAGUtilities` builds the LangChain history-aware retrieval chain (contextualize-question prompt + stuff-documents QA chain), manages the embedding model and LLM, and loads/caches per-channel vector stores (`load_embeddings(channel_id)`, `create_retriever(channel_id)`). Holds the QA system prompts.
- **`utilities/optimum_embeddings.py`** — `OptimumEmbeddingWrapper` (local ONNX) and `FastEmbedWrapper` (fallback), both adapting their backend to LangChain's `embed_documents`/`embed_query` interface so Chroma can use them.
- **`database/redis.py`** — chat-history persistence. Serializes history with `dill` (`save`/`load`/`delete_session_to_redis`), 20-minute TTL. Exposes module-level `redis_client` (may be `None` if Redis is unreachable; callers guard for this).
- **`config/settings.py`** — pydantic-settings `Settings` singleton (`settings`), all paths relative to `BASE_DIR`. Phase-1 knobs: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHROMA_COLLECTION_NAME`, `CHANNEL_TTL_SECONDS`. **`config/logger.py`** — preconfigured loguru `logger`; import this, not loguru directly.

### Key design points and gotchas

- **Per-channel vector stores.** Each `channel_id` gets one Chroma collection at `data/database/<channel_id>/` (uploads land in `data/uploads/`) holding chunks from all its documents, written and read with the single consistent `collection_name = settings.CHROMA_COLLECTION_NAME` (`"rag_channel"`). The earlier write/read collection-name mismatch bug is **fixed**. Note: `chat_with_document` still passes the request's `filename` where a `channel_id` is expected — a Phase-2 leftover to reconcile.
- **Channel document manifest in Redis.** `repository/channel_repository.py` is the source of truth for channel→docs, replacing the former in-memory `SESSION_FILES` dict (which wasn't shared across the 2 uvicorn workers). Expiry rides the manifest's Redis TTL; on-disk cleanup of expired channel dirs is deferred to Phase 3.
- **Caching is process-global and stateful.** `RAGUtilities` caches the LLM and embedding model at the class level (`_llm_instance`, `_model_instance`); `VECTOR_STORE_CACHE` and `SESSION_HISTORY` are module-level dicts. Not shared across workers and not restart-surviving.
- **Re-upload duplicates chunks** for now — stable `doc_id` exists but delete-before-insert is a Phase-2 item.
- **Singleton via repeated instantiation.** Code calls `RAGUtilities()` / `RAGController()` freely — class-level caches make this cheap, but constructors still run.
- **Sentry** is initialized in `main.py` with a hardcoded DSN; `GET /sentry-debug` deliberately raises a `ZeroDivisionError` to test it.

### Testing

`pytest` (+ `fakeredis`) is set up; run the suite with `.venv\Scripts\python.exe -m pytest`. Tests live in `tests/` and follow TDD per the phase plans. Note: the project venv is uv-managed — if `python` errors with a missing interpreter path, run `uv python install 3.11.11` to restore it.

## Conventions

- All endpoint responses follow the `{success, message, data, error}` envelope. Preserve it when adding endpoints.
- Errors are logged via the loguru `logger` and swallowed into the response envelope rather than propagated — controller methods catch broadly and return structured error dicts.
- File-format support: extraction handles TXT/PDF/DOCX, but the upload endpoint only accepts PDF/DOCX.
