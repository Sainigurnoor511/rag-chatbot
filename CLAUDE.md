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

> **Production upgrade in progress.** This codebase is being upgraded from naive single-document RAG to a per-channel multi-document, hybrid-retrieval system. The design spec is at `docs/superpowers/specs/2026-05-29-production-rag-design.md` and phased plans under `docs/superpowers/plans/`. **Phases 1 (per-channel storage) and 2 (hybrid retrieval + cross-encoder rerank) are complete**; Phases 3–4 (production cross-cutting: auth/rate-limiting/observability/caching; eval harness) are not yet built. The notes below reflect the post-Phase-2 state.

- **`routes/rag_routes.py`** — HTTP layer. Endpoints (all under prefix `/api/v1/rag-chatbot`): `GET /status`, `POST /upload` (multipart: `channel_id` form field + `file`), `POST /chat` (JSON `{channel_id, message, filename?}` — `filename` optional, restricts retrieval to one doc in the channel), `GET /sentry-debug`. Validates uploads (PDF/DOCX only, ≤50MB), defines the standardized JSON envelope (`success`/`message`/`data`/`error`) via `create_error_response`. On upload it registers the doc in the Redis channel manifest.
- **`controller/rag_controller.py`** — orchestration. `create_document_embeddings(channel_id, file_path)` chunks text and **dual-writes**: upserts into the channel's Chroma collection AND appends to the channel BM25 index. `chat_with_document` is the explicit hybrid flow: validate → `load_embeddings(channel_id)` (404 if none) → load Redis history → `contextualize_question` (LLM rewrite) → `HybridRetriever.retrieve` → if no docs, return a grounded fallback **without** calling the LLM → else `answer` (LLM) → append+save history → envelope.
- **`services/rag_service.py`** — pure text extraction from TXT/PDF (PyMuPDF/`fitz`) / DOCX (`python-docx`). Stateless static methods.
- **`retrieval/`** — the retrieval engine. `chunking.py` (`chunk_text` → `Document`s tagged `{channel_id, source, doc_id, chunk_id}`, stable `compute_doc_id`); `bm25_index.py` (per-channel BM25 corpus persisted at `data/database/<channel_id>/bm25.pkl`; `add_documents` / `search`); `fusion.py` (`reciprocal_rank_fusion`, keyed on `chunk_id`); `reranker.py` (`CrossEncoderReranker`, lazy/class-cached `bge-reranker-base`); `hybrid_retriever.py` (`HybridRetriever(channel_id, vectorstore).retrieve(query, filename=None)` → dense+BM25 → RRF → rerank, dense errors degrade to sparse-only).
- **`repository/channel_repository.py`** — Redis-backed manifest of which documents belong to a channel (`channel:{id}:docs` hash, TTL `CHANNEL_TTL_SECONDS`). `register_document` / `list_documents` / `remove_channel`. Worker-shared and restart-survivable.
- **`utilities/rag_utilities.py`** — manages the embedding model + Groq LLM (class-cached), loads/caches per-channel Chroma stores (`load_embeddings(channel_id)`), and holds the chat helpers `contextualize_question(message, history)` and `answer(user_input, context, history, filename)` plus the prompt builders. (The older `create_retriever`/`create_conversational_chain_history` chain methods are now unused — slated for removal.)
- **`utilities/optimum_embeddings.py`** — `OptimumEmbeddingWrapper` (local ONNX) and `FastEmbedWrapper` (fallback), both adapting their backend to LangChain's `embed_documents`/`embed_query` interface so Chroma can use them.
- **`database/redis.py`** — chat-history persistence. Serializes history with `dill` (`save`/`load`/`delete_session_to_redis`), 20-minute TTL. Exposes module-level `redis_client` (may be `None` if Redis is unreachable; callers guard for this).
- **`config/settings.py`** — pydantic-settings `Settings` singleton (`settings`), all paths relative to `BASE_DIR`. Knobs: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHROMA_COLLECTION_NAME`, `CHANNEL_TTL_SECONDS` (Phase 1); `DENSE_TOP_K`, `BM25_TOP_K`, `RRF_K`, `RERANK_TOP_N`, `RERANKER_MODEL` (Phase 2). **`config/logger.py`** — preconfigured loguru `logger`; import this, not loguru directly.

### Key design points and gotchas

- **Per-channel hybrid retrieval.** Each `channel_id` gets one Chroma collection at `data/database/<channel_id>/` (consistent `collection_name = "rag_channel"`) plus a co-located `bm25.pkl`. Ingestion writes the **same** `chunk_id`-tagged docs to both, so dense and sparse results fuse/dedupe correctly via RRF on `chunk_id`. Retrieval: dense top-`DENSE_TOP_K` + BM25 top-`BM25_TOP_K` → RRF (`RRF_K`) → cross-encoder rerank to top-`RERANK_TOP_N`.
- **Grounding guard.** If hybrid retrieval returns no docs, chat returns a canned "couldn't find relevant information" message and does **not** call the LLM — avoids ungrounded answers.
- **Channel document manifest in Redis.** `repository/channel_repository.py` is the source of truth for channel→docs, replacing the former in-memory `SESSION_FILES` dict. Expiry rides the manifest's Redis TTL; on-disk cleanup of expired channel dirs is deferred to Phase 3.
- **Reranker cost.** `bge-reranker-base` is downloaded on first live chat (no warmup yet) and `bm25_index.search` rebuilds the BM25 index from the persisted corpus per query — both acceptable at current scale; caching/warmup is Phase 3.
- **Caching is process-global and stateful.** `RAGUtilities` caches the LLM and embedding model at the class level; `VECTOR_STORE_CACHE` and `CrossEncoderReranker._model` are process-global. Not shared across workers, not restart-surviving.
- **Re-upload duplicates chunks** for now (both Chroma and BM25 accumulate) — stable `doc_id` exists but delete-before-insert is still deferred.
- **Singleton via repeated instantiation.** Code calls `RAGUtilities()` / `RAGController()` freely — class-level caches make this cheap, but constructors still run.
- **Sentry** is initialized in `main.py` with a hardcoded DSN; `GET /sentry-debug` deliberately raises a `ZeroDivisionError` to test it.

### Testing

`pytest` (+ `fakeredis`) is set up; run the suite with `.venv\Scripts\python.exe -m pytest`. Tests live in `tests/` and follow TDD per the phase plans. Note: the project venv is uv-managed — if `python` errors with a missing interpreter path, run `uv python install 3.11.11` to restore it.

## Conventions

- All endpoint responses follow the `{success, message, data, error}` envelope. Preserve it when adding endpoints.
- Errors are logged via the loguru `logger` and swallowed into the response envelope rather than propagated — controller methods catch broadly and return structured error dicts.
- File-format support: extraction handles TXT/PDF/DOCX, but the upload endpoint only accepts PDF/DOCX.
