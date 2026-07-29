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

The preferred embedding path expects an ONNX model at `app/models/bge-large-en-v1.5_ONNX` (1024-dim), which is **not** in the repo. Create it once with:

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
model = ORTModelForFeatureExtraction.from_pretrained("BAAI/bge-large-en-v1.5", export=True)
tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
model.save_pretrained("app/models/bge-large-en-v1.5_ONNX")
tok.save_pretrained("app/models/bge-large-en-v1.5_ONNX")
```

`OptimumEmbeddingWrapper` (`app/utilities/optimum_embeddings.py`) loads this directly via `optimum`+`transformers` — no llama_index dependency (removing it cut ~6s off process startup import time). If this model is missing or fails to load, the app automatically falls back to FastEmbed's `BAAI/bge-large-en-v1.5` on the CUDA execution provider — so a GPU + CUDA is effectively assumed. Changing the embedding model/dimension requires clearing existing per-channel Chroma stores under `data/database/` (dimension mismatch breaks old collections).

## Architecture

Layered FastAPI app under `app/`, wired together in `main.py`:

> **Production RAG (upgrade complete).** This codebase was upgraded from naive single-document RAG to a per-channel multi-document, hybrid-retrieval system across 4 phases. The design spec is at `docs/superpowers/specs/2026-05-29-production-rag-design.md` and phased plans under `docs/superpowers/plans/`. **All 4 phases are complete**: (1) per-channel storage, (2) hybrid retrieval + cross-encoder rerank, (3) production cross-cutting (auth, rate limiting, metrics, query cache, channel sweep), (4) evaluation harness. The notes below reflect the final state.

- **`routes/rag_routes.py`** — HTTP layer. Endpoints (all under prefix `/api/v1/rag-chatbot`): `GET /status` (open), `POST /upload` (multipart: `channel_id` form field + `file`), `POST /chat` (JSON `{channel_id, message, filename?}`), `GET /sentry-debug` (raises 404 in production). `/upload` and `/chat` carry `Depends(require_api_key)` + a slowapi rate-limit decorator and take a `request: Request` param (slowapi requires it). Standardized JSON envelope (`success`/`message`/`data`/`error`) via `create_error_response`. On upload it registers the doc in the Redis channel manifest.
- **`controller/rag_controller.py`** — orchestration. `create_document_embeddings(channel_id, file_path)` chunks text and **dual-writes**: upserts into the channel's Chroma collection AND appends to the channel BM25 index. `chat_with_document` is the explicit hybrid flow: validate → `load_embeddings(channel_id)` (404 if none) → load Redis history → `contextualize_question` (LLM rewrite) → `HybridRetriever.retrieve` → if no docs, return a grounded fallback **without** calling the LLM → else `answer` (LLM) → append+save history → envelope.
- **`services/rag_service.py`** — pure text extraction from TXT/PDF (PyMuPDF/`fitz`) / DOCX (`python-docx`). Stateless static methods. No longer used by the `/upload` path (superseded by `app.ingestion.parser.parse_document`, Phase 5) — still present/unremoved, may still be referenced elsewhere.
- **`retrieval/`** — the retrieval engine. `chunking.py` (`chunk_text` → `Document`s tagged `{channel_id, source, doc_id, chunk_id}`, stable `compute_doc_id`); `bm25_index.py` (per-channel BM25 corpus persisted at `data/database/<channel_id>/bm25.pkl`; `add_documents` / `search`); `fusion.py` (`reciprocal_rank_fusion`, keyed on `chunk_id`); `reranker.py` (`CrossEncoderReranker`, lazy/class-cached `bge-reranker-base`); `hybrid_retriever.py` (`HybridRetriever(channel_id, vectorstore).retrieve(query, filename=None)` → dense+BM25 → RRF → rerank, dense errors degrade to sparse-only).
- **`repository/channel_repository.py`** — Redis-backed manifest of which documents belong to a channel (`channel:{id}:docs` hash, TTL `CHANNEL_TTL_SECONDS`). `register_document` / `list_documents` / `remove_channel`. Worker-shared and restart-survivable.
- **`utilities/rag_utilities.py`** — manages the embedding model + Groq LLM (class-cached), loads/caches per-channel Chroma stores (`load_embeddings(channel_id)`), and holds the chat helpers `contextualize_question(message, history)` and `answer(user_input, context, history, filename)` plus the prompt builders (`_contextualize_prompt`, `create_qa_prompt`).
- **`utilities/optimum_embeddings.py`** — `OptimumEmbeddingWrapper` (local ONNX) and `FastEmbedWrapper` (fallback), both adapting their backend to LangChain's `embed_documents`/`embed_query` interface so Chroma can use them.
- **`database/redis.py`** — chat-history persistence. Serializes history with `dill` (`save`/`load`/`delete_session_to_redis`), 20-minute TTL. Exposes module-level `redis_client` (may be `None` if Redis is unreachable; callers guard for this).
- **`middleware/`** — `auth.py` (`require_api_key` FastAPI dependency: enforces `X-API-Key` against `settings.api_keys_list()`; **disabled when `API_KEYS` is empty** — dev default); `rate_limit.py` (slowapi `limiter` keyed by API-key-or-IP, plus a `rate_limit_handler` returning a 429 in the standard envelope). Wired in `main.py` via `app.state.limiter` + exception handler.
- **`observability/metrics.py`** — `instrument(app, enabled)` mounts Prometheus `/metrics` via `prometheus-fastapi-instrumentator` (gated by `METRICS_ENABLED`). `/metrics` is unauthenticated — network-restrict it at the edge in prod.
- **`repository/query_cache.py`** — optional Redis cache of chat answers (`make_key`/`get_cached`/`set_cached`), gated by `ENABLE_QUERY_CACHE` (default off). Only grounded answers are cached (never the no-docs fallback). **`repository/channel_sweeper.py`** — `sweep_once()` / `sweep_loop()` background task (scheduled in the lifespan) deletes on-disk channel dirs with no live Redis manifest whose mtime is older than `CHANNEL_TTL_SECONDS`.
- **`config/settings.py`** — pydantic-settings `Settings` singleton (`settings`), all paths relative to `BASE_DIR`. Knobs: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHROMA_COLLECTION_NAME`, `CHANNEL_TTL_SECONDS` (P1); `DENSE_TOP_K`, `BM25_TOP_K`, `RRF_K`, `RERANK_TOP_N`, `RERANKER_MODEL` (P2); `API_KEYS`, `RATE_LIMIT_CHAT`, `RATE_LIMIT_UPLOAD`, `ENABLE_QUERY_CACHE`, `QUERY_CACHE_TTL`, `METRICS_ENABLED` + `api_keys_list()` (P3). **`config/logger.py`** — preconfigured loguru `logger`; import this, not loguru directly.

### Key design points and gotchas

- **Per-channel hybrid retrieval.** Each `channel_id` gets one Chroma collection at `data/database/<channel_id>/` (consistent `collection_name = "rag_channel"`) plus a co-located `bm25.pkl`. Ingestion writes the **same** `chunk_id`-tagged docs to both, so dense and sparse results fuse/dedupe correctly via RRF on `chunk_id`. Retrieval: dense top-`DENSE_TOP_K` + BM25 top-`BM25_TOP_K` → RRF (`RRF_K`) → cross-encoder rerank to top-`RERANK_TOP_N`.
- **Grounding guard.** If hybrid retrieval returns no docs, chat returns a canned "couldn't find relevant information" message and does **not** call the LLM — avoids ungrounded answers.
- **Channel document manifest in Redis.** `repository/channel_repository.py` is the source of truth for channel→docs, replacing the former in-memory `SESSION_FILES` dict. Expiry rides the manifest's Redis TTL; on-disk cleanup of expired channel dirs is deferred to Phase 3.
- **Reranker cost.** `bge-reranker-base` is downloaded on first live chat (no warmup yet) and `bm25_index.search` rebuilds the BM25 index from the persisted corpus per query — both acceptable at current scale; caching/warmup is Phase 3.
- **Caching is process-global and stateful.** `RAGUtilities` caches the LLM and embedding model at the class level; `VECTOR_STORE_CACHE` and `CrossEncoderReranker._model` are process-global. Not shared across workers, not restart-surviving.
- **Re-upload duplicates chunks** for now (both Chroma and BM25 accumulate) — stable `doc_id` exists but delete-before-insert is still deferred.
- **Singleton via repeated instantiation.** Code calls `RAGUtilities()` / `RAGController()` freely — class-level caches make this cheap, but constructors still run.
- **Sentry** is initialized in `main.py` with a hardcoded DSN; `GET /sentry-debug` deliberately raises a `ZeroDivisionError` to test it.

- **`eval/`** — offline evaluation harness (Phase 4). `retrieval_metrics.py` (pure `hit_at_k`/`reciprocal_rank`/`summarize`); `golden_set.py` (generate/save/load a synthetic Q&A golden set via the LLM); `run_eval.py` (`compare_pipelines` naive dense-only vs hybrid, `format_report`/`write_report`, and `maybe_ragas_scores` which lazily imports RAGAS). **RAGAS deps are isolated in `requirements-eval.txt`, NOT the serving `requirements.txt`** (they would upgrade shared `langchain-core`/`dill`); install them in a separate venv to run answer-quality scoring. The serving test suite runs without RAGAS.
- **`ingestion/`** — document parsing and web-crawl ingestion (Phase 5). `parser.py` (`parse_document(source, source_type)` via Docling — PDF/DOCX/HTML all go through one layout-aware pipeline: paragraphs, markdown tables, and figures; figures are captioned inline via `captioning.py`'s Groq vision call and folded into `ParsedDocument.to_text_stream()`); `captioning.py` (`caption_figure`, vision-capable Groq model per `settings.GROQ_VISION_MODEL`, never raises — returns `""` on failure); `crawler.py` (`crawl_site`, Scrapy same-domain crawl bounded by `CRAWL_MAX_PAGES`/`CRAWL_MAX_DEPTH`, optional `include_paths` prefix filter, runs in a subprocess since Scrapy's Twisted reactor can't share a process with uvicorn's asyncio loop); `pipeline.py` (`run_crawl_job`, the full crawl→parse→caption→chunk→embed flow, writing into the *same* Chroma collection + BM25 index as file uploads). Crawl job progress is tracked in Redis via `repository/crawl_jobs.py` (`create_job`/`update_job`/`get_job`), polled via `GET /crawl/{job_id}`.
- File uploads (`POST /upload`) now parse PDF/DOCX via Docling (`app.ingestion.parser.parse_document`) instead of PyMuPDF/python-docx directly — tables and figures are preserved, not just plain text.
- **Vector store is still Chroma in this phase.** Qdrant migration and porting this ingestion logic to `closeloop-backend` are explicitly deferred to future work (see `docs/superpowers/specs/2026-07-29-crawler-docling-ingestion-design.md`).

### Testing

`pytest` (+ `fakeredis`) is set up; run the suite with `.venv\Scripts\python.exe -m pytest` (23 tests). Tests live in `tests/` and follow TDD per the phase plans. The reranker and LLM are always mocked in tests — no model downloads or network calls. Note: the project venv is uv-managed — if `python` errors with a missing interpreter path, run `uv python install 3.11.11` to restore it.

**Do not use Starlette's `TestClient` for route tests in this repo** — it hangs indefinitely here (confirmed via `faulthandler` stack dump: its thread-based blocking portal deadlocks against Sentry's global threading instrumentation, which patches `Thread.run`). Use `httpx.AsyncClient` with `httpx.ASGITransport(app=app)` directly instead (async test, `pytest.mark.anyio`) — see `tests/test_crawl_route.py` for the pattern.

## Conventions

- All endpoint responses follow the `{success, message, data, error}` envelope. Preserve it when adding endpoints.
- Errors are logged via the loguru `logger` and swallowed into the response envelope rather than propagated — controller methods catch broadly and return structured error dicts.
- File-format support: extraction handles TXT/PDF/DOCX, but the upload endpoint only accepts PDF/DOCX.
