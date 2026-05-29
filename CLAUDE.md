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

- **`routes/rag_routes.py`** — HTTP layer. Endpoints (all under prefix `/api/v1/rag-chatbot`): `GET /status`, `POST /upload` (multipart file), `POST /chat` (JSON `{channel_id, message, filename}`), `GET /sentry-debug`. Validates uploads (PDF/DOCX only, ≤50MB), defines the standardized JSON envelope (`success`/`message`/`data`/`error`) via `create_error_response`.
- **`controller/rag_controller.py`** — orchestration. `create_document_embeddings` chunks text and builds the Chroma store; `chat_with_document` loads Redis history, builds the retrieval chain, runs a manual similarity search to attach context, invokes the chain, and saves history back.
- **`services/rag_service.py`** — pure text extraction from TXT/PDF (PyMuPDF/`fitz`) / DOCX (`python-docx`). Stateless static methods.
- **`utilities/rag_utilities.py`** — the RAG engine. `RAGUtilities` builds the LangChain history-aware retrieval chain (contextualize-question prompt + stuff-documents QA chain), manages the embedding model and LLM, and loads/caches vector stores. Holds the QA system prompts (answers are constrained strictly to the uploaded document).
- **`utilities/optimum_embeddings.py`** — `OptimumEmbeddingWrapper` (local ONNX) and `FastEmbedWrapper` (fallback), both adapting their backend to LangChain's `embed_documents`/`embed_query` interface so Chroma can use them.
- **`utilities/file_embeddings_handler.py`** — in-memory `SESSION_FILES` registry + `cleanup_expired_files` background task (launched in the lifespan) that deletes uploads and embedding folders 30 minutes after registration.
- **`database/redis.py`** — session persistence. Serializes chat history with `dill` (`save`/`load`/`delete_session_to_redis`), 20-minute TTL.
- **`config/settings.py`** — pydantic-settings `Settings` singleton (`settings`), all paths resolved relative to `BASE_DIR`. **`config/logger.py`** — preconfigured loguru `logger` (console + rotating `logs/app.log`); import this, not loguru directly.

### Key design points and gotchas

- **Per-document vector stores.** Each uploaded file gets its own Chroma persist directory at `data/database/<filename>/`. The `filename` field in a chat request must match the uploaded filename for retrieval to find the store. Uploads live in `data/uploads/`.
- **Collection-name mismatch (latent bug).** `create_document_embeddings` writes Chroma with `collection_name="rag"`, but `load_embeddings` reads with `collection_name=f"{filename}_collection"`. Be aware of this if retrieval returns empty results.
- **Caching is process-global and stateful.** `RAGUtilities` caches the LLM and embedding model at the class level (`_llm_instance`, `_model_instance`); `VECTOR_STORE_CACHE` and `SESSION_HISTORY` are module-level dicts. State does not survive restarts and is not shared across the 2 uvicorn workers — each worker has its own caches and its own `SESSION_FILES`.
- **Two parallel session-expiry mechanisms:** Redis TTL (20 min, for chat history) and the in-memory cleanup task (30 min, for files/embeddings on disk). They are independent.
- **Singleton via repeated instantiation.** Code calls `RAGUtilities()` / `RAGController()` freely — the class-level caches make this cheap, but constructors still run.
- **Sentry** is initialized in `main.py` with a hardcoded DSN; `GET /sentry-debug` deliberately raises a `ZeroDivisionError` to test it.

## Conventions

- All endpoint responses follow the `{success, message, data, error}` envelope. Preserve it when adding endpoints.
- Errors are logged via the loguru `logger` and swallowed into the response envelope rather than propagated — controller methods catch broadly and return structured error dicts.
- File-format support: extraction handles TXT/PDF/DOCX, but the upload endpoint only accepts PDF/DOCX.
