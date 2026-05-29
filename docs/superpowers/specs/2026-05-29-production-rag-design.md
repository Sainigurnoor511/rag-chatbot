# Production-Grade RAG Upgrade — Design

**Date:** 2026-05-29
**Status:** Approved (design); pending spec review
**Author:** Claude Code + Gurnoor Singh Saini

## Goal

Upgrade the current naive RAG service (single-document dense-only retrieval over
Chroma + Groq) into a production-grade, per-channel multi-document RAG system with
hybrid retrieval, cross-encoder reranking, evaluation, and production serving
hardening — without a full rewrite.

## Decisions (locked)

| Area | Decision |
|------|----------|
| Vector DB | **Keep Chroma** (dense vectors) |
| Sparse retrieval | **Local BM25** (`rank-bm25`), per channel |
| Fusion | **Reciprocal Rank Fusion (RRF)** |
| Reranking | **Local cross-encoder** `BAAI/bge-reranker-base` (GPU, class-cached) |
| Retrieval scope | **Per-channel multi-document** |
| Eval | **RAGAS + synthetic golden set** (Groq as judge) |
| Cross-cutting | Auth (API key), rate limiting, observability, caching, config knobs, bug fixes |
| Approach | **In-place enhancement + thin new packages** (lowest blast radius) |

## Architecture

Retain existing layering (`routes → controller → services → utilities`) and add:

```
app/
  retrieval/
    chunking.py          # config-driven RecursiveCharacterTextSplitter + metadata
    bm25_index.py        # build / persist / load per-channel BM25 (rank-bm25)
    hybrid_retriever.py  # HybridRerankRetriever(BaseRetriever): dense + BM25 -> RRF -> rerank
    reranker.py          # CrossEncoderReranker (bge-reranker-base, lazy + class-cached)
  repository/
    channel_repository.py # Redis-backed channel -> docs manifest (worker-shared)
  eval/
    generate_golden_set.py
    run_eval.py
  middleware/
    auth.py              # X-API-Key dependency
    rate_limit.py        # slowapi limiter config
```

### Storage model (per channel)

- **Dense:** one Chroma collection per channel at `data/database/<channel_id>/`,
  with a single consistent `collection_name` (**fixes the existing
  write `"rag"` vs read `f"{filename}_collection"` mismatch bug**).
  Every chunk carries metadata `{channel_id, source, doc_id, chunk_id}`.
- **Sparse:** per-channel BM25 index persisted at
  `data/database/<channel_id>/bm25.pkl`, rebuilt on each new upload to the channel.
- **Manifest:** Redis key `channel:{id}:docs` listing the channel's documents.
  **Replaces the in-memory `SESSION_FILES` dict** so state is shared across the 2
  uvicorn workers and survives restarts. **Unifies** the two competing expiry
  mechanisms (20-min Redis TTL, 30-min in-memory cleanup) into one TTL-driven scheme.

## Ingestion pipeline — `POST /upload`

API change: accepts `file` **and** a `channel_id` form field.

1. Validate (PDF/DOCX, ≤50MB).
2. Extract text via `RAGService` (unchanged).
3. Chunk (`chunking.py`): tuned separators `["\n\n", "\n", ". ", " "]`,
   config-driven `chunk_size`/`overlap`; tag metadata `{channel_id, source, doc_id, chunk_id}`.
   `doc_id = hash(filename)` so re-uploading the same file **replaces** rather than duplicates.
4. Upsert chunks into the channel's Chroma collection.
5. Rebuild the channel BM25 index from all channel chunks; persist `bm25.pkl`;
   invalidate in-process cache.
6. Update Redis channel manifest + TTL.

## Retrieval pipeline — `POST /chat`

Request becomes `{channel_id, message}`; `filename` optional (metadata filter to
restrict to one doc within the channel).

1. Load chat history from Redis (unchanged).
2. **Query contextualization** — keep existing history-aware rewrite prompt.
3. **Hybrid retrieval** over the channel:
   - Dense: Chroma similarity search, top `DENSE_TOP_K` (default 20).
   - Sparse: BM25, top `BM25_TOP_K` (default 20).
   - **RRF fusion**: `score = Σ 1/(RRF_K + rank)`, default `RRF_K=60` → unique candidates.
4. **Cross-encoder rerank**: score `(query, chunk)` with `bge-reranker-base`,
   keep top `RERANK_TOP_N` (default 5).
5. Assemble reranked context → existing QA prompt → Groq → answer; save history.

**Removes current redundancy:** today the controller does a manual
`similarity_search(k=2)` *and* a separate `create_retrieval_chain` retrieval.
Both are replaced by one `HybridRerankRetriever` (LangChain `BaseRetriever`)
plugged into a single chain — one retrieval path.

## Production cross-cutting

- **Auth:** `X-API-Key` header dependency; valid keys from `API_KEYS` env
  (comma-separated). Applied to `/upload` and `/chat`; `/status` open.
  `/sentry-debug` gated behind `ENVIRONMENT != "production"`.
- **Rate limiting:** `slowapi` (per-key/per-IP), config-driven
  (defaults: chat 30/min, upload 10/min).
- **Observability:** `prometheus-fastapi-instrumentator` exposing `/metrics`;
  structured per-stage retrieval traces via loguru (dense/BM25 counts, fused
  candidate count, rerank scores, latency per stage); keep Sentry.
- **Caching:** in-process LRU for loaded Chroma + BM25 per channel; reranker &
  embedding models class-cached; optional Redis query-result cache (config flag, short TTL).

## Configuration (new `Settings` knobs)

`CHUNK_SIZE`, `CHUNK_OVERLAP`, `DENSE_TOP_K`, `BM25_TOP_K`, `RRF_K`,
`RERANK_TOP_N`, `RERANKER_MODEL`, rate-limit values, `API_KEYS`, cache flags.

## Evaluation (`app/eval/`)

- `generate_golden_set.py` — Groq generates Q&A pairs from a channel's docs → JSON golden set.
- `run_eval.py` — RAGAS metrics (faithfulness, answer_relevancy, context_precision,
  context_recall); compares **naive vs hybrid+rerank** → report.

## Testing

Add `pytest`. Unit tests for: RRF fusion, BM25 build/query, chunking + metadata,
channel repository (Redis mocked), reranker ordering (model mocked). Eval harness
is run separately from unit tests.

## New dependencies

`rank-bm25`, `sentence-transformers`, `ragas`, `datasets`, `slowapi`,
`prometheus-fastapi-instrumentator`, `pytest`.

## Bugs fixed as part of this work

1. Chroma `collection_name` write/read mismatch (silent empty retrieval).
2. Redundant double retrieval in chat.
3. `SESSION_FILES` per-process state not shared across workers → Redis manifest.
4. Two independent session-expiry mechanisms → unified TTL scheme.

## Out of scope (YAGNI)

- Switching vector DBs (Pinecone/Qdrant/pgvector) — Chroma retained.
- Streaming responses, multi-tenant billing, UI.
- Distributed/GPU-pooled reranking — single-process GPU is sufficient.

## Suggested implementation phases

1. Per-channel storage refactor + Redis manifest + bug fixes (foundation).
2. Hybrid retrieval (BM25 + RRF) + cross-encoder rerank + chain cleanup.
3. Production cross-cutting (auth, rate limiting, observability, caching, config).
4. Eval harness (RAGAS + golden set) + unit tests.
