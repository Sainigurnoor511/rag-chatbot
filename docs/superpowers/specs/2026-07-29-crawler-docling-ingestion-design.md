# Design: Web-Crawl Ingestion + Docling Document Parsing

Date: 2026-07-29

## Context

Current ingestion only accepts a single uploaded PDF/DOCX per call, extracted via PyMuPDF/python-docx (plain text only — no tables, no OCR, no diagrams). Multi-document-per-channel already works (verified: each upload appends to the channel's Chroma collection + BM25 index via `channel_repository`; `filename` param optionally scopes retrieval to one doc).

This design adds two things:
1. **Docling-based parsing** — replaces PyMuPDF/python-docx text extraction with layout-aware parsing (tables, structure, OCR for scanned content) for both uploaded files and scraped web pages.
2. **URL-crawl ingestion** — given a base URL + path filters, crawl same-domain pages (via Scrapy), scrape+parse them, and feed them into the same embedding pipeline as file uploads.

**Vector store stays Chroma.** Qdrant migration is explicitly deferred to a future scope (per user direction: build crawling/parsing first, port to closeloop-backend next, Qdrant swap comes after that).

## Non-goals

- No Qdrant migration in this phase.
- No porting to closeloop-backend in this phase (separate future project).
- No Sarvam OCR integration (user decided: Docling only).
- No changes to multi-doc-per-channel behavior (already works).

## Architecture

New `app/ingestion/` package, consumed by `rag_controller.py` alongside the existing file-upload path. Both file uploads and crawl jobs converge on the same chunk → embed → dual-write (Chroma + BM25) pipeline already in `RAGController.create_document_embeddings`.

```
                    ┌─────────────────┐
   file upload ────▶│                 │
                    │  app/ingestion/  │──▶ chunk_text ──▶ Chroma.from_documents
   crawl job   ────▶│  parser.py       │                └─▶ bm25_index.add_documents
                    │  crawler.py      │
                    │  captioning.py   │
                    └─────────────────┘
```

### `app/ingestion/parser.py`

Replaces `RAGService.get_text` (PyMuPDF/python-docx) as the extraction step for PDF, DOCX, and HTML input. Uses Docling (`docling.document_converter.DocumentConverter`) for all three formats — one parsing pipeline regardless of source type.

- `parse_document(file_path_or_html: str, source_type: Literal["pdf","docx","html"]) -> ParsedDocument`
- `ParsedDocument` holds: `text_blocks: list[str]` (paragraphs/sections in reading order), `tables: list[str]` (each table serialized to markdown so it chunks/embeds as readable text), `figures: list[ExtractedFigure]` (image bytes + page/position metadata, for captioning).
- Docling's OCR path (built-in, e.g. EasyOCR/Tesseract backend) handles scanned pages automatically — no separate OCR wiring needed.
- Output is flattened into a single ordered text stream (paragraphs + inline markdown tables + inline figure captions once captioned) before handing off to the existing `chunk_text`.

### `app/ingestion/captioning.py`

- `caption_figure(image_bytes: bytes) -> str` — calls Groq's vision-capable model via a new `settings.GROQ_VISION_MODEL` setting, using the same `ChatGroq`-style client already in `rag_utilities.py` (separate instance/config, not reusing the chat `GROQ_MODEL`).
- Captions are inserted into the parsed document's text stream at the figure's original position (e.g. `[Figure: <caption text>]`), so retrieval can match on diagram content like any other text.
- Failures (vision call errors, rate limits) are caught and logged; the figure is skipped (captioned as empty) rather than failing the whole document — consistent with the codebase's existing pattern of broad catch + structured error swallowing at the controller boundary, but scoped to a single figure so one bad image doesn't kill a multi-page document ingest.

### `app/ingestion/crawler.py`

- `crawl_site(base_url: str, include_paths: list[str], max_pages: int, max_depth: int) -> list[CrawledPage]` — Scrapy's Twisted reactor is process-global and can only start once per OS process, and the crawl background task runs inside uvicorn's asyncio event loop. To avoid conflicting with that loop (or with a second crawl job), each call runs the spider in a dedicated subprocess via Python's `multiprocessing` (spawn a worker that runs `CrawlerProcess(...).start()` and returns results through a `multiprocessing.Queue` or a temp file), rather than importing Scrapy's reactor into the main server process.
- Same-domain restriction: spider's `allowed_domains` derived from `base_url`'s netloc.
- Path filtering: only follow/yield links whose path starts with one of `include_paths` (if empty, no path restriction beyond same-domain).
- Bounds: `settings.CRAWL_MAX_PAGES` (default e.g. 200) and `settings.CRAWL_MAX_DEPTH` (default e.g. 5), both overridable per-request.
- `CrawledPage`: `url: str, html: str`.
- Each page's `html` is handed to `parser.py`'s Docling HTML path (same table/structure extraction as PDF/DOCX).

### Async crawl job (`routes/rag_routes.py` + `repository/crawl_jobs.py`)

- `POST /crawl` — body `{channel_id, base_url, include_paths?: list[str], max_pages?: int}`. Validates, creates a job record in Redis, kicks off a background task (asyncio `create_task`, mirroring the existing `sweep_loop` pattern in `main.py`), returns `{job_id, status: "queued"}` immediately.
- `GET /crawl/{job_id}` — returns job status from Redis: `{status: queued|crawling|parsing|embedding|done|failed, pages_found, pages_processed, error?}`.
- `repository/crawl_jobs.py`: `create_job`, `update_job`, `get_job` — thin wrapper over `redis_client`, JSON-serialized job dict, similar TTL discipline to existing session storage (e.g. job record TTL 1 hour after completion).
- The background task: crawl → for each page, parse via Docling → caption figures → chunk → embed (Chroma + BM25) → update job progress after each page. On any page-level failure, log and continue (don't fail the whole job for one bad page); job only moves to `failed` on a crawl-level exception (e.g. base_url unreachable).

## Data flow (crawl job)

1. `POST /crawl` validates payload, `create_job` in Redis (`status=queued`), returns `job_id`.
2. Background task: `status=crawling`, `crawl_site(...)` returns list of `CrawledPage`.
3. `status=parsing`, for each page: Docling parse → caption figures → append to running document list, update `pages_processed` counter after each.
4. `status=embedding`: for each parsed page, `chunk_text` (existing) → `Chroma.from_documents` + `bm25_index.add_documents` (existing path, same as file upload) → `register_document` in the channel manifest (page URL as the "filename").
5. `status=done`.
6. Any exception during crawl itself → `status=failed`, `error` message stored.

## Error handling

- Per-page parse/caption failures: logged, page skipped, job continues (matches CLAUDE.md's existing convention of broad catch + structured logging rather than propagating).
- Crawl-level failure (unreachable base_url, no pages found): job marked `failed` with a message, no partial embedding attempted.
- Docling parse failure on an uploaded file: existing `create_document_embeddings` error path already returns a 500 via the envelope — Docling errors surface the same way PyMuPDF errors do today.

## Settings additions (`app/config/settings.py`)

- `GROQ_VISION_MODEL: str` — vision-capable Groq model for figure captioning.
- `CRAWL_MAX_PAGES: int = 200`
- `CRAWL_MAX_DEPTH: int = 5`
- `CRAWL_JOB_TTL_SECONDS: int = 3600`

## Dependencies added

- `docling` (parsing, replaces PyMuPDF/python-docx text-extraction role — `pymupdf`/`python-docx` may still be needed as Docling internals or can be dropped once confirmed unused elsewhere)
- `scrapy` (crawling)

## Testing

Per existing convention (`tests/`, pytest, mocks for LLM/reranker — though note `tests/` was deleted in a recent merge and needs to exist again for this work): mock Docling's `DocumentConverter` output, mock Scrapy's crawl results, mock the Groq vision call. Test the job state machine (`crawl_jobs.py`) against `fakeredis`. Test `crawler.py`'s path-filtering logic with a small fixture site (or mocked responses) independent of parsing.

## Open items carried forward as explicit future scope

- Qdrant migration (deferred, per user direction)
- Port to closeloop-backend (deferred, separate project, different tenancy/DB model)
- Sarvam OCR (explicitly declined by user in favor of Docling-only)
