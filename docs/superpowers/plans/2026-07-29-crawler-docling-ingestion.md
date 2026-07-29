# Crawler + Docling Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Docling-based document parsing (tables, OCR, figures) for uploads, and a new async URL-crawl ingestion pipeline (Scrapy), both feeding the existing Chroma + BM25 embedding pipeline unchanged.

**Architecture:** New `app/ingestion/` package (`parser.py`, `captioning.py`, `crawler.py`) plus `app/repository/crawl_jobs.py` for Redis-backed job state. Both file uploads and crawl jobs converge on the existing `chunk_text` → `Chroma.from_documents` + `bm25_index.add_documents` flow in `rag_controller.py`. Chroma is unchanged in this plan — Qdrant migration is future scope.

**Tech Stack:** Docling (parsing/OCR/tables), Scrapy (crawling, run in a subprocess via `multiprocessing` to avoid Twisted-reactor conflicts with uvicorn's asyncio loop), Groq vision model (figure captioning), existing Redis client (job state), pytest + fakeredis (tests).

## Global Constraints

- Standardized JSON envelope (`{success, message, data, error}`) on every new endpoint, per CLAUDE.md.
- Errors logged via loguru `logger` (`from app.config.logger import logger`) and swallowed into structured responses at the route/controller boundary, not propagated raw.
- No test runner beyond pytest; mock all network/model calls (Docling model downloads, Groq vision calls, Scrapy HTTP requests) — no live network calls in tests.
- `tests/` directory does not currently exist in this repo (deleted in a prior commit) — Task 1 recreates it.
- Settings go in `app/config/settings.py` as typed `Settings` fields, following existing naming (`UPPER_SNAKE`).
- Chroma vector store and `bm25_index` module are NOT modified — both are consumed as-is.

---

### Task 1: Recreate test scaffolding

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: pytest fixture `fake_redis` (a `fakeredis.FakeRedis` instance) that later tasks' tests import via `from tests.conftest import fake_redis` or use directly as a pytest fixture.

- [ ] **Step 1: Create `tests/__init__.py`**

Empty file.

- [ ] **Step 2: Create `tests/conftest.py`**

```python
import fakeredis
import pytest


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis(decode_responses=False)
```

- [ ] **Step 3: Verify pytest discovers the tests directory**

Run: `.venv\Scripts\python.exe -m pytest --collect-only`
Expected: exits 0, "no tests ran" (no test files yet, but no collection errors)

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: recreate tests/ scaffolding with fakeredis fixture"
```

---

### Task 2: Add new settings fields

**Files:**
- Modify: `app/config/settings.py`

**Interfaces:**
- Produces: `settings.GROQ_VISION_MODEL: str`, `settings.CRAWL_MAX_PAGES: int`, `settings.CRAWL_MAX_DEPTH: int`, `settings.CRAWL_JOB_TTL_SECONDS: int`

- [ ] **Step 1: Add the fields**

In `app/config/settings.py`, inside `class Settings(BaseSettings)`, after the `METRICS_ENABLED: bool = True` line, add:

```python
    # Ingestion: crawling + vision captioning (Phase 5)
    GROQ_VISION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    CRAWL_MAX_PAGES: int = 200
    CRAWL_MAX_DEPTH: int = 5
    CRAWL_JOB_TTL_SECONDS: int = 3600
```

- [ ] **Step 2: Verify settings load**

Run: `.venv\Scripts\python.exe -c "from app.config.settings import settings; print(settings.GROQ_VISION_MODEL, settings.CRAWL_MAX_PAGES, settings.CRAWL_MAX_DEPTH, settings.CRAWL_JOB_TTL_SECONDS)"`
Expected: prints `meta-llama/llama-4-scout-17b-16e-instruct 200 5 3600`

- [ ] **Step 3: Commit**

```bash
git add app/config/settings.py
git commit -m "feat: add settings for crawl limits and vision captioning model"
```

---

### Task 3: Add Docling and Scrapy dependencies

**Files:**
- Modify: `requirements.txt` (UTF-16 encoded — must preserve encoding)

**Interfaces:**
- Produces: `docling`, `scrapy` importable in the venv.

- [ ] **Step 1: Install packages into the venv**

Run: `.venv\Scripts\python.exe -m pip install docling scrapy -q`
Expected: exits 0

- [ ] **Step 2: Verify imports**

Run: `.venv\Scripts\python.exe -c "from docling.document_converter import DocumentConverter; import scrapy; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Append to requirements.txt preserving UTF-16 encoding**

```bash
.venv\Scripts\python.exe -c "
import io
path = 'requirements.txt'
with io.open(path, 'r', encoding='utf-16-le') as f:
    content = f.read()
docling_ver = __import__('importlib.metadata', fromlist=['version']).version('docling')
scrapy_ver = __import__('importlib.metadata', fromlist=['version']).version('scrapy')
lines = content.splitlines()
lines.append(f'docling=={docling_ver}')
lines.append(f'scrapy=={scrapy_ver}')
new_content = '\r\n'.join(lines) + '\r\n'
with io.open(path, 'w', encoding='utf-16-le') as f:
    f.write(new_content)
print('appended', docling_ver, scrapy_ver)
"
```
Expected: prints the appended versions, no error

- [ ] **Step 4: Verify file still parses as UTF-16 with new lines present**

Run: `.venv\Scripts\python.exe -c "import io; print([l for l in io.open('requirements.txt', encoding='utf-16-le').read().splitlines() if 'docling' in l or 'scrapy' in l])"`
Expected: prints both new lines

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "chore: add docling and scrapy dependencies"
```

---

### Task 4: `app/ingestion/parser.py` — Docling-based document parsing

**Files:**
- Create: `app/ingestion/__init__.py`
- Create: `app/ingestion/parser.py`
- Test: `tests/test_ingestion_parser.py`

**Interfaces:**
- Produces:
  - `dataclass ExtractedFigure(image_bytes: bytes, position_hint: str)`
  - `dataclass ParsedDocument(text_blocks: list[str], tables: list[str], figures: list[ExtractedFigure])`
  - `ParsedDocument.to_text_stream() -> str` — joins `text_blocks` and `tables` (markdown) into one ordered string, with a placeholder `f"[[FIGURE:{i}]]"` inline for each figure at the position it appeared (figures list index order — this plan doesn't attempt true positional interleaving, since Docling's element order already approximates reading order; each `[[FIGURE:i]]` placeholder is appended after all text_blocks/tables for simplicity of this first pass)
  - `parse_document(source: str, source_type: Literal["pdf", "docx", "html"]) -> ParsedDocument` — `source` is a file path for `"pdf"`/`"docx"`, raw HTML string for `"html"`.

- [ ] **Step 1: Create `app/ingestion/__init__.py`**

Empty file.

- [ ] **Step 2: Write the failing test**

Create `tests/test_ingestion_parser.py`:

```python
from unittest.mock import MagicMock, patch

from app.ingestion.parser import parse_document, ParsedDocument, ExtractedFigure


def _make_mock_docling_document(text_items, table_items, picture_items):
    mock_doc = MagicMock()

    text_mocks = []
    for text in text_items:
        m = MagicMock()
        m.text = text
        text_mocks.append(m)
    mock_doc.texts = text_mocks

    table_mocks = []
    for md in table_items:
        m = MagicMock()
        m.export_to_markdown.return_value = md
        table_mocks.append(m)
    mock_doc.tables = table_mocks

    picture_mocks = []
    for img_bytes in picture_items:
        m = MagicMock()
        pil_image_mock = MagicMock()

        def save_side_effect(buf, format=None, _b=img_bytes):
            buf.write(_b)

        pil_image_mock.save.side_effect = save_side_effect
        m.get_image.return_value = pil_image_mock
        picture_mocks.append(m)
    mock_doc.pictures = picture_mocks

    return mock_doc


def test_parse_pdf_extracts_text_tables_and_figures():
    mock_document = _make_mock_docling_document(
        text_items=["First paragraph.", "Second paragraph."],
        table_items=["| a | b |\n|---|---|\n| 1 | 2 |"],
        picture_items=[b"fake-png-bytes"],
    )
    mock_result = MagicMock()
    mock_result.document = mock_document

    with patch("app.ingestion.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_result

        parsed = parse_document("some/path.pdf", source_type="pdf")

    assert parsed.text_blocks == ["First paragraph.", "Second paragraph."]
    assert parsed.tables == ["| a | b |\n|---|---|\n| 1 | 2 |"]
    assert len(parsed.figures) == 1
    assert parsed.figures[0].image_bytes == b"fake-png-bytes"


def test_parsed_document_to_text_stream_includes_all_parts():
    parsed = ParsedDocument(
        text_blocks=["Hello world."],
        tables=["| x |\n|---|\n| 1 |"],
        figures=[ExtractedFigure(image_bytes=b"abc", position_hint="page1")],
    )
    stream = parsed.to_text_stream()
    assert "Hello world." in stream
    assert "| x |" in stream
    assert "[[FIGURE:0]]" in stream


def test_parse_document_empty_source_returns_empty_parsed_document():
    mock_document = _make_mock_docling_document([], [], [])
    mock_result = MagicMock()
    mock_result.document = mock_document

    with patch("app.ingestion.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_result
        parsed = parse_document("empty.pdf", source_type="pdf")

    assert parsed.text_blocks == []
    assert parsed.tables == []
    assert parsed.figures == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ingestion_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingestion.parser'`

- [ ] **Step 4: Write `app/ingestion/parser.py`**

```python
import io
from dataclasses import dataclass, field
from typing import Literal

from docling.document_converter import DocumentConverter

from app.config.logger import logger


@dataclass
class ExtractedFigure:
    image_bytes: bytes
    position_hint: str


@dataclass
class ParsedDocument:
    text_blocks: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    figures: list[ExtractedFigure] = field(default_factory=list)

    def to_text_stream(self) -> str:
        parts = list(self.text_blocks) + list(self.tables)
        for i in range(len(self.figures)):
            parts.append(f"[[FIGURE:{i}]]")
        return "\n\n".join(parts)


def _extract_figure_bytes(picture) -> bytes | None:
    try:
        pil_image = picture.get_image()
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Failed to extract figure image: {e}")
        return None


def parse_document(source: str, source_type: Literal["pdf", "docx", "html"]) -> ParsedDocument:
    """Parse a PDF/DOCX file path or raw HTML string into text, tables, and figures via Docling."""
    converter = DocumentConverter()
    result = converter.convert(source)
    doc = result.document

    text_blocks = [t.text for t in doc.texts if getattr(t, "text", "").strip()]
    tables = [t.export_to_markdown() for t in doc.tables]

    figures = []
    for i, picture in enumerate(doc.pictures):
        img_bytes = _extract_figure_bytes(picture)
        if img_bytes:
            figures.append(ExtractedFigure(image_bytes=img_bytes, position_hint=f"figure-{i}"))

    return ParsedDocument(text_blocks=text_blocks, tables=tables, figures=figures)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ingestion_parser.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add app/ingestion/__init__.py app/ingestion/parser.py tests/test_ingestion_parser.py
git commit -m "feat: add Docling-based document parser (text, tables, figures)"
```

---

### Task 5: `app/ingestion/captioning.py` — figure captioning via Groq vision

**Files:**
- Create: `app/ingestion/captioning.py`
- Test: `tests/test_ingestion_captioning.py`

**Interfaces:**
- Consumes: `settings.GROQ_VISION_MODEL`, `settings.GROQ_API_KEY` (both from Task 2 / existing settings)
- Produces: `caption_figure(image_bytes: bytes) -> str` — returns a text description, or `""` on failure (never raises).

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestion_captioning.py`:

```python
from unittest.mock import MagicMock, patch

from app.ingestion.captioning import caption_figure


def test_caption_figure_returns_llm_text_on_success():
    mock_response = MagicMock()
    mock_response.content = "A bar chart showing quarterly revenue."

    with patch("app.ingestion.captioning._get_vision_llm") as mock_get_llm:
        mock_get_llm.return_value.invoke.return_value = mock_response
        result = caption_figure(b"fake-image-bytes")

    assert result == "A bar chart showing quarterly revenue."


def test_caption_figure_returns_empty_string_on_failure():
    with patch("app.ingestion.captioning._get_vision_llm") as mock_get_llm:
        mock_get_llm.return_value.invoke.side_effect = RuntimeError("API error")
        result = caption_figure(b"fake-image-bytes")

    assert result == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ingestion_captioning.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingestion.captioning'`

- [ ] **Step 3: Write `app/ingestion/captioning.py`**

```python
import base64

from langchain_core.messages import HumanMessage

from app.config.logger import logger
from app.config.settings import settings

_vision_llm_instance = None


def _get_vision_llm():
    global _vision_llm_instance
    if _vision_llm_instance is None:
        from langchain_groq import ChatGroq
        _vision_llm_instance = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            temperature=0.1,
            model_name=settings.GROQ_VISION_MODEL,
        )
    return _vision_llm_instance


def caption_figure(image_bytes: bytes) -> str:
    """Generate a short text description of a figure/diagram image via a vision-capable LLM.

    Returns an empty string on any failure so one bad image never fails a whole document ingest.
    """
    try:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this figure or diagram in 1-2 concise sentences, focusing on what information it conveys."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
            ]
        )
        response = _get_vision_llm().invoke([message])
        return response.content
    except Exception as e:
        logger.warning(f"Figure captioning failed: {e}")
        return ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ingestion_captioning.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/ingestion/captioning.py tests/test_ingestion_captioning.py
git commit -m "feat: add figure captioning via Groq vision model"
```

---

### Task 6: Wire captioning into `ParsedDocument.to_text_stream`

**Files:**
- Modify: `app/ingestion/parser.py`
- Test: `tests/test_ingestion_parser.py`

**Interfaces:**
- Consumes: `caption_figure(image_bytes: bytes) -> str` from Task 5
- Produces: `ParsedDocument.to_text_stream()` now calls `caption_figure` per figure and inlines `f"[Figure: {caption}]"` instead of the raw `[[FIGURE:i]]` placeholder (empty captions are omitted entirely rather than leaving an empty `[Figure: ]` tag)

- [ ] **Step 1: Update the existing text-stream test to expect captions**

In `tests/test_ingestion_parser.py`, replace `test_parsed_document_to_text_stream_includes_all_parts` with:

```python
def test_parsed_document_to_text_stream_includes_captions():
    with patch("app.ingestion.parser.caption_figure") as mock_caption:
        mock_caption.return_value = "A flowchart of the onboarding process."

        parsed = ParsedDocument(
            text_blocks=["Hello world."],
            tables=["| x |\n|---|\n| 1 |"],
            figures=[ExtractedFigure(image_bytes=b"abc", position_hint="page1")],
        )
        stream = parsed.to_text_stream()

    assert "Hello world." in stream
    assert "| x |" in stream
    assert "[Figure: A flowchart of the onboarding process.]" in stream


def test_parsed_document_to_text_stream_skips_empty_captions():
    with patch("app.ingestion.parser.caption_figure") as mock_caption:
        mock_caption.return_value = ""

        parsed = ParsedDocument(
            text_blocks=["Hello world."],
            tables=[],
            figures=[ExtractedFigure(image_bytes=b"abc", position_hint="page1")],
        )
        stream = parsed.to_text_stream()

    assert "Hello world." in stream
    assert "[Figure:" not in stream
```

Also add `from unittest.mock import patch` to the top imports if not already present (it already is, from Step 2 of Task 4).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ingestion_parser.py -v`
Expected: FAIL — `to_text_stream` still emits `[[FIGURE:i]]`, not captions

- [ ] **Step 3: Update `app/ingestion/parser.py`**

Add the import and update `to_text_stream`:

```python
from app.ingestion.captioning import caption_figure
```

Replace the `to_text_stream` method body:

```python
    def to_text_stream(self) -> str:
        parts = list(self.text_blocks) + list(self.tables)
        for figure in self.figures:
            caption = caption_figure(figure.image_bytes)
            if caption:
                parts.append(f"[Figure: {caption}]")
        return "\n\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ingestion_parser.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add app/ingestion/parser.py tests/test_ingestion_parser.py
git commit -m "feat: caption figures inline when building document text stream"
```

---

### Task 7: Replace `RAGService.get_text` PDF/DOCX extraction with Docling in the upload path

**Files:**
- Modify: `app/controller/rag_controller.py:33-74` (`create_document_embeddings`)
- Test: `tests/test_ingestion.py` (recreate — this test existed before the earlier test-suite deletion, per CLAUDE.md's description of prior coverage)

**Interfaces:**
- Consumes: `parse_document(source, source_type) -> ParsedDocument` (Task 4), `ParsedDocument.to_text_stream() -> str` (Task 6)
- Produces: `RAGController.create_document_embeddings(channel_id, file_path)` — same signature and same return shape (`{"message", "doc_id", "path", "chunks"}` or `None`) as before, now parses via Docling instead of `RAGService.get_text`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestion.py`:

```python
from unittest.mock import MagicMock, patch

import pytest

from app.controller.rag_controller import RAGController


@pytest.fixture
def controller():
    with patch("app.controller.rag_controller.RAGUtilities") as MockUtils:
        MockUtils.return_value.get_embedding_model.return_value = MagicMock()
        yield RAGController()


def test_create_document_embeddings_uses_docling_parser(tmp_path, controller):
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake pdf bytes")

    mock_parsed = MagicMock()
    mock_parsed.to_text_stream.return_value = "Extracted paragraph text via Docling."

    with patch("app.controller.rag_controller.parse_document", return_value=mock_parsed) as mock_parse, \
         patch("app.controller.rag_controller.Chroma") as MockChroma, \
         patch("app.controller.rag_controller.bm25_index") as mock_bm25:

        result = controller.create_document_embeddings(
            channel_id="chan1", file_path=str(file_path)
        )

    mock_parse.assert_called_once_with(str(file_path), source_type="pdf")
    assert result is not None
    assert result["chunks"] > 0
    MockChroma.from_documents.assert_called_once()
    mock_bm25.add_documents.assert_called_once()


def test_create_document_embeddings_returns_none_for_empty_parse(tmp_path, controller):
    file_path = tmp_path / "empty.docx"
    file_path.write_bytes(b"fake docx bytes")

    mock_parsed = MagicMock()
    mock_parsed.to_text_stream.return_value = ""

    with patch("app.controller.rag_controller.parse_document", return_value=mock_parsed):
        result = controller.create_document_embeddings(
            channel_id="chan1", file_path=str(file_path)
        )

    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ingestion.py -v`
Expected: FAIL — `parse_document` is not imported/used in `rag_controller.py` yet

- [ ] **Step 3: Modify `app/controller/rag_controller.py`**

Add the import near the top (after the existing `from app.retrieval import bm25_index` line):

```python
from app.ingestion.parser import parse_document
```

Replace the body of `create_document_embeddings` (currently calling `RAGService.get_text(file_path)`) — change these lines:

```python
            filename = os.path.basename(file_path)
            logger.info(f"Embedding '{filename}' into channel '{channel_id}'")

            text = RAGService.get_text(file_path)
            if not text:
                logger.warning(f"No content extracted from file: {filename}.")
                return None
```

to:

```python
            filename = os.path.basename(file_path)
            logger.info(f"Embedding '{filename}' into channel '{channel_id}'")

            ext = os.path.splitext(filename)[1].lower().lstrip(".")
            source_type = "pdf" if ext == "pdf" else "docx"
            parsed = parse_document(file_path, source_type=source_type)
            text = parsed.to_text_stream()
            if not text:
                logger.warning(f"No content extracted from file: {filename}.")
                return None
```

The `RAGService` import can stay (still potentially used elsewhere) — do not remove it in this task.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ingestion.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `.venv\Scripts\python.exe -m pytest -v`
Expected: all tests PASS (no other test files reference `RAGService.get_text` from the controller path at this point, since this is the first ingestion test recreated)

- [ ] **Step 6: Commit**

```bash
git add app/controller/rag_controller.py tests/test_ingestion.py
git commit -m "feat: parse uploaded PDF/DOCX via Docling instead of PyMuPDF/python-docx"
```

---

### Task 8: `app/repository/crawl_jobs.py` — Redis-backed crawl job state

**Files:**
- Create: `app/repository/crawl_jobs.py`
- Test: `tests/test_crawl_jobs.py`

**Interfaces:**
- Consumes: `redis_client` from `app.database.redis` (existing), `settings.CRAWL_JOB_TTL_SECONDS` (Task 2)
- Produces:
  - `create_job(job_id: str, channel_id: str, base_url: str) -> bool`
  - `update_job(job_id: str, **fields) -> bool` — merges `fields` into the stored job dict (e.g. `status="crawling"`, `pages_found=12`)
  - `get_job(job_id: str) -> dict | None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_crawl_jobs.py`:

```python
from unittest.mock import patch

from app.repository import crawl_jobs


def test_create_and_get_job(fake_redis):
    with patch("app.repository.crawl_jobs.redis_client", fake_redis):
        ok = crawl_jobs.create_job("job1", channel_id="chan1", base_url="https://example.com")
        assert ok is True

        job = crawl_jobs.get_job("job1")

    assert job["channel_id"] == "chan1"
    assert job["base_url"] == "https://example.com"
    assert job["status"] == "queued"
    assert job["pages_found"] == 0
    assert job["pages_processed"] == 0


def test_update_job_merges_fields(fake_redis):
    with patch("app.repository.crawl_jobs.redis_client", fake_redis):
        crawl_jobs.create_job("job2", channel_id="chan1", base_url="https://example.com")
        crawl_jobs.update_job("job2", status="crawling", pages_found=5)

        job = crawl_jobs.get_job("job2")

    assert job["status"] == "crawling"
    assert job["pages_found"] == 5
    assert job["base_url"] == "https://example.com"  # untouched fields survive


def test_get_job_returns_none_when_missing(fake_redis):
    with patch("app.repository.crawl_jobs.redis_client", fake_redis):
        job = crawl_jobs.get_job("does-not-exist")

    assert job is None


def test_create_job_returns_false_when_redis_unavailable():
    with patch("app.repository.crawl_jobs.redis_client", None):
        ok = crawl_jobs.create_job("job3", channel_id="chan1", base_url="https://example.com")

    assert ok is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_crawl_jobs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.repository.crawl_jobs'`

- [ ] **Step 3: Write `app/repository/crawl_jobs.py`**

```python
import json

from app.database.redis import redis_client
from app.config.logger import logger
from app.config.settings import settings


def _job_key(job_id: str) -> str:
    return f"crawl_job:{job_id}"


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def create_job(job_id: str, channel_id: str, base_url: str) -> bool:
    """Create a new crawl job record with status=queued."""
    if redis_client is None:
        logger.warning("Redis unavailable; cannot create crawl job")
        return False
    job = {
        "channel_id": channel_id,
        "base_url": base_url,
        "status": "queued",
        "pages_found": 0,
        "pages_processed": 0,
        "error": None,
    }
    try:
        redis_client.setex(_job_key(job_id), settings.CRAWL_JOB_TTL_SECONDS, json.dumps(job))
        return True
    except Exception as e:
        logger.error(f"create_job failed: {e}")
        return False


def update_job(job_id: str, **fields) -> bool:
    """Merge fields into an existing job record, refreshing its TTL."""
    if redis_client is None:
        logger.warning("Redis unavailable; cannot update crawl job")
        return False
    try:
        existing = get_job(job_id) or {}
        existing.update(fields)
        redis_client.setex(_job_key(job_id), settings.CRAWL_JOB_TTL_SECONDS, json.dumps(existing))
        return True
    except Exception as e:
        logger.error(f"update_job failed: {e}")
        return False


def get_job(job_id: str) -> dict | None:
    """Return the job record dict, or None if missing/unavailable."""
    if redis_client is None:
        return None
    try:
        raw = redis_client.get(_job_key(job_id))
        if raw is None:
            return None
        return json.loads(_decode(raw))
    except Exception as e:
        logger.error(f"get_job failed: {e}")
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_crawl_jobs.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add app/repository/crawl_jobs.py tests/test_crawl_jobs.py
git commit -m "feat: add Redis-backed crawl job state repository"
```

---

### Task 9: `app/ingestion/crawler.py` — Scrapy-based same-domain crawler

**Files:**
- Create: `app/ingestion/crawler.py`
- Test: `tests/test_ingestion_crawler.py`

**Interfaces:**
- Produces:
  - `dataclass CrawledPage(url: str, html: str)`
  - `crawl_site(base_url: str, include_paths: list[str], max_pages: int, max_depth: int) -> list[CrawledPage]`
  - Internal: `_run_spider_subprocess(base_url, include_paths, max_pages, max_depth, result_queue)` — the function executed in the child process (module-level, not nested, so it's picklable by `multiprocessing`)
  - Internal helper `_path_allowed(url: str, include_paths: list[str]) -> bool` — used directly by tests to verify filtering logic without running Scrapy

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestion_crawler.py`:

```python
from unittest.mock import patch

from app.ingestion.crawler import _path_allowed, crawl_site, CrawledPage


def test_path_allowed_with_no_filter_allows_everything():
    assert _path_allowed("https://example.com/anything", []) is True


def test_path_allowed_matches_prefix():
    assert _path_allowed("https://example.com/docs/page1", ["/docs"]) is True
    assert _path_allowed("https://example.com/blog/post1", ["/docs"]) is False


def test_path_allowed_matches_any_of_multiple_prefixes():
    assert _path_allowed("https://example.com/blog/post1", ["/docs", "/blog"]) is True


def test_crawl_site_returns_pages_from_subprocess():
    fake_pages = [
        CrawledPage(url="https://example.com/docs/a", html="<html>A</html>"),
        CrawledPage(url="https://example.com/docs/b", html="<html>B</html>"),
    ]

    with patch("app.ingestion.crawler._run_spider_and_collect", return_value=fake_pages) as mock_run:
        pages = crawl_site(
            base_url="https://example.com",
            include_paths=["/docs"],
            max_pages=50,
            max_depth=3,
        )

    mock_run.assert_called_once_with("https://example.com", ["/docs"], 50, 3)
    assert pages == fake_pages
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ingestion_crawler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingestion.crawler'`

- [ ] **Step 3: Write `app/ingestion/crawler.py`**

```python
import multiprocessing
from dataclasses import dataclass
from urllib.parse import urlparse

from app.config.logger import logger


@dataclass
class CrawledPage:
    url: str
    html: str


def _path_allowed(url: str, include_paths: list[str]) -> bool:
    """Return True if url's path starts with one of include_paths (or include_paths is empty)."""
    if not include_paths:
        return True
    path = urlparse(url).path
    return any(path.startswith(prefix) for prefix in include_paths)


def _spider_worker(base_url: str, include_paths: list[str], max_pages: int,
                    max_depth: int, result_queue: multiprocessing.Queue) -> None:
    """Runs inside a child process: starts a Scrapy CrawlerProcess and pushes results to the queue."""
    import scrapy
    from scrapy.crawler import CrawlerProcess

    domain = urlparse(base_url).netloc
    collected: list[dict] = []

    class SiteSpider(scrapy.Spider):
        name = "site_spider"
        allowed_domains = [domain]
        start_urls = [base_url]

        custom_settings = {
            "DEPTH_LIMIT": max_depth,
            "CLOSESPIDER_PAGECOUNT": max_pages,
            "LOG_ENABLED": False,
            "ROBOTSTXT_OBEY": True,
        }

        def parse(self, response):
            if _path_allowed(response.url, include_paths):
                collected.append({"url": response.url, "html": response.text})
            for href in response.css("a::attr(href)").getall():
                next_url = response.urljoin(href)
                if urlparse(next_url).netloc == domain and _path_allowed(next_url, include_paths):
                    yield response.follow(next_url, callback=self.parse)

    process = CrawlerProcess(settings={"LOG_ENABLED": False})
    process.crawl(SiteSpider)
    process.start()

    result_queue.put(collected)


def _run_spider_and_collect(base_url: str, include_paths: list[str],
                             max_pages: int, max_depth: int) -> list[CrawledPage]:
    """Runs the Scrapy spider in a subprocess (Scrapy's Twisted reactor can only start once
    per OS process, and the caller runs inside uvicorn's asyncio loop) and collects results."""
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_spider_worker,
        args=(base_url, include_paths, max_pages, max_depth, result_queue),
    )
    proc.start()
    proc.join()

    if result_queue.empty():
        logger.warning(f"Crawl subprocess for {base_url} produced no results")
        return []

    raw_pages = result_queue.get()
    return [CrawledPage(url=p["url"], html=p["html"]) for p in raw_pages]


def crawl_site(base_url: str, include_paths: list[str], max_pages: int, max_depth: int) -> list[CrawledPage]:
    """Crawl base_url's domain (optionally restricted to include_paths prefixes), bounded by
    max_pages/max_depth, returning each matched page's URL and raw HTML."""
    return _run_spider_and_collect(base_url, include_paths, max_pages, max_depth)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ingestion_crawler.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add app/ingestion/crawler.py tests/test_ingestion_crawler.py
git commit -m "feat: add Scrapy-based same-domain crawler with path filtering"
```

---

### Task 10: Crawl-to-embedding pipeline function

**Files:**
- Create: `app/ingestion/pipeline.py`
- Test: `tests/test_ingestion_pipeline.py`

**Interfaces:**
- Consumes: `crawl_site(...)` (Task 9), `parse_document(source, source_type)` + `ParsedDocument.to_text_stream()` (Tasks 4/6), `chunk_text(text, channel_id, filename)` from `app.retrieval.chunking` (existing), `Chroma` from `langchain_chroma` (existing usage pattern from `rag_controller.py`), `bm25_index.add_documents` (existing), `register_document` from `app.repository.channel_repository` (existing), `update_job` from `app.repository.crawl_jobs` (Task 8)
- Produces: `run_crawl_job(job_id: str, channel_id: str, base_url: str, include_paths: list[str], max_pages: int, max_depth: int, embedding_model) -> None` — the full background-task body; updates job status via `update_job` at each stage, never raises (catches at the top level and calls `update_job(job_id, status="failed", error=str(e))`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestion_pipeline.py`:

```python
from unittest.mock import MagicMock, patch

from app.ingestion.pipeline import run_crawl_job
from app.ingestion.crawler import CrawledPage


def test_run_crawl_job_happy_path():
    fake_pages = [
        CrawledPage(url="https://example.com/docs/a", html="<html>A content</html>"),
        CrawledPage(url="https://example.com/docs/b", html="<html>B content</html>"),
    ]
    mock_parsed = MagicMock()
    mock_parsed.to_text_stream.return_value = "Some extracted page text."

    with patch("app.ingestion.pipeline.crawl_site", return_value=fake_pages) as mock_crawl, \
         patch("app.ingestion.pipeline.parse_document", return_value=mock_parsed) as mock_parse, \
         patch("app.ingestion.pipeline.Chroma") as MockChroma, \
         patch("app.ingestion.pipeline.bm25_index") as mock_bm25, \
         patch("app.ingestion.pipeline.register_document") as mock_register, \
         patch("app.ingestion.pipeline.update_job") as mock_update_job:

        run_crawl_job(
            job_id="job1",
            channel_id="chan1",
            base_url="https://example.com",
            include_paths=["/docs"],
            max_pages=50,
            max_depth=3,
            embedding_model=MagicMock(),
        )

    mock_crawl.assert_called_once_with(
        base_url="https://example.com", include_paths=["/docs"], max_pages=50, max_depth=3
    )
    assert mock_parse.call_count == 2
    assert MockChroma.from_documents.call_count == 2
    assert mock_bm25.add_documents.call_count == 2
    assert mock_register.call_count == 2

    status_calls = [c.kwargs.get("status") for c in mock_update_job.call_args_list if "status" in c.kwargs]
    assert "crawling" in status_calls
    assert "parsing" in status_calls
    assert "embedding" in status_calls
    assert "done" in status_calls


def test_run_crawl_job_marks_failed_on_crawl_exception():
    with patch("app.ingestion.pipeline.crawl_site", side_effect=RuntimeError("unreachable")), \
         patch("app.ingestion.pipeline.update_job") as mock_update_job:

        run_crawl_job(
            job_id="job2",
            channel_id="chan1",
            base_url="https://bad-url.invalid",
            include_paths=[],
            max_pages=50,
            max_depth=3,
            embedding_model=MagicMock(),
        )

    fail_calls = [c for c in mock_update_job.call_args_list if c.kwargs.get("status") == "failed"]
    assert len(fail_calls) == 1
    assert "unreachable" in fail_calls[0].kwargs["error"]


def test_run_crawl_job_skips_page_on_parse_failure_but_continues():
    fake_pages = [
        CrawledPage(url="https://example.com/docs/a", html="<html>A</html>"),
        CrawledPage(url="https://example.com/docs/b", html="<html>B</html>"),
    ]
    good_parsed = MagicMock()
    good_parsed.to_text_stream.return_value = "Good page text."

    with patch("app.ingestion.pipeline.crawl_site", return_value=fake_pages), \
         patch("app.ingestion.pipeline.parse_document", side_effect=[RuntimeError("bad html"), good_parsed]), \
         patch("app.ingestion.pipeline.Chroma") as MockChroma, \
         patch("app.ingestion.pipeline.bm25_index") as mock_bm25, \
         patch("app.ingestion.pipeline.register_document"), \
         patch("app.ingestion.pipeline.update_job") as mock_update_job:

        run_crawl_job(
            job_id="job3",
            channel_id="chan1",
            base_url="https://example.com",
            include_paths=["/docs"],
            max_pages=50,
            max_depth=3,
            embedding_model=MagicMock(),
        )

    assert MockChroma.from_documents.call_count == 1
    assert mock_bm25.add_documents.call_count == 1
    status_calls = [c.kwargs.get("status") for c in mock_update_job.call_args_list if "status" in c.kwargs]
    assert "done" in status_calls
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ingestion_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingestion.pipeline'`

- [ ] **Step 3: Write `app/ingestion/pipeline.py`**

```python
import os

from langchain_chroma import Chroma

from app.config.logger import logger
from app.config.settings import settings
from app.ingestion.crawler import crawl_site
from app.ingestion.parser import parse_document
from app.repository import bm25_index as _bm25_index_module
from app.repository.channel_repository import register_document
from app.repository.crawl_jobs import update_job
from app.retrieval.chunking import chunk_text

# Imported this way so tests can patch `app.ingestion.pipeline.bm25_index`
from app.retrieval import bm25_index


def run_crawl_job(job_id: str, channel_id: str, base_url: str, include_paths: list[str],
                   max_pages: int, max_depth: int, embedding_model) -> None:
    """Full crawl -> parse -> caption -> chunk -> embed pipeline for one crawl job.

    Never raises: crawl-level failures mark the job 'failed'; per-page parse failures
    are logged and skipped so one bad page doesn't fail the whole job.
    """
    try:
        update_job(job_id, status="crawling")
        pages = crawl_site(base_url=base_url, include_paths=include_paths,
                            max_pages=max_pages, max_depth=max_depth)
        update_job(job_id, pages_found=len(pages))

        update_job(job_id, status="parsing")
        parsed_texts: list[tuple[str, str]] = []  # (url, text)
        for page in pages:
            try:
                parsed = parse_document(page.html, source_type="html")
                text = parsed.to_text_stream()
                if text:
                    parsed_texts.append((page.url, text))
            except Exception as e:
                logger.warning(f"Failed to parse page {page.url}: {e}")

        update_job(job_id, status="embedding")
        persist_directory = os.path.join(settings.EMBEDDING_DIR, channel_id)
        os.makedirs(persist_directory, exist_ok=True)

        pages_processed = 0
        for url, text in parsed_texts:
            docs = chunk_text(text, channel_id=channel_id, filename=url)
            if not docs:
                continue
            doc_id = docs[0].metadata["doc_id"]
            Chroma.from_documents(
                documents=docs,
                embedding=embedding_model,
                persist_directory=persist_directory,
                collection_name=settings.CHROMA_COLLECTION_NAME,
            )
            bm25_index.add_documents(channel_id, docs)
            register_document(channel_id, doc_id, url)
            pages_processed += 1
            update_job(job_id, pages_processed=pages_processed)

        update_job(job_id, status="done")

    except Exception as e:
        logger.error(f"Crawl job {job_id} failed: {e}")
        update_job(job_id, status="failed", error=str(e))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ingestion_pipeline.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/ingestion/pipeline.py tests/test_ingestion_pipeline.py
git commit -m "feat: add crawl-to-embedding pipeline (crawl, parse, chunk, embed)"
```

---

### Task 11: `POST /crawl` and `GET /crawl/{job_id}` routes

**Files:**
- Modify: `app/routes/rag_routes.py`
- Test: `tests/test_crawl_route.py`

**Interfaces:**
- Consumes: `create_job`, `get_job` from `app.repository.crawl_jobs` (Task 8), `run_crawl_job` from `app.ingestion.pipeline` (Task 10), `RAGUtilities().get_embedding_model()` (existing, same pattern `RAGController.__init__` already uses), `settings.CRAWL_MAX_PAGES`/`CRAWL_MAX_DEPTH` (Task 2)
- Produces: `POST /api/v1/rag-chatbot/crawl` and `GET /api/v1/rag-chatbot/crawl/{job_id}`, both behind `Depends(require_api_key)` like `/upload` and `/chat`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_crawl_route.py`:

```python
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_post_crawl_returns_job_id_and_queued_status():
    with patch("app.routes.rag_routes.create_job", return_value=True) as mock_create, \
         patch("app.routes.rag_routes.asyncio.create_task") as mock_create_task, \
         patch("app.routes.rag_routes.settings") as mock_settings:
        mock_settings.CRAWL_MAX_PAGES = 200
        mock_settings.CRAWL_MAX_DEPTH = 5
        mock_settings.api_keys_list.return_value = []  # auth disabled (dev default)

        response = client.post(
            "/api/v1/rag-chatbot/crawl",
            json={"channel_id": "chan1", "base_url": "https://example.com", "include_paths": ["/docs"]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "job_id" in body["data"]
    assert body["data"]["status"] == "queued"
    mock_create.assert_called_once()
    mock_create_task.assert_called_once()


def test_get_crawl_job_returns_status():
    fake_job = {
        "channel_id": "chan1",
        "base_url": "https://example.com",
        "status": "embedding",
        "pages_found": 10,
        "pages_processed": 4,
        "error": None,
    }
    with patch("app.routes.rag_routes.get_job", return_value=fake_job):
        response = client.get("/api/v1/rag-chatbot/crawl/some-job-id")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["status"] == "embedding"
    assert body["data"]["pages_processed"] == 4


def test_get_crawl_job_returns_404_when_missing():
    with patch("app.routes.rag_routes.get_job", return_value=None):
        response = client.get("/api/v1/rag-chatbot/crawl/does-not-exist")

    assert response.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_crawl_route.py -v`
Expected: FAIL — routes don't exist yet (404 on POST, or import errors for patched names)

- [ ] **Step 3: Modify `app/routes/rag_routes.py`**

Add imports near the top (after the existing `from app.repository.channel_repository import register_document` line):

```python
import asyncio
import uuid

from app.repository.crawl_jobs import create_job, get_job
from app.ingestion.pipeline import run_crawl_job
from app.utilities.rag_utilities import RAGUtilities
```

Add a new request model near `ChatRequest`:

```python
class CrawlRequest(BaseModel):
    channel_id: str
    base_url: str
    include_paths: list[str] = []
    max_pages: int | None = None
    max_depth: int | None = None
```

Add the two new routes (place after the `/upload` route, before `/chat`):

```python
@router.post("/crawl", dependencies=[Depends(require_api_key)])
@limiter.limit(lambda: settings.RATE_LIMIT_UPLOAD)
async def start_crawl(request: Request, body: CrawlRequest):
    """Start an async crawl job: scrape a site's same-domain pages and embed them into a channel."""
    job_id = str(uuid.uuid4())
    max_pages = body.max_pages or settings.CRAWL_MAX_PAGES
    max_depth = body.max_depth or settings.CRAWL_MAX_DEPTH

    ok = create_job(job_id, channel_id=body.channel_id, base_url=body.base_url)
    if not ok:
        return create_error_response("Failed to create crawl job.", 500)

    embedding_model = RAGUtilities().get_embedding_model()
    asyncio.create_task(
        asyncio.to_thread(
            run_crawl_job,
            job_id=job_id,
            channel_id=body.channel_id,
            base_url=body.base_url,
            include_paths=body.include_paths,
            max_pages=max_pages,
            max_depth=max_depth,
            embedding_model=embedding_model,
        )
    )

    return {
        "success": True,
        "message": "Crawl job started",
        "data": {"job_id": job_id, "status": "queued"},
        "error": None,
    }


@router.get("/crawl/{job_id}", dependencies=[Depends(require_api_key)])
async def get_crawl_status(job_id: str):
    """Poll the status of a crawl job."""
    job = get_job(job_id)
    if job is None:
        return create_error_response("Crawl job not found.", 404)

    return {
        "success": True,
        "message": "Crawl job status",
        "data": job,
        "error": None,
    }
```

Note: `run_crawl_job` is synchronous (it internally spawns a blocking `multiprocessing` subprocess for the crawl step), so it's dispatched via `asyncio.to_thread` inside the background `asyncio.create_task` to avoid blocking uvicorn's event loop — matching the async-task pattern already used for `sweep_loop` in `main.py` while keeping the blocking work off the loop thread.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_crawl_route.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full test suite**

Run: `.venv\Scripts\python.exe -m pytest -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/routes/rag_routes.py tests/test_crawl_route.py
git commit -m "feat: add POST /crawl and GET /crawl/{job_id} endpoints"
```

---

### Task 12: Route scraped HTML pages through the same Docling table/figure handling as file uploads (verification pass)

This task is a targeted check, not new code: Task 10's `run_crawl_job` already calls `parse_document(page.html, source_type="html")`, which (via Task 4/6) already extracts tables and captions figures identically to the PDF/DOCX path. This task writes one integration-style test proving a scraped page with an HTML `<table>` and an `<img>` ends up with both a markdown table and a figure caption in the final embedded text — closing the loop on the spec's requirement that crawled content gets the same table/diagram treatment as uploads.

**Files:**
- Test: `tests/test_ingestion_pipeline.py` (add one test)

**Interfaces:**
- Consumes: everything from Tasks 4, 6, 9, 10 — no new production code.

- [ ] **Step 1: Write the test**

Append to `tests/test_ingestion_pipeline.py`:

```python
def test_run_crawl_job_html_page_gets_table_and_figure_via_docling():
    from app.ingestion.parser import ParsedDocument, ExtractedFigure

    fake_pages = [CrawledPage(url="https://example.com/docs/report", html="<html>...</html>")]

    real_parsed = ParsedDocument(
        text_blocks=["Quarterly report."],
        tables=["| Q1 | Q2 |\n|----|----|\n| 10 | 20 |"],
        figures=[ExtractedFigure(image_bytes=b"chart-bytes", position_hint="fig1")],
    )

    with patch("app.ingestion.pipeline.crawl_site", return_value=fake_pages), \
         patch("app.ingestion.pipeline.parse_document", return_value=real_parsed), \
         patch("app.ingestion.parser.caption_figure", return_value="Revenue grew from 10 to 20."), \
         patch("app.ingestion.pipeline.Chroma") as MockChroma, \
         patch("app.ingestion.pipeline.bm25_index") as mock_bm25, \
         patch("app.ingestion.pipeline.register_document"), \
         patch("app.ingestion.pipeline.update_job"):

        run_crawl_job(
            job_id="job4",
            channel_id="chan1",
            base_url="https://example.com",
            include_paths=["/docs"],
            max_pages=50,
            max_depth=3,
            embedding_model=MagicMock(),
        )

    call_kwargs = MockChroma.from_documents.call_args.kwargs
    embedded_texts = [d.page_content for d in call_kwargs["documents"]]
    combined = "\n".join(embedded_texts)
    assert "Q1 | Q2" in combined
    assert "Revenue grew from 10 to 20." in combined
```

- [ ] **Step 2: Run the test**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ingestion_pipeline.py -v -k html_page_gets_table`
Expected: PASS (this exercises real `ParsedDocument.to_text_stream` logic from Task 6, only `caption_figure` and the crawl/Chroma/bm25/job-state boundaries are mocked)

- [ ] **Step 3: Run the full test suite one final time**

Run: `.venv\Scripts\python.exe -m pytest -v`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_ingestion_pipeline.py
git commit -m "test: verify scraped HTML pages get table+figure treatment via Docling"
```

---

### Task 13: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add a section describing the new ingestion package**

After the existing `- **`eval/`**` bullet in the Architecture section, add:

```markdown
- **`ingestion/`** — document parsing and web-crawl ingestion (Phase 5). `parser.py` (`parse_document(source, source_type)` via Docling — PDF/DOCX/HTML all go through one layout-aware pipeline: paragraphs, markdown tables, and figures; figures are captioned inline via `captioning.py`'s Groq vision call and folded into `ParsedDocument.to_text_stream()`); `captioning.py` (`caption_figure`, vision-capable Groq model per `settings.GROQ_VISION_MODEL`, never raises — returns `""` on failure); `crawler.py` (`crawl_site`, Scrapy same-domain crawl bounded by `CRAWL_MAX_PAGES`/`CRAWL_MAX_DEPTH`, optional `include_paths` prefix filter, runs in a subprocess since Scrapy's Twisted reactor can't share a process with uvicorn's asyncio loop); `pipeline.py` (`run_crawl_job`, the full crawl→parse→caption→chunk→embed flow, writing into the *same* Chroma collection + BM25 index as file uploads). Crawl job progress is tracked in Redis via `repository/crawl_jobs.py` (`create_job`/`update_job`/`get_job`), polled via `GET /crawl/{job_id}`.
- File uploads (`POST /upload`) now parse PDF/DOCX via Docling (`app.ingestion.parser.parse_document`) instead of PyMuPDF/python-docx directly — tables and figures are preserved, not just plain text.
- **Vector store is still Chroma in this phase.** Qdrant migration and porting this ingestion logic to `closeloop-backend` are explicitly deferred to future work (see `docs/superpowers/specs/2026-07-29-crawler-docling-ingestion-design.md`).
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document the ingestion package (Docling parsing + crawl pipeline)"
```

---

## Post-plan manual verification (not automated)

After all tasks are complete, do one live end-to-end smoke test before considering this done:

1. Start the server (`.venv\Scripts\python.exe -m uvicorn main:app`).
2. `POST /api/v1/rag-chatbot/upload` with a real PDF containing a table — confirm the response succeeds and the table content is retrievable via `POST /api/v1/rag-chatbot/chat`.
3. `POST /api/v1/rag-chatbot/crawl` with a small real site (e.g. a docs site with a handful of pages under one path prefix) and a low `max_pages` (e.g. 5) — poll `GET /api/v1/rag-chatbot/crawl/{job_id}` until `status: done`, then chat against that channel and confirm answers are grounded in the crawled content.
4. Confirm Scrapy's subprocess actually exits cleanly (no zombie process left running) after a crawl completes.
