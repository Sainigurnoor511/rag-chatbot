# Production RAG — Phase 2: Hybrid Retrieval + Cross-Encoder Rerank — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the naive single-document dense retrieval in chat with per-channel hybrid retrieval — dense (Chroma) + sparse (local BM25) fused by Reciprocal Rank Fusion, then reranked by a local cross-encoder — and rewrite `chat_with_document` to use it, removing the redundant manual similarity search and the filename/channel_id confusion left from Phase 1.

**Architecture:** New `app/retrieval/` modules: `fusion.py` (pure RRF), `bm25_index.py` (per-channel BM25 corpus persisted alongside the Chroma dir), `reranker.py` (lazy/class-cached `sentence-transformers` CrossEncoder), `hybrid_retriever.py` (orchestrates dense+sparse→RRF→rerank). Ingestion additionally appends chunks to the channel BM25 corpus. `chat_with_document` becomes an explicit flow: load history → contextualize query (LLM) → hybrid retrieve → assemble context → answer (LLM) → save history. The brittle `create_retrieval_chain`/`RunnableWithMessageHistory` machinery is dropped in favor of testable explicit steps.

**Tech Stack:** `rank-bm25`, `sentence-transformers` (CrossEncoder `BAAI/bge-reranker-base`), LangChain (`Document`, prompt templates, `ChatGroq`), Chroma, pytest.

**Environment note for implementers:** the venv is uv-managed; use `.venv\Scripts\python.exe` for all python/pytest (PowerShell call: `& ".\.venv\Scripts\python.exe" -m pytest ...`), run from project root (`.env` supplies GROQ vars). Branch `feature/production-rag`. End commits with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Tests must NEVER download the reranker model — always mock it.

---

### Task 1: Phase 2 settings + dependencies

**Files:**
- Modify: `requirements.txt`
- Modify: `app/config/settings.py`
- Test: `tests/test_settings_phase2.py`

- [ ] **Step 1: Add dependencies to requirements.txt**

Append (matching the file's existing UTF-16 encoding — append, don't rewrite):
```
rank-bm25==0.2.2
sentence-transformers==3.3.1
```

- [ ] **Step 2: Install**

Run: `& ".\.venv\Scripts\python.exe" -m pip install rank-bm25==0.2.2 sentence-transformers==3.3.1`
Expected: success (torch is already present). This may take a few minutes.

- [ ] **Step 3: Write the failing test**

Create `tests/test_settings_phase2.py`:
```python
from app.config.settings import settings


def test_phase2_retrieval_settings_defaults():
    assert settings.DENSE_TOP_K == 20
    assert settings.BM25_TOP_K == 20
    assert settings.RRF_K == 60
    assert settings.RERANK_TOP_N == 5
    assert settings.RERANKER_MODEL == "BAAI/bge-reranker-base"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_settings_phase2.py -v`
Expected: FAIL with `AttributeError: ... 'DENSE_TOP_K'`.

- [ ] **Step 5: Add the settings fields**

In `app/config/settings.py`, after the `CHANNEL_TTL_SECONDS` line, add:
```python
    # Hybrid retrieval (Phase 2)
    DENSE_TOP_K: int = 20
    BM25_TOP_K: int = 20
    RRF_K: int = 60
    RERANK_TOP_N: int = 5
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
```

- [ ] **Step 6: Run test to verify it passes**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_settings_phase2.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt app/config/settings.py tests/test_settings_phase2.py
git commit -m "feat: add hybrid-retrieval settings and rank-bm25/sentence-transformers deps"
```

---

### Task 2: Reciprocal Rank Fusion (pure function)

**Files:**
- Create: `app/retrieval/fusion.py`
- Test: `tests/test_fusion.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fusion.py`:
```python
from langchain.schema import Document
from app.retrieval.fusion import reciprocal_rank_fusion


def _doc(cid, text="x"):
    return Document(page_content=text, metadata={"chunk_id": cid})


def test_rrf_ranks_consensus_doc_first():
    # B appears high in both lists -> should win.
    list_a = [_doc("A"), _doc("B"), _doc("C")]
    list_b = [_doc("B"), _doc("D"), _doc("A")]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    keys = [d.metadata["chunk_id"] for d in fused]
    assert keys[0] == "B"
    # all unique docs present, no duplicates
    assert sorted(keys) == ["A", "B", "C", "D"]


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_rrf_dedupes_same_chunk():
    fused = reciprocal_rank_fusion([[_doc("A")], [_doc("A")]], k=60)
    assert len(fused) == 1
    assert fused[0].metadata["chunk_id"] == "A"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_fusion.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'app.retrieval.fusion'`.

- [ ] **Step 3: Implement**

Create `app/retrieval/fusion.py`:
```python
from langchain.schema import Document


def _doc_key(doc: Document) -> str:
    """Identity for fusion: prefer chunk_id metadata, fall back to content."""
    return doc.metadata.get("chunk_id") or doc.page_content


def reciprocal_rank_fusion(result_lists: list[list[Document]], k: int = 60) -> list[Document]:
    """Fuse ranked Document lists via RRF: score = sum 1/(k + rank), rank starting at 1."""
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = _doc_key(doc)
            docs_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [docs_by_key[key] for key in ranked_keys]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_fusion.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/fusion.py tests/test_fusion.py
git commit -m "feat: add reciprocal rank fusion for hybrid retrieval"
```

---

### Task 3: Per-channel BM25 index

**Files:**
- Create: `app/retrieval/bm25_index.py`
- Test: `tests/test_bm25_index.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_bm25_index.py`:
```python
import app.retrieval.bm25_index as bm25
from langchain.schema import Document


def _doc(text, cid, source="a.pdf"):
    return Document(page_content=text, metadata={"chunk_id": cid, "source": source})


def test_add_and_search_returns_relevant_first(monkeypatch, tmp_path):
    monkeypatch.setattr(bm25.settings, "EMBEDDING_DIR", str(tmp_path))
    docs = [
        _doc("the cat sat on the mat", "c1"),
        _doc("financial report quarterly revenue", "c2"),
        _doc("dogs and cats are pets", "c3"),
    ]
    bm25.add_documents("chan-1", docs)
    results = bm25.search("chan-1", "quarterly revenue", top_k=2)
    assert results
    assert results[0].metadata["chunk_id"] == "c2"


def test_search_missing_channel_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(bm25.settings, "EMBEDDING_DIR", str(tmp_path))
    assert bm25.search("nope", "anything", top_k=5) == []


def test_add_documents_accumulates(monkeypatch, tmp_path):
    monkeypatch.setattr(bm25.settings, "EMBEDDING_DIR", str(tmp_path))
    bm25.add_documents("chan-1", [_doc("alpha text", "c1")])
    bm25.add_documents("chan-1", [_doc("beta text", "c2")])
    results = bm25.search("chan-1", "beta", top_k=5)
    keys = {d.metadata["chunk_id"] for d in results}
    assert "c2" in keys and len(keys) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_bm25_index.py -v`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `app/retrieval/bm25_index.py`:
```python
import os
import re
import pickle

from langchain.schema import Document
from rank_bm25 import BM25Okapi

from app.config.settings import settings
from app.config.logger import logger

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def bm25_path(channel_id: str) -> str:
    return os.path.join(settings.EMBEDDING_DIR, channel_id, "bm25.pkl")


def _load_corpus(channel_id: str) -> dict | None:
    path = bm25_path(channel_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load BM25 corpus for {channel_id}: {e}")
        return None


def add_documents(channel_id: str, docs: list[Document]) -> None:
    """Append documents to the channel's BM25 corpus and persist it."""
    corpus = _load_corpus(channel_id) or {"texts": [], "metadatas": []}
    for d in docs:
        corpus["texts"].append(d.page_content)
        corpus["metadatas"].append(d.metadata)
    path = bm25_path(channel_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(corpus, f)


def search(channel_id: str, query: str, top_k: int) -> list[Document]:
    """Return the top_k BM25 matches for the query as Documents (empty if no corpus)."""
    corpus = _load_corpus(channel_id)
    if not corpus or not corpus["texts"]:
        return []
    tokenized_corpus = [_tokenize(t) for t in corpus["texts"]]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        Document(page_content=corpus["texts"][i], metadata=corpus["metadatas"][i])
        for i in ranked
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_bm25_index.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/bm25_index.py tests/test_bm25_index.py
git commit -m "feat: add per-channel BM25 sparse index"
```

---

### Task 4: Cross-encoder reranker

**Files:**
- Create: `app/retrieval/reranker.py`
- Test: `tests/test_reranker.py`

- [ ] **Step 1: Write the failing test** (mocks the model — never downloads)

Create `tests/test_reranker.py`:
```python
from langchain.schema import Document
import app.retrieval.reranker as rr


def _doc(text, cid):
    return Document(page_content=text, metadata={"chunk_id": cid})


class _FakeModel:
    def predict(self, pairs):
        # score = length of the candidate text (second element) — deterministic
        return [float(len(c)) for _q, c in pairs]


def test_rerank_orders_by_score_and_truncates(monkeypatch):
    monkeypatch.setattr(rr.CrossEncoderReranker, "_get_model",
                        classmethod(lambda cls: _FakeModel()))
    docs = [_doc("short", "c1"), _doc("a much longer candidate", "c2"), _doc("mid len", "c3")]
    out = rr.CrossEncoderReranker().rerank("q", docs, top_n=2)
    assert [d.metadata["chunk_id"] for d in out] == ["c2", "c3"]


def test_rerank_empty_returns_empty(monkeypatch):
    monkeypatch.setattr(rr.CrossEncoderReranker, "_get_model",
                        classmethod(lambda cls: _FakeModel()))
    assert rr.CrossEncoderReranker().rerank("q", [], top_n=5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_reranker.py -v`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `app/retrieval/reranker.py`:
```python
from langchain.schema import Document

from app.config.settings import settings
from app.config.logger import logger


class CrossEncoderReranker:
    """Local cross-encoder reranker. Model is loaded once and cached on the class."""

    _model = None

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker model: {settings.RERANKER_MODEL}")
            cls._model = CrossEncoder(settings.RERANKER_MODEL)
        return cls._model

    def rerank(self, query: str, documents: list[Document], top_n: int) -> list[Document]:
        """Score (query, doc) pairs and return the top_n documents by descending score."""
        if not documents:
            return []
        model = self._get_model()
        pairs = [(query, d.page_content) for d in documents]
        scores = model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked[:top_n]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_reranker.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/reranker.py tests/test_reranker.py
git commit -m "feat: add cross-encoder reranker (lazy/class-cached)"
```

---

### Task 5: Hybrid retriever (dense + BM25 → RRF → rerank)

**Files:**
- Create: `app/retrieval/hybrid_retriever.py`
- Test: `tests/test_hybrid_retriever.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hybrid_retriever.py`:
```python
from langchain.schema import Document
import app.retrieval.hybrid_retriever as hr


def _doc(cid, text="x", source="a.pdf"):
    return Document(page_content=text, metadata={"chunk_id": cid, "source": source})


class _FakeVectorstore:
    def __init__(self, docs):
        self._docs = docs
        self.last_filter = "unset"

    def similarity_search(self, query, k, filter=None):
        self.last_filter = filter
        return self._docs[:k]


class _FakeReranker:
    def rerank(self, query, documents, top_n):
        return documents[:top_n]


def test_retrieve_fuses_dense_and_sparse_then_reranks(monkeypatch):
    dense = [_doc("d1"), _doc("shared")]
    sparse = [_doc("shared"), _doc("s1")]
    monkeypatch.setattr(hr.bm25_index, "search", lambda channel_id, query, top_k: sparse)
    vs = _FakeVectorstore(dense)
    retr = hr.HybridRetriever("chan-1", vs, reranker=_FakeReranker())
    out = retr.retrieve("q")
    keys = [d.metadata["chunk_id"] for d in out]
    # "shared" appears in both -> ranked first by RRF; no duplicates
    assert keys[0] == "shared"
    assert len(keys) == len(set(keys))


def test_retrieve_passes_source_filter_when_filename_given(monkeypatch):
    monkeypatch.setattr(hr.bm25_index, "search", lambda channel_id, query, top_k: [])
    vs = _FakeVectorstore([_doc("d1")])
    retr = hr.HybridRetriever("chan-1", vs, reranker=_FakeReranker())
    retr.retrieve("q", filename="a.pdf")
    assert vs.last_filter == {"source": "a.pdf"}


def test_retrieve_empty_when_nothing_found(monkeypatch):
    monkeypatch.setattr(hr.bm25_index, "search", lambda channel_id, query, top_k: [])
    vs = _FakeVectorstore([])
    retr = hr.HybridRetriever("chan-1", vs, reranker=_FakeReranker())
    assert retr.retrieve("q") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_hybrid_retriever.py -v`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `app/retrieval/hybrid_retriever.py`:
```python
from langchain.schema import Document

from app.config.settings import settings
from app.config.logger import logger
from app.retrieval import bm25_index
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import CrossEncoderReranker


class HybridRetriever:
    """Per-channel hybrid retrieval: dense (Chroma) + sparse (BM25) -> RRF -> rerank."""

    def __init__(self, channel_id: str, vectorstore, reranker=None):
        self.channel_id = channel_id
        self.vectorstore = vectorstore
        self.reranker = reranker or CrossEncoderReranker()

    def retrieve(self, query: str, filename: str | None = None) -> list[Document]:
        source_filter = {"source": filename} if filename else None

        try:
            dense = self.vectorstore.similarity_search(
                query, k=settings.DENSE_TOP_K, filter=source_filter
            )
        except Exception as e:
            logger.error(f"Dense search failed for channel {self.channel_id}: {e}")
            dense = []

        sparse = bm25_index.search(self.channel_id, query, settings.BM25_TOP_K)
        if filename:
            sparse = [d for d in sparse if d.metadata.get("source") == filename]

        fused = reciprocal_rank_fusion([dense, sparse], k=settings.RRF_K)
        if not fused:
            return []

        reranked = self.reranker.rerank(query, fused, settings.RERANK_TOP_N)
        logger.info(
            f"Hybrid retrieve channel={self.channel_id}: "
            f"dense={len(dense)} sparse={len(sparse)} fused={len(fused)} kept={len(reranked)}"
        )
        return reranked
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_hybrid_retriever.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/hybrid_retriever.py tests/test_hybrid_retriever.py
git commit -m "feat: add hybrid retriever orchestrating dense+bm25+rrf+rerank"
```

---

### Task 6: Build the BM25 index during ingestion

**Files:**
- Modify: `app/controller/rag_controller.py` (the `create_document_embeddings` method)
- Test: `tests/test_ingestion_bm25.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestion_bm25.py`:
```python
import pytest
import app.controller.rag_controller as ctrl_mod
from app.controller.rag_controller import RAGController


class _FakeVectorstore:
    @classmethod
    def from_documents(cls, **kwargs):
        return cls()


@pytest.fixture
def patched(monkeypatch, tmp_path):
    monkeypatch.setattr(ctrl_mod.RAGUtilities, "__init__", lambda self: None)
    monkeypatch.setattr(ctrl_mod.RAGUtilities, "get_embedding_model", lambda self: object())
    monkeypatch.setattr(ctrl_mod, "EMBEDDING_DIR", str(tmp_path))
    monkeypatch.setattr(ctrl_mod, "Chroma", _FakeVectorstore)
    monkeypatch.setattr(ctrl_mod.RAGService, "get_text", staticmethod(lambda p: "hello world. " * 100))
    calls = {}
    monkeypatch.setattr(ctrl_mod.bm25_index, "add_documents",
                        lambda channel_id, docs: calls.update({"channel_id": channel_id, "n": len(docs)}))
    return RAGController(), calls


def test_ingestion_also_populates_bm25(patched, tmp_path):
    controller, calls = patched
    doc_path = tmp_path / "alpha.pdf"
    doc_path.write_text("placeholder")
    result = controller.create_document_embeddings(channel_id="chan-1", file_path=str(doc_path))
    assert result["chunks"] > 0
    assert calls["channel_id"] == "chan-1"
    assert calls["n"] == result["chunks"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_ingestion_bm25.py -v`
Expected: FAIL — `ctrl_mod.bm25_index` does not exist (not imported) / `add_documents` not called.

- [ ] **Step 3: Implement**

In `app/controller/rag_controller.py`, add to the import block (after `from app.retrieval.chunking import chunk_text`):
```python
from app.retrieval import bm25_index
```

Then in `create_document_embeddings`, immediately AFTER the `Chroma.from_documents(...)` call and BEFORE the `logger.info(f"Embedded {len(docs)} chunks ...")` line, add:
```python
            bm25_index.add_documents(channel_id, docs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_ingestion_bm25.py -v`
Expected: PASS. Also run the full suite and confirm the existing `tests/test_ingestion.py` still passes (its `_FakeVectorstore` has no `add_documents` interaction, and `bm25_index.add_documents` will run against the real module writing into the test's `tmp_path` EMBEDDING_DIR — confirm it doesn't error; if `tests/test_ingestion.py` fails because `add_documents` writes to a real path, that test already monkeypatches `EMBEDDING_DIR` to tmp_path so it is safe).

Run: `& ".\.venv\Scripts\python.exe" -m pytest -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/controller/rag_controller.py tests/test_ingestion_bm25.py
git commit -m "feat: populate per-channel BM25 index during ingestion"
```

---

### Task 7: Add `contextualize_question` and `answer` to RAGUtilities

**Files:**
- Modify: `app/utilities/rag_utilities.py` (add two methods; do not remove existing ones)
- Test: `tests/test_rag_utilities_chat_helpers.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rag_utilities_chat_helpers.py`:
```python
import app.utilities.rag_utilities as util_mod
from langchain_core.messages import HumanMessage, AIMessage


class _FakeLLM:
    """Stand-in for ChatGroq: records the last prompt-value and returns a canned message."""
    def __init__(self, content="REWRITTEN"):
        self.content = content
        self.calls = []

    def invoke(self, prompt_value):
        self.calls.append(prompt_value)
        class _Resp:
            pass
        r = _Resp()
        r.content = self.content
        return r


def _utils_with_llm(monkeypatch, llm):
    monkeypatch.setattr(util_mod.RAGUtilities, "__init__", lambda self: None)
    u = util_mod.RAGUtilities()
    u.llm = llm
    return u


def test_contextualize_returns_message_unchanged_without_history(monkeypatch):
    llm = _FakeLLM()
    u = _utils_with_llm(monkeypatch, llm)
    out = u.contextualize_question("what is it?", [])
    assert out == "what is it?"
    assert llm.calls == []  # no LLM call when there's no history


def test_contextualize_uses_llm_with_history(monkeypatch):
    llm = _FakeLLM(content="standalone question")
    u = _utils_with_llm(monkeypatch, llm)
    history = [HumanMessage(content="Tell me about cats"), AIMessage(content="Cats are pets")]
    out = u.contextualize_question("and dogs?", history)
    assert out == "standalone question"
    assert len(llm.calls) == 1


def test_answer_invokes_llm_and_returns_content(monkeypatch):
    llm = _FakeLLM(content="the answer")
    u = _utils_with_llm(monkeypatch, llm)
    out = u.answer("question", context="some context", history_messages=[], filename="a.pdf")
    assert out == "the answer"
    assert len(llm.calls) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_rag_utilities_chat_helpers.py -v`
Expected: FAIL — `RAGUtilities` has no `contextualize_question` / `answer`.

- [ ] **Step 3: Implement**

In `app/utilities/rag_utilities.py`, add these two methods to the `RAGUtilities` class (place them after `create_qa_prompt`). They reuse the existing `create_qa_prompt(filename)` (which already has a `{context}` placeholder, a `chat_history` MessagesPlaceholder, and a `{input}` human turn). Add a sibling contextualize prompt builder:
```python
    def _contextualize_prompt(self) -> ChatPromptTemplate:
        system_prompt = (
            "Given a chat history and the latest user question which might reference context "
            "in the chat history, reformulate it into a standalone question understandable "
            "without the chat history. Do NOT answer it; only reformulate it if needed, "
            "otherwise return it as is."
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def contextualize_question(self, message: str, history_messages: list) -> str:
        """Rewrite a follow-up into a standalone question using chat history (LLM)."""
        if not history_messages:
            return message
        try:
            prompt = self._contextualize_prompt()
            value = prompt.invoke({"chat_history": history_messages, "input": message})
            return self.llm.invoke(value).content
        except Exception as e:
            logger.error(f"contextualize_question failed, using raw message: {e}")
            return message

    def answer(self, user_input: str, context: str, history_messages: list, filename: str) -> str:
        """Generate a grounded answer from context + chat history (LLM)."""
        prompt = self.create_qa_prompt(filename)
        value = prompt.invoke(
            {"context": context, "chat_history": history_messages, "input": user_input}
        )
        return self.llm.invoke(value).content
```

Confirm `ChatPromptTemplate` and `MessagesPlaceholder` are already imported at the top of the file (they are — used by the existing prompt builders).

- [ ] **Step 4: Run test to verify it passes**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_rag_utilities_chat_helpers.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add app/utilities/rag_utilities.py tests/test_rag_utilities_chat_helpers.py
git commit -m "feat: add contextualize_question and answer helpers to RAGUtilities"
```

---

### Task 8: Rewrite `chat_with_document` to the hybrid flow

**Files:**
- Modify: `app/controller/rag_controller.py` (replace the `chat_with_document` method)
- Test: `tests/test_chat_hybrid.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chat_hybrid.py`:
```python
import pytest
from langchain.schema import Document
import app.controller.rag_controller as ctrl_mod
from app.controller.rag_controller import RAGController


class _FakeVectorstore:
    pass


class _FakeUtils:
    """Replaces RAGUtilities() inside chat_with_document."""
    def __init__(self):
        pass

    def load_embeddings(self, channel_id):
        return _FakeVectorstore()

    def contextualize_question(self, message, history_messages):
        return message  # passthrough

    def answer(self, user_input, context, history_messages, filename):
        return f"answer using context[{len(context)}]"


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(ctrl_mod.RAGUtilities, "__init__", lambda self: None)
    monkeypatch.setattr(ctrl_mod.RAGUtilities, "get_embedding_model", lambda self: object())
    controller = RAGController()
    # Replace RAGUtilities used inside chat with the fake.
    monkeypatch.setattr(ctrl_mod, "RAGUtilities", _FakeUtils)
    # No redis.
    monkeypatch.setattr(ctrl_mod, "load_session_from_redis", lambda cid: None)
    saved = {}
    monkeypatch.setattr(ctrl_mod, "save_session_to_redis", lambda cid, data: saved.update({"cid": cid}))
    # Hybrid retriever returns fixed docs.
    monkeypatch.setattr(
        ctrl_mod.HybridRetriever, "retrieve",
        lambda self, query, filename=None: [Document(page_content="ctx", metadata={"chunk_id": "c1"})],
    )
    return controller, saved


def test_chat_happy_path(patched):
    controller, saved = patched
    resp = controller.chat_with_document(
        {"channel_id": "chan-1", "message": "hello", "filename": None}
    )
    assert resp["success"] is True
    assert "answer using context" in resp["data"]["bot_output"]
    assert saved["cid"] == "chan-1"


def test_chat_missing_fields_returns_400(patched):
    controller, _ = patched
    resp = controller.chat_with_document({"channel_id": "", "message": "", "filename": None})
    assert resp["success"] is False
    assert resp["error"]["code"] == 400


def test_chat_no_embeddings_returns_404(patched, monkeypatch):
    controller, _ = patched
    monkeypatch.setattr(_FakeUtils, "load_embeddings", lambda self, cid: None)
    resp = controller.chat_with_document(
        {"channel_id": "chan-1", "message": "hi", "filename": None}
    )
    assert resp["success"] is False
    assert resp["error"]["code"] == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_chat_hybrid.py -v`
Expected: FAIL — `HybridRetriever` not imported in controller, and the old `chat_with_document` does not match (uses `create_retriever`, manual similarity search, etc.).

- [ ] **Step 3: Implement**

In `app/controller/rag_controller.py`:

(a) Add to the import block:
```python
from app.retrieval.hybrid_retriever import HybridRetriever
```

(b) The following imports are no longer needed by the rewritten method but may still be used elsewhere — leave them only if used; otherwise the code-quality pass will remove them. Do NOT remove them in this step.

(c) Replace the ENTIRE `chat_with_document` method with:
```python
    @timer
    def chat_with_document(self, request: dict):
        """Hybrid RAG chat: contextualize -> hybrid retrieve -> answer, with Redis history."""
        try:
            channel_id = request.get("channel_id")
            message = request.get("message")
            filename = request.get("filename")  # optional: restricts to one doc in the channel

            if not channel_id or not message:
                logger.warning("Invalid request payload")
                return {
                    "success": False,
                    "message": "Invalid request payload",
                    "data": {},
                    "error": {"code": 400,
                              "message": "Missing required fields: channel_id or message"},
                }

            logger.info(f"Processing chat for channel: {channel_id}")
            user_input = message.strip()

            utils = RAGUtilities()
            vectorstore = utils.load_embeddings(channel_id)
            if vectorstore is None:
                logger.error(f"No embeddings for channel {channel_id}")
                return {
                    "success": False,
                    "message": "No documents found for this channel",
                    "data": {},
                    "error": {"code": 404,
                              "message": "Please upload a document first to generate embeddings"},
                }

            session_data = load_session_from_redis(channel_id)
            chat_history = (
                session_data.get(channel_id, ChatMessageHistory(messages=[]))
                if session_data else ChatMessageHistory(messages=[])
            )

            standalone_query = utils.contextualize_question(user_input, chat_history.messages)

            retriever = HybridRetriever(channel_id, vectorstore)
            docs = retriever.retrieve(standalone_query, filename=filename)
            context = "\n\n".join(d.page_content for d in docs)

            output = utils.answer(user_input, context, chat_history.messages, filename or channel_id)

            chat_history.messages.append(HumanMessage(content=user_input))
            chat_history.messages.append(AIMessage(content=output))
            save_session_to_redis(channel_id, {channel_id: chat_history})

            logger.info("Chat response generated successfully.")
            return {
                "success": True,
                "message": "Response generated successfully",
                "data": {"user_input": user_input, "bot_output": output},
                "error": None,
            }

        except Exception as e:
            logger.error(f"Error in chat_with_document: {str(e)}")
            return {
                "success": False,
                "message": "Internal server error during chat processing",
                "data": {},
                "error": {"code": 500, "message": str(e)},
            }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_chat_hybrid.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full suite**

Run: `& ".\.venv\Scripts\python.exe" -m pytest -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add app/controller/rag_controller.py tests/test_chat_hybrid.py
git commit -m "feat: rewrite chat_with_document to per-channel hybrid retrieval flow"
```

---

### Task 9: Update the /chat route to drop required filename

**Files:**
- Modify: `app/routes/rag_routes.py` (the `ChatRequest` model)
- Test: `tests/test_chat_route_optional_filename.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chat_route_optional_filename.py`:
```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import app.routes.rag_routes as routes_mod


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(routes_mod.RAGController, "__init__", lambda self: None)
    monkeypatch.setattr(
        routes_mod.RAGController, "chat_with_document",
        lambda self, request: {"success": True, "message": "ok",
                               "data": {"user_input": request.get("message"),
                                        "bot_output": "hi", "filename_seen": request.get("filename")},
                               "error": None},
    )
    app = FastAPI()
    app.include_router(routes_mod.router)
    return TestClient(app)


def test_chat_works_without_filename(client):
    resp = client.post("/chat", json={"channel_id": "chan-1", "message": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["filename_seen"] is None


def test_chat_accepts_optional_filename(client):
    resp = client.post("/chat", json={"channel_id": "chan-1", "message": "hello", "filename": "a.pdf"})
    assert resp.status_code == 200
    assert resp.json()["data"]["filename_seen"] == "a.pdf"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_chat_route_optional_filename.py -v`
Expected: FAIL on `test_chat_works_without_filename` with 422 (filename currently required).

- [ ] **Step 3: Implement**

In `app/routes/rag_routes.py`, change the `ChatRequest` model from:
```python
class ChatRequest(BaseModel):
    channel_id: str
    message: str
    filename: str
    # file_path: str
```
to:
```python
class ChatRequest(BaseModel):
    channel_id: str
    message: str
    filename: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& ".\.venv\Scripts\python.exe" -m pytest tests/test_chat_route_optional_filename.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full suite**

Run: `& ".\.venv\Scripts\python.exe" -m pytest -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add app/routes/rag_routes.py tests/test_chat_route_optional_filename.py
git commit -m "feat: make filename optional in chat request (channel-scoped retrieval)"
```

---

## Self-Review

**Spec coverage (Phase 2 portion of the spec):**
- Hybrid retrieval dense+BM25 → Tasks 3, 5 ✓
- RRF fusion → Task 2 ✓
- Cross-encoder rerank (`bge-reranker-base`, lazy/cached) → Task 4 ✓
- BM25 built per channel during ingestion → Tasks 3, 6 ✓
- Chat rewrite: contextualize → hybrid retrieve → answer; remove redundant manual similarity search and the old chain → Tasks 7, 8 ✓
- `filename` becomes optional metadata filter → Tasks 5 (filter), 8 (passthrough), 9 (request model) ✓
- Config knobs (DENSE_TOP_K, BM25_TOP_K, RRF_K, RERANK_TOP_N, RERANKER_MODEL) → Task 1 ✓

**Deferred (intentional):** idempotent re-upload replace (still appends; acceptable — the BM25 corpus also accumulates), Redis query-result caching and per-channel in-process LRU (Phase 3), on-disk expired-channel sweep (Phase 3), eval (Phase 4).

**Placeholder scan:** No TBD/TODO; every code step is complete. ✓

**Type consistency:** `HybridRetriever(channel_id, vectorstore, reranker=None).retrieve(query, filename=None)` used identically in Tasks 5 and 8. `bm25_index.add_documents(channel_id, docs)` / `search(channel_id, query, top_k)` consistent across Tasks 3, 5, 6. `CrossEncoderReranker().rerank(query, documents, top_n)` consistent across Tasks 4, 5. `RAGUtilities.contextualize_question(message, history_messages)` / `answer(user_input, context, history_messages, filename)` / `load_embeddings(channel_id)` consistent across Tasks 7, 8. ✓

**Known risk flagged for implementers:** the chat rewrite (Task 8) calls `RAGUtilities()` which, with the real (non-test) `__init__`, loads the embedding model + Groq LLM. Tests monkeypatch `RAGUtilities`/`__init__`, so they don't. End-to-end runtime requires the local reranker model to be downloadable on first chat — acceptable for this phase (a model-warmup step can be added in Phase 3).
