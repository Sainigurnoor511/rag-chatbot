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
