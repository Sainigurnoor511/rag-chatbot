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
