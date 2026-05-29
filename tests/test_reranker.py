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
