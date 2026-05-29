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
