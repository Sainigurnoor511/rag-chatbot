import pytest

import app.controller.rag_controller as ctrl_mod
from app.controller.rag_controller import RAGController


class _FakeVectorstore:
    last_kwargs = None

    def __init__(self):
        pass

    @classmethod
    def from_documents(cls, **kwargs):
        cls.last_kwargs = kwargs
        return cls()


@pytest.fixture
def patched_controller(monkeypatch, tmp_path):
    # Avoid loading real embedding model / LLM.
    monkeypatch.setattr(
        ctrl_mod.RAGUtilities, "__init__", lambda self: None
    )
    monkeypatch.setattr(
        ctrl_mod.RAGUtilities, "get_embedding_model", lambda self: object()
    )
    # Point EMBEDDING_DIR at a temp dir and stub Chroma + text extraction.
    monkeypatch.setattr(ctrl_mod, "EMBEDDING_DIR", str(tmp_path))
    monkeypatch.setattr(ctrl_mod, "Chroma", _FakeVectorstore)
    monkeypatch.setattr(
        ctrl_mod.RAGService, "get_text", staticmethod(lambda p: "hello world. " * 100)
    )
    return RAGController()


def test_create_embeddings_uses_channel_collection(patched_controller, tmp_path):
    # The production code guards on os.path.isfile before extracting text, so
    # create the file on disk; RAGService.get_text is stubbed regardless.
    doc_path = tmp_path / "alpha.pdf"
    doc_path.write_text("placeholder")
    result = patched_controller.create_document_embeddings(
        channel_id="chan-1", file_path=str(doc_path)
    )
    assert result["doc_id"]
    kwargs = _FakeVectorstore.last_kwargs
    assert kwargs["collection_name"] == "rag_channel"
    assert kwargs["persist_directory"].endswith("chan-1")
    # Every chunk carries channel metadata.
    assert all(d.metadata["channel_id"] == "chan-1" for d in kwargs["documents"])
