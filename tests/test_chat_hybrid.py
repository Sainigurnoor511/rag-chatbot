import pytest
from langchain.schema import Document
import app.controller.rag_controller as ctrl_mod
from app.controller.rag_controller import RAGController


class _FakeVectorstore:
    pass


class _FakeUtils:
    def __init__(self):
        pass

    def load_embeddings(self, channel_id):
        return _FakeVectorstore()

    def contextualize_question(self, message, history_messages):
        return message

    def answer(self, user_input, context, history_messages, filename):
        return f"answer using context[{len(context)}]"


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(ctrl_mod.RAGUtilities, "__init__", lambda self: None)
    monkeypatch.setattr(ctrl_mod.RAGUtilities, "get_embedding_model", lambda self: object())
    controller = RAGController()
    monkeypatch.setattr(ctrl_mod, "RAGUtilities", _FakeUtils)
    monkeypatch.setattr(ctrl_mod, "load_session_from_redis", lambda cid: None)
    saved = {}
    monkeypatch.setattr(ctrl_mod, "save_session_to_redis", lambda cid, data: saved.update({"cid": cid}))
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


def test_chat_no_relevant_docs_returns_grounded_fallback(patched, monkeypatch):
    controller, saved = patched
    monkeypatch.setattr(ctrl_mod.HybridRetriever, "retrieve",
                        lambda self, query, filename=None: [])
    resp = controller.chat_with_document({"channel_id": "chan-1", "message": "hello", "filename": None})
    assert resp["success"] is True
    out = resp["data"]["bot_output"].lower()
    assert ("couldn't find" in out) or ("no relevant" in out)
    assert "answer using context" not in out   # the LLM answer() path was skipped
    assert saved["cid"] == "chan-1"


def test_chat_returns_cached_answer_when_enabled(patched, monkeypatch):
    controller, saved = patched
    monkeypatch.setattr(ctrl_mod.settings, "ENABLE_QUERY_CACHE", True)
    monkeypatch.setattr(ctrl_mod.query_cache, "get_cached",
                        lambda channel_id, message, filename: "CACHED ANSWER")
    resp = controller.chat_with_document({"channel_id": "chan-1", "message": "hi", "filename": None})
    assert resp["success"] is True
    assert resp["data"]["bot_output"] == "CACHED ANSWER"
    assert "(cached)" in resp["message"]
