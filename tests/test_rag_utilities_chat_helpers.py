import app.utilities.rag_utilities as util_mod
from langchain_core.messages import HumanMessage, AIMessage


class _FakeLLM:
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
    assert llm.calls == []


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
