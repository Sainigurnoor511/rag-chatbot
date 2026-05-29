import json
from langchain.schema import Document
from app.eval import golden_set


class _FakeLLM:
    def invoke(self, prompt_value):
        class _R: ...
        r = _R()
        r.content = "What does this chunk say?"
        return r


def _chunk(cid, text):
    return Document(page_content=text, metadata={"chunk_id": cid, "source": "a.pdf"})


def test_generate_builds_items_with_relevant_chunk_id():
    chunks = [_chunk("c1", "The capital of France is Paris."),
              _chunk("c2", "Water boils at 100 degrees Celsius.")]
    items = golden_set.generate_golden_set(chunks, llm=_FakeLLM(), max_questions=2)
    assert len(items) == 2
    for it in items:
        assert it["question"] == "What does this chunk say?"
        assert it["relevant_chunk_ids"]
        assert it["relevant_chunk_ids"][0] in {"c1", "c2"}
        assert it["ground_truth_context"]


def test_generate_respects_max_questions():
    chunks = [_chunk(f"c{i}", f"text {i}") for i in range(10)]
    items = golden_set.generate_golden_set(chunks, llm=_FakeLLM(), max_questions=3)
    assert len(items) == 3


def test_save_and_load_roundtrip(tmp_path):
    items = [{"question": "q", "relevant_chunk_ids": ["c1"], "ground_truth_context": "ctx"}]
    path = tmp_path / "golden.json"
    golden_set.save_golden_set(items, str(path))
    loaded = golden_set.load_golden_set(str(path))
    assert loaded == items
