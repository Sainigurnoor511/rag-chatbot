from langchain.schema import Document
from app.retrieval.fusion import reciprocal_rank_fusion


def _doc(cid, text="x"):
    return Document(page_content=text, metadata={"chunk_id": cid})


def test_rrf_ranks_consensus_doc_first():
    list_a = [_doc("A"), _doc("B"), _doc("C")]
    list_b = [_doc("B"), _doc("D"), _doc("A")]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    keys = [d.metadata["chunk_id"] for d in fused]
    assert keys[0] == "B"
    assert sorted(keys) == ["A", "B", "C", "D"]


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_rrf_dedupes_same_chunk():
    fused = reciprocal_rank_fusion([[_doc("A")], [_doc("A")]], k=60)
    assert len(fused) == 1
    assert fused[0].metadata["chunk_id"] == "A"
