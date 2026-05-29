from app.eval.retrieval_metrics import hit_at_k, reciprocal_rank, summarize


def test_hit_at_k():
    assert hit_at_k(["a", "b", "c"], {"c"}, k=3) == 1.0
    assert hit_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0
    assert hit_at_k(["a", "b"], {"z"}, k=2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_summarize_averages_per_query():
    per_query = [
        {"hit@k": 1.0, "mrr": 1.0},
        {"hit@k": 0.0, "mrr": 0.0},
        {"hit@k": 1.0, "mrr": 0.5},
    ]
    out = summarize(per_query)
    assert round(out["hit@k"], 4) == round(2 / 3, 4)
    assert round(out["mrr"], 4) == round(1.5 / 3, 4)


def test_summarize_empty():
    assert summarize([]) == {"hit@k": 0.0, "mrr": 0.0}
