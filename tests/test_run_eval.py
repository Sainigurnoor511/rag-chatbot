from app.eval import run_eval


def test_compare_pipelines_computes_metrics_for_both():
    golden = [
        {"question": "q1", "relevant_chunk_ids": ["c1"]},
        {"question": "q2", "relevant_chunk_ids": ["c2"]},
    ]
    def naive_fn(question):
        return {"q1": ["x", "c1"], "q2": ["y", "z"]}[question]

    def hybrid_fn(question):
        return {"q1": ["c1", "x"], "q2": ["c2", "y"]}[question]

    result = run_eval.compare_pipelines(golden, naive_fn, hybrid_fn, k=3)
    assert "naive" in result and "hybrid" in result
    assert result["hybrid"]["mrr"] == 1.0
    assert result["naive"]["mrr"] == 0.25
    assert result["hybrid"]["hit@k"] == 1.0
    assert result["naive"]["hit@k"] == 0.5


def test_format_report_contains_both_pipelines():
    result = {"naive": {"hit@k": 0.5, "mrr": 0.25}, "hybrid": {"hit@k": 1.0, "mrr": 1.0}}
    md = run_eval.format_report(result, k=3, ragas_scores=None)
    assert "naive" in md.lower()
    assert "hybrid" in md.lower()
    assert "hit@k" in md.lower() or "hit@3" in md.lower()


def test_format_report_includes_ragas_when_present():
    result = {"naive": {"hit@k": 0.5, "mrr": 0.25}, "hybrid": {"hit@k": 1.0, "mrr": 1.0}}
    md = run_eval.format_report(result, k=3, ragas_scores={"faithfulness": 0.9, "answer_relevancy": 0.8})
    assert "faithfulness" in md.lower()
    assert "0.9" in md
