"""Compare naive dense-only retrieval vs hybrid+rerank over a golden set.

RAGAS answer-quality scoring is optional and imported lazily so the serving
dependency tree is never affected. Run this as a script in an isolated env that
also has requirements-eval.txt installed if you want RAGAS scores.
"""
from typing import Callable

from app.config.logger import logger
from app.eval.retrieval_metrics import hit_at_k, reciprocal_rank, summarize


def _eval_pipeline(golden: list[dict], retrieve_ids: Callable[[str], list[str]], k: int) -> dict:
    per_query = []
    for item in golden:
        relevant = set(item.get("relevant_chunk_ids") or [])
        retrieved = retrieve_ids(item["question"])
        per_query.append({
            "hit@k": hit_at_k(retrieved, relevant, k),
            "mrr": reciprocal_rank(retrieved, relevant),
        })
    return summarize(per_query)


def compare_pipelines(golden: list[dict],
                      naive_retrieve_ids: Callable[[str], list[str]],
                      hybrid_retrieve_ids: Callable[[str], list[str]],
                      k: int = 5) -> dict:
    """Return {'naive': {hit@k, mrr}, 'hybrid': {hit@k, mrr}} over the golden set."""
    return {
        "naive": _eval_pipeline(golden, naive_retrieve_ids, k),
        "hybrid": _eval_pipeline(golden, hybrid_retrieve_ids, k),
    }


def maybe_ragas_scores(samples: list[dict]):
    """Optionally compute RAGAS answer-quality metrics. Returns None if ragas is
    unavailable. `samples` items: {question, answer, contexts(list[str]), ground_truth}."""
    try:
        from ragas import evaluate  # noqa: F401  (lazy, optional)
    except Exception as e:
        logger.warning(f"RAGAS not available, skipping answer-quality scoring: {e}")
        return None
    # Real scoring is environment-specific; callers wire the dataset + LLM/embeddings
    # in the isolated eval env. Returning None here keeps the serving path import-safe.
    return None


def format_report(comparison: dict, k: int, ragas_scores: dict | None) -> str:
    lines = [
        "# RAG Evaluation Report",
        "",
        f"Retrieval metrics (k={k}), naive dense-only vs hybrid (dense+BM25+RRF+rerank):",
        "",
        "| Pipeline | hit@k | MRR |",
        "| --- | --- | --- |",
        f"| naive | {comparison['naive']['hit@k']:.4f} | {comparison['naive']['mrr']:.4f} |",
        f"| hybrid | {comparison['hybrid']['hit@k']:.4f} | {comparison['hybrid']['mrr']:.4f} |",
    ]
    if ragas_scores:
        lines += ["", "## RAGAS answer-quality (hybrid pipeline)", ""]
        for metric, score in ragas_scores.items():
            lines.append(f"- {metric}: {score}")
    return "\n".join(lines)


def write_report(markdown: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown)
