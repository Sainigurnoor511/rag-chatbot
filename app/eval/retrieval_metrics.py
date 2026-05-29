"""Pure retrieval-quality metrics — no heavy dependencies."""


def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if any relevant id appears in the top-k retrieved ids, else 0.0."""
    return 1.0 if set(retrieved_ids[:k]) & relevant_ids else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1/rank of the first relevant id (rank starting at 1), else 0.0."""
    for index, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (index + 1)
    return 0.0


def summarize(per_query: list[dict]) -> dict:
    """Average each metric across per-query results."""
    if not per_query:
        return {"hit@k": 0.0, "mrr": 0.0}
    n = len(per_query)
    return {
        "hit@k": sum(q["hit@k"] for q in per_query) / n,
        "mrr": sum(q["mrr"] for q in per_query) / n,
    }
