from langchain.schema import Document


def _doc_key(doc: Document) -> str:
    """Identity for fusion: prefer chunk_id metadata, fall back to content."""
    return doc.metadata.get("chunk_id") or doc.page_content


def reciprocal_rank_fusion(result_lists: list[list[Document]], k: int = 60) -> list[Document]:
    """Fuse ranked Document lists via RRF: score = sum 1/(k + rank), rank starting at 1."""
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = _doc_key(doc)
            docs_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [docs_by_key[key] for key in ranked_keys]
