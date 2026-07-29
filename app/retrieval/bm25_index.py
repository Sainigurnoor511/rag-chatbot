import os
import re
import pickle

from langchain.schema import Document
from rank_bm25 import BM25Okapi

from app.config.settings import settings
from app.config.logger import logger

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Per-channel cache of the built BM25Okapi index, keyed by channel_id.
# Avoids re-tokenizing the whole corpus and rebuilding the index on every
# chat request (search() was doing this from scratch each call).
_BM25_INDEX_CACHE: dict[str, BM25Okapi] = {}
_BM25_CORPUS_CACHE: dict[str, dict] = {}


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def bm25_path(channel_id: str) -> str:
    return os.path.join(settings.EMBEDDING_DIR, channel_id, "bm25.pkl")


def _load_corpus(channel_id: str) -> dict | None:
    path = bm25_path(channel_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load BM25 corpus for {channel_id}: {e}")
        return None


def add_documents(channel_id: str, docs: list[Document]) -> None:
    """Append documents to the channel's BM25 corpus and persist it."""
    corpus = _load_corpus(channel_id) or {"texts": [], "metadatas": []}
    for d in docs:
        corpus["texts"].append(d.page_content)
        corpus["metadatas"].append(d.metadata)
    path = bm25_path(channel_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(corpus, f)

    # Invalidate the cached index so the next search() picks up the new docs.
    _BM25_INDEX_CACHE.pop(channel_id, None)
    _BM25_CORPUS_CACHE.pop(channel_id, None)


def _get_index(channel_id: str) -> tuple[BM25Okapi, dict] | tuple[None, None]:
    """Return the cached (index, corpus) for a channel, building it once if needed."""
    if channel_id in _BM25_INDEX_CACHE:
        return _BM25_INDEX_CACHE[channel_id], _BM25_CORPUS_CACHE[channel_id]

    corpus = _load_corpus(channel_id)
    if not corpus or not corpus["texts"]:
        return None, None

    tokenized_corpus = [_tokenize(t) for t in corpus["texts"]]
    bm25 = BM25Okapi(tokenized_corpus)
    _BM25_INDEX_CACHE[channel_id] = bm25
    _BM25_CORPUS_CACHE[channel_id] = corpus
    return bm25, corpus


def search(channel_id: str, query: str, top_k: int) -> list[Document]:
    """Return the top_k BM25 matches for the query as Documents (empty if no corpus)."""
    bm25, corpus = _get_index(channel_id)
    if bm25 is None:
        return []
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        Document(page_content=corpus["texts"][i], metadata=corpus["metadatas"][i])
        for i in ranked
    ]
