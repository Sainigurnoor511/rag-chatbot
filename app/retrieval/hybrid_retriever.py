from langchain.schema import Document

from app.config.settings import settings
from app.config.logger import logger
from app.retrieval import bm25_index
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import CrossEncoderReranker


class HybridRetriever:
    """Per-channel hybrid retrieval: dense (Chroma) + sparse (BM25) -> RRF -> rerank."""

    def __init__(self, channel_id: str, vectorstore, reranker=None):
        self.channel_id = channel_id
        self.vectorstore = vectorstore
        self.reranker = reranker or CrossEncoderReranker()

    def retrieve(self, query: str, filename: str | None = None) -> list[Document]:
        source_filter = {"source": filename} if filename else None

        try:
            dense = self.vectorstore.similarity_search(
                query, k=settings.DENSE_TOP_K, filter=source_filter
            )
        except Exception as e:
            logger.error(f"Dense search failed for channel {self.channel_id}: {e}")
            dense = []

        sparse = bm25_index.search(self.channel_id, query, settings.BM25_TOP_K)
        if filename:
            sparse = [d for d in sparse if d.metadata.get("source") == filename]

        fused = reciprocal_rank_fusion([dense, sparse], k=settings.RRF_K)
        if not fused:
            return []

        reranked = self.reranker.rerank(query, fused, settings.RERANK_TOP_N)
        logger.info(
            f"Hybrid retrieve channel={self.channel_id}: "
            f"dense={len(dense)} sparse={len(sparse)} fused={len(fused)} kept={len(reranked)}"
        )
        return reranked
