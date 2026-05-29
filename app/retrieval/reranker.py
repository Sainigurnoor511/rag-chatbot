from langchain.schema import Document

from app.config.settings import settings
from app.config.logger import logger


class CrossEncoderReranker:
    """Local cross-encoder reranker. Model is loaded once and cached on the class."""

    _model = None

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker model: {settings.RERANKER_MODEL}")
            cls._model = CrossEncoder(settings.RERANKER_MODEL)
        return cls._model

    def rerank(self, query: str, documents: list[Document], top_n: int) -> list[Document]:
        """Score (query, doc) pairs and return the top_n documents by descending score."""
        if not documents:
            return []
        model = self._get_model()
        pairs = [(query, d.page_content) for d in documents]
        scores = model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked[:top_n]]
