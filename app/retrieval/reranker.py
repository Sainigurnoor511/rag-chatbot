from langchain.schema import Document

from app.config.settings import settings
from app.config.logger import logger


class CrossEncoderReranker:
    """Cohere-hosted reranker. Client is created once and cached on the class."""

    _client = None

    @classmethod
    def _get_client(cls):
        if cls._client is None:
            import cohere
            logger.info(f"Using Cohere reranker model: {settings.RERANKER_MODEL}")
            cls._client = cohere.Client(settings.COHERE_API_KEY)
        return cls._client

    def rerank(self, query: str, documents: list[Document], top_n: int) -> list[Document]:
        """Score (query, doc) pairs via Cohere Rerank and return the top_n documents."""
        if not documents:
            return []
        client = self._get_client()
        response = client.rerank(
            query=query,
            documents=[d.page_content for d in documents],
            model=settings.RERANKER_MODEL,
            top_n=top_n,
        )
        return [documents[result.index] for result in response.results]
