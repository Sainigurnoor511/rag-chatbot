from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from app.config.logger import logger


class OptimumEmbeddingWrapper:
    """Wrapper class to make OptimumEmbedding compatible with LangChain."""
    
    
    def __init__(self, folder_name):
        self.embedding_model = OptimumEmbedding(folder_name=folder_name, device=0)
    

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        try:
            embeddings = [self.embedding_model.get_text_embedding(text) for text in texts]
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed documents: {str(e)}")
            raise
    

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        try:
            return self.embedding_model.get_text_embedding(text)
        except Exception as e:
            logger.error(f"Failed to embed query: {str(e)}")
            raise




class FastEmbedWrapper:
    """Wrapper to make FastEmbed compatible with Chroma."""
    

    def __init__(self, model):
        self.model = model


    def embed_documents(self, texts):
        """Wrapper method to make FastEmbed compatible with Chroma."""
        return self.model.embed(texts)


    def embed_query(self, text):
        """Wrapper method to embed a single query."""
        return self.model.embed([text])[0]
