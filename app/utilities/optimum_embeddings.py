from app.config.logger import logger

# BGE query-side instruction prefix (bge-*-en models expect this on queries only,
# not on indexed documents). See BAAI/bge-base-en-v1.5 model card.
BGE_QUERY_INSTRUCTION = "Represent this question for searching relevant passages: "


class OptimumEmbeddingWrapper:
    """ONNX embedding via optimum + transformers directly (no llama_index).

    Replicates llama_index's OptimumEmbedding behavior (CLS pooling, L2 normalize,
    BGE query prefix) without importing llama_index's torch/nltk-heavy dependency
    chain, which added ~8s to process startup for a single wrapper class.
    """

    def __init__(self, folder_name, pooling: str = "cls", normalize: bool = True):
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        self.model = ORTModelForFeatureExtraction.from_pretrained(folder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(folder_name)
        self.pooling = pooling
        self.normalize = normalize

        try:
            self.max_length = min(
                int(self.model.config.max_position_embeddings),
                int(self.tokenizer.model_max_length),
            )
        except Exception:
            self.max_length = int(self.model.config.max_position_embeddings)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import torch

        encoded = self.tokenizer(
            texts,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_output = self.model(**encoded)

        if self.pooling == "cls":
            embeddings = model_output[0][:, 0]
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).expand(model_output[0].size()).float()
            embeddings = torch.sum(model_output[0] * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        try:
            return self._embed(texts)
        except Exception as e:
            logger.error(f"Failed to embed documents: {str(e)}")
            raise

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        try:
            return self._embed([BGE_QUERY_INSTRUCTION + text])[0]
        except Exception as e:
            logger.error(f"Failed to embed query: {str(e)}")
            raise




class FastEmbedWrapper:
    """Wrapper to make FastEmbed compatible with Chroma."""

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        """Wrapper method to make FastEmbed compatible with Chroma."""
        return list(self.model.embed(texts))

    def embed_query(self, text):
        """Wrapper method to embed a single query."""
        return list(self.model.embed([text]))[0]
