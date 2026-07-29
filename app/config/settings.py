from pydantic_settings import BaseSettings
from pathlib import Path

# Dynamically determine the base project directory
BASE_DIR = Path(__file__).resolve().parents[2]  # Adjust this according to the structure

class Settings(BaseSettings):

    # Project Configuration
    PROJECT_NAME: str = "RAG CHATBOT API"
    VERSION: str = "1.0.0"
    GROQ_API_KEY: str
    GROQ_MODEL: str
    ENVIRONMENT: str = "development"
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # Correctly resolved paths
    UPLOAD_DIR: str = str(BASE_DIR / "data" / "uploads" )
    EMBEDDING_DIR: str = str(BASE_DIR / "data" / "database" )
    LOCAL_EMBEDDING_MODEL: str = str(BASE_DIR / "app" / "models" / "bge-large-en-v1.5_ONNX" )

    # Logs directory
    LOG_DIR: str = str(BASE_DIR / "logs")

    # Model Configuration
    FAST_EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"

    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # Per-channel storage
    CHROMA_COLLECTION_NAME: str = "rag_channel"
    CHANNEL_TTL_SECONDS: int = 1800  # 30 minutes

    # Hybrid retrieval (Phase 2)
    DENSE_TOP_K: int = 20
    BM25_TOP_K: int = 20
    RRF_K: int = 60
    RERANK_TOP_N: int = 5
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # Production cross-cutting (Phase 3)
    API_KEYS: str = ""  # comma-separated; empty disables auth (dev)
    RATE_LIMIT_CHAT: str = "30/minute"
    RATE_LIMIT_UPLOAD: str = "10/minute"
    ENABLE_QUERY_CACHE: bool = False
    QUERY_CACHE_TTL: int = 300
    METRICS_ENABLED: bool = True

    # Ingestion: crawling + vision captioning (Phase 5)
    GROQ_VISION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    CRAWL_MAX_PAGES: int = 200
    CRAWL_MAX_DEPTH: int = 5
    CRAWL_JOB_TTL_SECONDS: int = 3600

    def api_keys_list(self) -> list[str]:
        return [k.strip() for k in self.API_KEYS.split(",") if k.strip()]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create the settings instance
settings = Settings()
