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
    LOCAL_EMBEDDING_MODEL: str = str(BASE_DIR / "app" / "models" / "bge-base-en-v1.5_ONNX" )

    # Logs directory
    LOG_DIR: str = str(BASE_DIR / "logs")

    # Model Configuration
    FAST_EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create the settings instance
settings = Settings()
