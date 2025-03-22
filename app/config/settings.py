from pydantic_settings import BaseSettings

# Define the Settings class
class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG CHATBOT API"
    VERSION: str = "1.0.0"
    GROQ_API_KEY: str
    ENVIRONMENT: str = "development"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create the settings instance
settings = Settings()
