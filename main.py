import os
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.config.settings import settings
from app.config.logger import logger
from app.routes.rag_routes import router
from app.utilities.rag_utilities import RAGUtilities

# Global variable for RAG model
rag_utilities = None

# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle RAG model initialization at startup and graceful shutdown."""
    global rag_utilities

    # Startup event
    try:
        rag_utilities = RAGUtilities()  # Load the model once
        yield  # The app runs here
    except Exception as e:
        logger.critical(f"Failed to load RAG model at startup: {str(e)}")
        yield  # Continue running even if initialization fails

    # Shutdown event (for cleanup if needed)
    finally:
        logger.info("Shutting down...")


# Initialize FastAPI app with `lifespan`
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="RAG CHATBOT API",
    docs_url="/docs",
    redoc_url="/redocs",
    lifespan=lifespan  # Use the lifespan handler
)

# Include Routes
app.include_router(router, prefix="/api/v1/rag-chatbot", tags=["RAG CHATBOT"])

# Ensure necessary directories exist
os.makedirs(settings.LOG_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.EMBEDDING_PATH, exist_ok=True)
os.makedirs(settings.LOCAL_EMBEDDING_MODEL, exist_ok=True)

# Main Entry Point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,       # Add delay to avoid frequent reloads
        workers=2              # Use multiple workers for smoother reload
    )
