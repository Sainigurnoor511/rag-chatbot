import os
import asyncio
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.config.settings import settings
from app.config.logger import logger
from app.routes.rag_routes import router
from app.utilities.rag_utilities import RAGUtilities
from app.utilities.file_embeddings_handler import cleanup_expired_files

# Global variable for RAG model
rag_utilities = None

# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle RAG model initialization at startup and graceful shutdown."""
    global rag_utilities

    # Startup event
    try:
        # Ensure necessary directories exist before loading the model
        logger.info("Creating necessary directories...")
        os.makedirs(settings.LOG_DIR, exist_ok=True)
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        os.makedirs(settings.EMBEDDING_DIR, exist_ok=True)
        os.makedirs(settings.LOCAL_EMBEDDING_MODEL, exist_ok=True)
        logger.info("Directories created successfully")

        rag_utilities = RAGUtilities()  # Load the model once

        # start background task for cleanup
        asyncio.create_task(cleanup_expired_files())

        yield  # Yield control to the app

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

# Main Entry Point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,       # Add delay to avoid frequent reloads
        workers=2              # Use multiple workers for smoother reload
    )
