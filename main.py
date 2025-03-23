import os
import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager  # Correct import
from app.routes.rag_routes import router
from app.config.settings import settings
from app.config.logger import logger
import uvicorn

# Lifespan context manager for graceful shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gracefully handle reload cancellations."""
    try:
        # logger.info("Starting up...")
        yield  # App runs here
    except asyncio.CancelledError:
        # logger.info("Reload or shutdown detected. Ignoring CancelledError.")
        await asyncio.sleep(0.1)  # Delay to allow graceful shutdown
    # finally:
        # logger.info("Shutting down...")

# Initialize FastAPI with lifespan and routes
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="RAG Embeddings API",
    docs_url="/docs",
    redoc_url="/redocs",
    lifespan=lifespan  # Use the lifespan here
)

# Include Routes
app.include_router(router, prefix="/api/v1/rag-chatbot", tags=["RAG Embeddings"])

os.makedirs(settings.LOG_DIR, exist_ok=True)  # Ensure log directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)  # Ensure upload directory exists
os.makedirs(settings.EMBEDDING_PATH, exist_ok=True)  # Ensure embedding directory exists
os.makedirs(settings.LOCAL_EMBEDDING_MODEL, exist_ok=True)  # Ensure model directory exists

# Main Entry Point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_delay=1,       # Add delay to avoid frequent reloads
        workers=2              # Use multiple workers for smoother reload
    )
