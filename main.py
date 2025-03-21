from fastapi import FastAPI
from app.routes.rag_routes import router
from app.config.settings import settings
import uvicorn

app = FastAPI(
    title="RAG Document Embedding API",
    description="API to create and manage RAG embeddings from PDF/DOCX",
    version="1.0.0"
)

# Include Routes
app.include_router(router, prefix="api/v1/rag-chatbot", tags=["RAG Embeddings"])

# Main Entry Point
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
