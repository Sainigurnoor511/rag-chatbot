import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from ..config.settings import settings
from ..controller.rag_controller import RAGController
from ..config.logger import logger
from app.utilities.file_embeddings_handler import register_file 
import os
import shutil
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from uuid import uuid4

router = APIRouter()

PROJECT_NAME = settings.PROJECT_NAME
PROJECT_VERSION = settings.VERSION,
PROJECT_ENVIRONMENT = settings.ENVIRONMENT
PROJECT_UPLOAD_DIRECTORY = settings.UPLOAD_DIR
PROJECT_EMBEDDING_DIRECTORY = settings.EMBEDDING_DIR

class ChatRequest(BaseModel):
    channel_id: str
    message: str
    filename: str
    # file_path: str


@router.get("/status")
async def status():
    """Health check endpoint."""
    return {
        "project": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "environment": PROJECT_ENVIRONMENT,
        "status": "API is up and running",
    }



@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads PDF or DOCX file via form-data and generates embeddings."""
    
    if not file.filename.endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Unsupported file format. Use PDF or DOCX.")

    # Ensure upload directory exists
    os.makedirs(PROJECT_UPLOAD_DIRECTORY, exist_ok=True)
    os.makedirs(PROJECT_EMBEDDING_DIRECTORY, exist_ok=True)

    session_id = str(uuid4())
    file_path = os.path.join(PROJECT_UPLOAD_DIRECTORY, file.filename)
    embedding_path = os.path.join(PROJECT_EMBEDDING_DIRECTORY, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = RAGController().create_document_embeddings(file_path=file_path)

        if result is None:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings.")

        register_file(session_id, file_path, embedding_path)

        return {
            "message": "File uploaded and embeddings created",
            "file_name": file.filename,
            "session_id": session_id
        }

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat(request: ChatRequest):
    """API endpoint to handle RAG chat requests."""
    try:
        request_dict = request.model_dump()
        response = RAGController().chat_with_document(request=request_dict)
        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
