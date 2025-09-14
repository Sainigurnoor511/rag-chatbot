from fastapi import APIRouter, UploadFile, File
from ..config.settings import settings
from ..controller.rag_controller import RAGController
from ..config.logger import logger
from app.utilities.file_embeddings_handler import register_file 
import os
import shutil
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from uuid import uuid4

def create_error_response(message: str, error_code: int = 500, details: dict = None):
    """Create a standardized error response."""
    return JSONResponse(
        status_code=error_code,
        content={
            "success": False,
            "message": message,
            "data": details or {},
            "error": {
                "code": error_code,
                "message": message
            }
        }
    )

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
    
    # Validate file
    if not file.filename:
        return create_error_response("No filename provided.", 400)
    
    if not file.filename.endswith((".pdf", ".docx")):
        return create_error_response("Unsupported file format. Use PDF or DOCX.", 400)
    
    # Check file size (limit to 50MB)
    if file.size and file.size > 50 * 1024 * 1024:
        return create_error_response("File too large. Maximum size is 50MB.", 400)

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
            # Clean up uploaded file if embedding creation fails
            if os.path.exists(file_path):
                os.remove(file_path)
            return create_error_response("Failed to generate embeddings.", 500)

        register_file(session_id, file_path, embedding_path)

        return {
            "success": True,
            "message": "File uploaded and embeddings created successfully",
            "data": {
                "session_id": session_id,
                "file_name": file.filename,
                # "file_path": file_path,
                # "embedding_path": embedding_path
            },
            "error": None
        }

    except Exception as e:
        logger.error(f"Unexpected error during file upload: {str(e)}")
        # Clean up uploaded file on unexpected error
        if os.path.exists(file_path):
            os.remove(file_path)
        return create_error_response("Internal server error during file processing", 500, {"details": str(e)})


@router.post("/chat")
async def chat(request: ChatRequest):
    """API endpoint to handle RAG chat requests."""
    try:
        request_dict = request.model_dump()
        response = RAGController().chat_with_document(request=request_dict)
        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return create_error_response("Internal server error during chat processing", 500, {"details": str(e)})
