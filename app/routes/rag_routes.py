from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Request
from ..config.settings import settings
from app.middleware.auth import require_api_key
from app.middleware.rate_limit import limiter
from ..controller.rag_controller import RAGController
from ..config.logger import logger
from app.repository.channel_repository import register_document
from fastapi import Form
import os
import shutil
from pydantic import BaseModel
from fastapi.responses import JSONResponse

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
    filename: str | None = None


@router.get("/status")
async def status():
    """Health check endpoint."""
    return {
        "project": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "environment": PROJECT_ENVIRONMENT,
        "status": "API is up and running",
    }



@router.post("/upload", dependencies=[Depends(require_api_key)])
@limiter.limit(lambda: settings.RATE_LIMIT_UPLOAD)
async def upload_file(request: Request, channel_id: str = Form(...), file: UploadFile = File(...)):
    """Upload a PDF/DOCX into a channel and generate embeddings."""
    if not file.filename:
        return create_error_response("No filename provided.", 400)

    if not file.filename.endswith((".pdf", ".docx")):
        return create_error_response("Unsupported file format. Use PDF or DOCX.", 400)

    if file.size and file.size > 50 * 1024 * 1024:
        return create_error_response("File too large. Maximum size is 50MB.", 400)

    os.makedirs(PROJECT_UPLOAD_DIRECTORY, exist_ok=True)
    file_path = os.path.join(PROJECT_UPLOAD_DIRECTORY, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = RAGController().create_document_embeddings(
            channel_id=channel_id, file_path=file_path
        )
        if result is None:
            if os.path.exists(file_path):
                os.remove(file_path)
            return create_error_response("Failed to generate embeddings.", 500)

        register_document(channel_id, result["doc_id"], file.filename)

        return {
            "success": True,
            "message": "File uploaded and embeddings created successfully",
            "data": {
                "channel_id": channel_id,
                "file_name": file.filename,
                "doc_id": result["doc_id"],
                "chunks": result.get("chunks"),
            },
            "error": None,
        }
    except Exception as e:
        logger.error(f"Unexpected error during file upload: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return create_error_response(
            "Internal server error during file processing", 500, {"details": str(e)}
        )


@router.post("/chat", dependencies=[Depends(require_api_key)])
@limiter.limit(lambda: settings.RATE_LIMIT_CHAT)
async def chat(request: Request, body: ChatRequest):
    """API endpoint to handle RAG chat requests."""
    try:
        request_dict = body.model_dump()
        response = RAGController().chat_with_document(request=request_dict)
        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return create_error_response("Internal server error during chat processing", 500, {"details": str(e)})

@router.get("/sentry-debug")
async def trigger_error():
    if settings.ENVIRONMENT == "production":
        raise HTTPException(status_code=404, detail="Not found")
    division_by_zero = 1 / 0
    return division_by_zero