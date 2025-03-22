from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Body
from ..config.settings import settings
from ..services.rag_service import RAGService
from ..controller.rag_controller import RAGController
from ..config.logger import logger
import os
import shutil
from pydantic import BaseModel
from fastapi.responses import JSONResponse

router = APIRouter()

UPLOAD_DIR = "E:/Projects/rag-chatbot/data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# FastAPI route to call chat_with_rag
class ChatRequest(BaseModel):
    channel_id: str
    message: str
    filename: str
    # file_path: str


@router.get("/status")
async def status():
    """Health check endpoint."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "API is up and running",
    }


@router.post("/test-logs")
async def test_logs():
    """Test logging levels."""
    return RAGController.logger_test()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads PDF or DOCX file via form-data and generates embeddings."""
    
    # Validate file format
    if not file.filename.endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Unsupported file format. Use PDF or DOCX.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save uploaded file
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Generate embeddings
    try:
        result = RAGController.prepare_rag_documents(file_path=file_path)
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings.")
        if result:
            return {
                "message": "File uploaded and embeddings created",
                "file_name": file.filename,
            }
            
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat(request: ChatRequest):
    """API endpoint to handle RAG chat requests."""
    try:
        request_dict = request.model_dump()
        response = RAGController().chat_with_rag(request=request_dict)
        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")