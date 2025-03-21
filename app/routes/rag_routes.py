from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from ..config.settings import settings
from ..services.rag_service import RAGService
from ..controller.rag_controller import RAGController
from ..config.logger import logger
import os
import shutil

router = APIRouter()

UPLOAD_DIR = "E:/Projects/rag-chatbot-api/data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


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
        result = RAGController.prepare_rag_documents(file_path)
        if result:
            return {
                "message": "File uploaded and embeddings created",
                "file_name": file.filename,
            }
            
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))
