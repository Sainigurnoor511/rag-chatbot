from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Body
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


@router.post("/chat-rag")
async def chat_rag(
    payload: dict = Body(...)
):
    """
    RAG Chat Route:
    - Takes JSON input with session_id.
    - Automatically maintains chat history by session.
    """

    file_name = payload.get("file_name")
    user_input = payload.get("user_input")
    session_id = payload.get("session_id")

    if not file_name or not user_input or not session_id:
        raise HTTPException(status_code=400, detail="Missing required parameters: 'file_name', 'user_input', or 'session_id'")

    file_path = os.path.join(UPLOAD_DIR, file_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{file_name}' not found")

    try:
        response = RAGController.chat_with_rag(file_path, user_input, session_id)

        return {
            "message": "Chat response generated successfully",
            "user_input": response["user_input"],
            "bot_output": response["bot_output"],
            "chat_history": response["chat_history"]
        }

    except Exception as e:
        logger.error(f"Error during RAG chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))