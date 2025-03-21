from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from ..services.rag_services import RAGService
import os
import shutil

router = APIRouter()

UPLOAD_DIR = "E:/Projects/ragbot/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/status")
async def status():
    """Health check endpoint."""
    return {"status": "API is running!"}

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
        result = RAGService.prepare_rag_documents(file_path)
        return {
            "message": "File uploaded and embeddings created",
            "file_name": file.filename,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
