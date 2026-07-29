from fastapi import APIRouter, UploadFile, File, Depends, Request
from ..config.settings import settings
from app.middleware.auth import require_api_key
from app.middleware.rate_limit import limiter
from ..controller.rag_controller import RAGController
from ..config.logger import logger
from app.repository.channel_repository import register_document
from fastapi import Form
import os
import shutil
import asyncio
import uuid
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from app.repository.crawl_jobs import create_job, get_job
from app.ingestion.pipeline import run_crawl_job
from app.utilities.rag_utilities import RAGUtilities

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
PROJECT_VERSION = settings.VERSION
PROJECT_ENVIRONMENT = settings.ENVIRONMENT
PROJECT_UPLOAD_DIRECTORY = settings.UPLOAD_DIR
PROJECT_EMBEDDING_DIRECTORY = settings.EMBEDDING_DIR

class ChatRequest(BaseModel):
    channel_id: str
    message: str
    filename: str | None = None


class CrawlRequest(BaseModel):
    channel_id: str
    base_url: str
    include_paths: list[str] = []
    max_pages: int | None = None
    max_depth: int | None = None


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


@router.post("/crawl", dependencies=[Depends(require_api_key)])
@limiter.limit(lambda: settings.RATE_LIMIT_UPLOAD)
async def start_crawl(request: Request, body: CrawlRequest):
    """Start an async crawl job: scrape a site's same-domain pages and embed them into a channel."""
    job_id = str(uuid.uuid4())
    max_pages = body.max_pages or settings.CRAWL_MAX_PAGES
    max_depth = body.max_depth or settings.CRAWL_MAX_DEPTH

    ok = create_job(job_id, channel_id=body.channel_id, base_url=body.base_url)
    if not ok:
        return create_error_response("Failed to create crawl job.", 500)

    embedding_model = RAGUtilities().get_embedding_model()
    asyncio.create_task(
        asyncio.to_thread(
            run_crawl_job,
            job_id=job_id,
            channel_id=body.channel_id,
            base_url=body.base_url,
            include_paths=body.include_paths,
            max_pages=max_pages,
            max_depth=max_depth,
            embedding_model=embedding_model,
        )
    )

    return {
        "success": True,
        "message": "Crawl job started",
        "data": {"job_id": job_id, "status": "queued"},
        "error": None,
    }


@router.get("/crawl/{job_id}", dependencies=[Depends(require_api_key)])
async def get_crawl_status(job_id: str):
    """Poll the status of a crawl job."""
    job = get_job(job_id)
    if job is None:
        return create_error_response("Crawl job not found.", 404)

    return {
        "success": True,
        "message": "Crawl job status",
        "data": job,
        "error": None,
    }


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