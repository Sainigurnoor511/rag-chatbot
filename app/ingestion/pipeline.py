import os

from langchain_chroma import Chroma

from app.config.logger import logger
from app.config.settings import settings
from app.ingestion.crawler import crawl_site
from app.ingestion.parser import parse_document
from app.repository.channel_repository import register_document
from app.repository.crawl_jobs import update_job
from app.retrieval.chunking import chunk_text
from app.retrieval import bm25_index


def run_crawl_job(job_id: str, channel_id: str, base_url: str, include_paths: list[str],
                   max_pages: int, max_depth: int, embedding_model) -> None:
    """Full crawl -> parse -> caption -> chunk -> embed pipeline for one crawl job.

    Never raises: crawl-level failures mark the job 'failed'; per-page parse failures
    are logged and skipped so one bad page doesn't fail the whole job.
    """
    try:
        update_job(job_id, status="crawling")
        pages = crawl_site(base_url=base_url, include_paths=include_paths,
                            max_pages=max_pages, max_depth=max_depth)
        update_job(job_id, pages_found=len(pages))

        update_job(job_id, status="parsing")
        parsed_texts: list[tuple[str, str]] = []  # (url, text)
        for page in pages:
            try:
                parsed = parse_document(page.html, source_type="html")
                text = parsed.to_text_stream()
                if text:
                    parsed_texts.append((page.url, text))
            except Exception as e:
                logger.warning(f"Failed to parse page {page.url}: {e}")

        update_job(job_id, status="embedding")
        persist_directory = os.path.join(settings.EMBEDDING_DIR, channel_id)
        os.makedirs(persist_directory, exist_ok=True)

        pages_processed = 0
        for url, text in parsed_texts:
            docs = chunk_text(text, channel_id=channel_id, filename=url)
            if not docs:
                continue
            doc_id = docs[0].metadata["doc_id"]
            Chroma.from_documents(
                documents=docs,
                embedding=embedding_model,
                persist_directory=persist_directory,
                collection_name=settings.CHROMA_COLLECTION_NAME,
            )
            bm25_index.add_documents(channel_id, docs)
            register_document(channel_id, doc_id, url)
            pages_processed += 1
            update_job(job_id, pages_processed=pages_processed)

        update_job(job_id, status="done")

    except Exception as e:
        logger.error(f"Crawl job {job_id} failed: {e}")
        update_job(job_id, status="failed", error=str(e))
