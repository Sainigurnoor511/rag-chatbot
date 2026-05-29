import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.config.settings import settings


def compute_doc_id(filename: str) -> str:
    """Stable 16-char id for a filename so re-uploads replace, not duplicate."""
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]


def chunk_text(text: str, channel_id: str, filename: str,
               chunk_size: int | None = None,
               chunk_overlap: int | None = None) -> list[Document]:
    """Split text into LangChain Documents tagged with channel/source metadata."""
    if not text or not text.strip():
        return []

    chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
    doc_id = compute_doc_id(filename)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    raw_chunks = splitter.split_text(text)

    return [
        Document(
            page_content=chunk,
            metadata={
                "channel_id": channel_id,
                "source": filename,
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}-{i}",
            },
        )
        for i, chunk in enumerate(raw_chunks)
    ]
