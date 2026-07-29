import io
from dataclasses import dataclass, field
from typing import Literal

from docling.document_converter import DocumentConverter

from app.config.logger import logger


@dataclass
class ExtractedFigure:
    image_bytes: bytes
    position_hint: str


@dataclass
class ParsedDocument:
    text_blocks: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    figures: list[ExtractedFigure] = field(default_factory=list)

    def to_text_stream(self) -> str:
        parts = list(self.text_blocks) + list(self.tables)
        for i in range(len(self.figures)):
            parts.append(f"[[FIGURE:{i}]]")
        return "\n\n".join(parts)


def _extract_figure_bytes(picture) -> bytes | None:
    try:
        pil_image = picture.get_image()
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Failed to extract figure image: {e}")
        return None


def parse_document(source: str, source_type: Literal["pdf", "docx", "html"]) -> ParsedDocument:
    """Parse a PDF/DOCX file path or raw HTML string into text, tables, and figures via Docling."""
    converter = DocumentConverter()
    result = converter.convert(source)
    doc = result.document

    text_blocks = [t.text for t in doc.texts if getattr(t, "text", "").strip()]
    tables = [t.export_to_markdown() for t in doc.tables]

    figures = []
    for i, picture in enumerate(doc.pictures):
        img_bytes = _extract_figure_bytes(picture)
        if img_bytes:
            figures.append(ExtractedFigure(image_bytes=img_bytes, position_hint=f"figure-{i}"))

    return ParsedDocument(text_blocks=text_blocks, tables=tables, figures=figures)
