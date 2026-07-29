from unittest.mock import MagicMock, patch

import pytest

from app.controller.rag_controller import RAGController


@pytest.fixture
def controller():
    with patch("app.controller.rag_controller.RAGUtilities") as MockUtils:
        MockUtils.return_value.get_embedding_model.return_value = MagicMock()
        yield RAGController()


def test_create_document_embeddings_uses_docling_parser(tmp_path, controller):
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake pdf bytes")

    mock_parsed = MagicMock()
    mock_parsed.to_text_stream.return_value = "Extracted paragraph text via Docling."

    with patch("app.controller.rag_controller.parse_document", return_value=mock_parsed) as mock_parse, \
         patch("app.controller.rag_controller.Chroma") as MockChroma, \
         patch("app.controller.rag_controller.bm25_index") as mock_bm25:

        result = controller.create_document_embeddings(
            channel_id="chan1", file_path=str(file_path)
        )

    mock_parse.assert_called_once_with(str(file_path), source_type="pdf")
    assert result is not None
    assert result["chunks"] > 0
    MockChroma.from_documents.assert_called_once()
    mock_bm25.add_documents.assert_called_once()


def test_create_document_embeddings_returns_none_for_empty_parse(tmp_path, controller):
    file_path = tmp_path / "empty.docx"
    file_path.write_bytes(b"fake docx bytes")

    mock_parsed = MagicMock()
    mock_parsed.to_text_stream.return_value = ""

    with patch("app.controller.rag_controller.parse_document", return_value=mock_parsed):
        result = controller.create_document_embeddings(
            channel_id="chan1", file_path=str(file_path)
        )

    assert result is None
