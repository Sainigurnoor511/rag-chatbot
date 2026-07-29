from unittest.mock import MagicMock, patch

from app.ingestion.parser import parse_document, ParsedDocument, ExtractedFigure


def _make_mock_docling_document(text_items, table_items, picture_items):
    mock_doc = MagicMock()

    text_mocks = []
    for text in text_items:
        m = MagicMock()
        m.text = text
        text_mocks.append(m)
    mock_doc.texts = text_mocks

    table_mocks = []
    for md in table_items:
        m = MagicMock()
        m.export_to_markdown.return_value = md
        table_mocks.append(m)
    mock_doc.tables = table_mocks

    picture_mocks = []
    for img_bytes in picture_items:
        m = MagicMock()
        pil_image_mock = MagicMock()

        def save_side_effect(buf, format=None, _b=img_bytes):
            buf.write(_b)

        pil_image_mock.save.side_effect = save_side_effect
        m.get_image.return_value = pil_image_mock
        picture_mocks.append(m)
    mock_doc.pictures = picture_mocks

    return mock_doc


def test_parse_pdf_extracts_text_tables_and_figures():
    mock_document = _make_mock_docling_document(
        text_items=["First paragraph.", "Second paragraph."],
        table_items=["| a | b |\n|---|---|\n| 1 | 2 |"],
        picture_items=[b"fake-png-bytes"],
    )
    mock_result = MagicMock()
    mock_result.document = mock_document

    with patch("app.ingestion.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_result

        parsed = parse_document("some/path.pdf", source_type="pdf")

    assert parsed.text_blocks == ["First paragraph.", "Second paragraph."]
    assert parsed.tables == ["| a | b |\n|---|---|\n| 1 | 2 |"]
    assert len(parsed.figures) == 1
    assert parsed.figures[0].image_bytes == b"fake-png-bytes"


def test_parsed_document_to_text_stream_includes_captions():
    with patch("app.ingestion.parser.caption_figure") as mock_caption:
        mock_caption.return_value = "A flowchart of the onboarding process."

        parsed = ParsedDocument(
            text_blocks=["Hello world."],
            tables=["| x |\n|---|\n| 1 |"],
            figures=[ExtractedFigure(image_bytes=b"abc", position_hint="page1")],
        )
        stream = parsed.to_text_stream()

    assert "Hello world." in stream
    assert "| x |" in stream
    assert "[Figure: A flowchart of the onboarding process.]" in stream


def test_parsed_document_to_text_stream_skips_empty_captions():
    with patch("app.ingestion.parser.caption_figure") as mock_caption:
        mock_caption.return_value = ""

        parsed = ParsedDocument(
            text_blocks=["Hello world."],
            tables=[],
            figures=[ExtractedFigure(image_bytes=b"abc", position_hint="page1")],
        )
        stream = parsed.to_text_stream()

    assert "Hello world." in stream
    assert "[Figure:" not in stream


def test_parse_document_empty_source_returns_empty_parsed_document():
    mock_document = _make_mock_docling_document([], [], [])
    mock_result = MagicMock()
    mock_result.document = mock_document

    with patch("app.ingestion.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_result
        parsed = parse_document("empty.pdf", source_type="pdf")

    assert parsed.text_blocks == []
    assert parsed.tables == []
    assert parsed.figures == []
