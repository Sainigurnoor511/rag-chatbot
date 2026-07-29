from unittest.mock import MagicMock, patch

from app.ingestion.pipeline import run_crawl_job
from app.ingestion.crawler import CrawledPage


def test_run_crawl_job_happy_path():
    fake_pages = [
        CrawledPage(url="https://example.com/docs/a", html="<html>A content</html>"),
        CrawledPage(url="https://example.com/docs/b", html="<html>B content</html>"),
    ]
    mock_parsed = MagicMock()
    mock_parsed.to_text_stream.return_value = "Some extracted page text."

    with patch("app.ingestion.pipeline.crawl_site", return_value=fake_pages) as mock_crawl, \
         patch("app.ingestion.pipeline.parse_document", return_value=mock_parsed) as mock_parse, \
         patch("app.ingestion.pipeline.Chroma") as MockChroma, \
         patch("app.ingestion.pipeline.bm25_index") as mock_bm25, \
         patch("app.ingestion.pipeline.register_document") as mock_register, \
         patch("app.ingestion.pipeline.update_job") as mock_update_job:

        run_crawl_job(
            job_id="job1",
            channel_id="chan1",
            base_url="https://example.com",
            include_paths=["/docs"],
            max_pages=50,
            max_depth=3,
            embedding_model=MagicMock(),
        )

    mock_crawl.assert_called_once_with(
        base_url="https://example.com", include_paths=["/docs"], max_pages=50, max_depth=3
    )
    assert mock_parse.call_count == 2
    assert MockChroma.from_documents.call_count == 2
    assert mock_bm25.add_documents.call_count == 2
    assert mock_register.call_count == 2

    status_calls = [c.kwargs.get("status") for c in mock_update_job.call_args_list if "status" in c.kwargs]
    assert "crawling" in status_calls
    assert "parsing" in status_calls
    assert "embedding" in status_calls
    assert "done" in status_calls


def test_run_crawl_job_marks_failed_on_crawl_exception():
    with patch("app.ingestion.pipeline.crawl_site", side_effect=RuntimeError("unreachable")), \
         patch("app.ingestion.pipeline.update_job") as mock_update_job:

        run_crawl_job(
            job_id="job2",
            channel_id="chan1",
            base_url="https://bad-url.invalid",
            include_paths=[],
            max_pages=50,
            max_depth=3,
            embedding_model=MagicMock(),
        )

    fail_calls = [c for c in mock_update_job.call_args_list if c.kwargs.get("status") == "failed"]
    assert len(fail_calls) == 1
    assert "unreachable" in fail_calls[0].kwargs["error"]


def test_run_crawl_job_skips_page_on_parse_failure_but_continues():
    fake_pages = [
        CrawledPage(url="https://example.com/docs/a", html="<html>A</html>"),
        CrawledPage(url="https://example.com/docs/b", html="<html>B</html>"),
    ]
    good_parsed = MagicMock()
    good_parsed.to_text_stream.return_value = "Good page text."

    with patch("app.ingestion.pipeline.crawl_site", return_value=fake_pages), \
         patch("app.ingestion.pipeline.parse_document", side_effect=[RuntimeError("bad html"), good_parsed]), \
         patch("app.ingestion.pipeline.Chroma") as MockChroma, \
         patch("app.ingestion.pipeline.bm25_index") as mock_bm25, \
         patch("app.ingestion.pipeline.register_document"), \
         patch("app.ingestion.pipeline.update_job") as mock_update_job:

        run_crawl_job(
            job_id="job3",
            channel_id="chan1",
            base_url="https://example.com",
            include_paths=["/docs"],
            max_pages=50,
            max_depth=3,
            embedding_model=MagicMock(),
        )

    assert MockChroma.from_documents.call_count == 1
    assert mock_bm25.add_documents.call_count == 1
    status_calls = [c.kwargs.get("status") for c in mock_update_job.call_args_list if "status" in c.kwargs]
    assert "done" in status_calls


def test_run_crawl_job_html_page_gets_table_and_figure_via_docling():
    from app.ingestion.parser import ParsedDocument, ExtractedFigure

    fake_pages = [CrawledPage(url="https://example.com/docs/report", html="<html>...</html>")]

    real_parsed = ParsedDocument(
        text_blocks=["Quarterly report."],
        tables=["| Q1 | Q2 |\n|----|----|\n| 10 | 20 |"],
        figures=[ExtractedFigure(image_bytes=b"chart-bytes", position_hint="fig1")],
    )

    with patch("app.ingestion.pipeline.crawl_site", return_value=fake_pages), \
         patch("app.ingestion.pipeline.parse_document", return_value=real_parsed), \
         patch("app.ingestion.parser.caption_figure", return_value="Revenue grew from 10 to 20."), \
         patch("app.ingestion.pipeline.Chroma") as MockChroma, \
         patch("app.ingestion.pipeline.bm25_index") as mock_bm25, \
         patch("app.ingestion.pipeline.register_document"), \
         patch("app.ingestion.pipeline.update_job"):

        run_crawl_job(
            job_id="job4",
            channel_id="chan1",
            base_url="https://example.com",
            include_paths=["/docs"],
            max_pages=50,
            max_depth=3,
            embedding_model=MagicMock(),
        )

    call_kwargs = MockChroma.from_documents.call_args.kwargs
    embedded_texts = [d.page_content for d in call_kwargs["documents"]]
    combined = "\n".join(embedded_texts)
    assert "Q1 | Q2" in combined
    assert "Revenue grew from 10 to 20." in combined
