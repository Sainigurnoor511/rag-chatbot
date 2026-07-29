from unittest.mock import patch

from app.ingestion.crawler import _path_allowed, crawl_site, CrawledPage


def test_path_allowed_with_no_filter_allows_everything():
    assert _path_allowed("https://example.com/anything", []) is True


def test_path_allowed_matches_prefix():
    assert _path_allowed("https://example.com/docs/page1", ["/docs"]) is True
    assert _path_allowed("https://example.com/blog/post1", ["/docs"]) is False


def test_path_allowed_matches_any_of_multiple_prefixes():
    assert _path_allowed("https://example.com/blog/post1", ["/docs", "/blog"]) is True


def test_crawl_site_returns_pages_from_subprocess():
    fake_pages = [
        CrawledPage(url="https://example.com/docs/a", html="<html>A</html>"),
        CrawledPage(url="https://example.com/docs/b", html="<html>B</html>"),
    ]

    with patch("app.ingestion.crawler._run_spider_and_collect", return_value=fake_pages) as mock_run:
        pages = crawl_site(
            base_url="https://example.com",
            include_paths=["/docs"],
            max_pages=50,
            max_depth=3,
        )

    mock_run.assert_called_once_with("https://example.com", ["/docs"], 50, 3)
    assert pages == fake_pages
