import multiprocessing
from dataclasses import dataclass
from urllib.parse import urlparse

from app.config.logger import logger


@dataclass
class CrawledPage:
    url: str
    html: str


def _path_allowed(url: str, include_paths: list[str]) -> bool:
    """Return True if url's path starts with one of include_paths (or include_paths is empty)."""
    if not include_paths:
        return True
    path = urlparse(url).path
    return any(path.startswith(prefix) for prefix in include_paths)


def _spider_worker(base_url: str, include_paths: list[str], max_pages: int,
                    max_depth: int, result_queue: multiprocessing.Queue) -> None:
    """Runs inside a child process: starts a Scrapy CrawlerProcess and pushes results to the queue."""
    import scrapy
    from scrapy.crawler import CrawlerProcess

    domain = urlparse(base_url).netloc
    collected: list[dict] = []

    class SiteSpider(scrapy.Spider):
        name = "site_spider"
        allowed_domains = [domain]
        start_urls = [base_url]

        custom_settings = {
            "DEPTH_LIMIT": max_depth,
            "CLOSESPIDER_PAGECOUNT": max_pages,
            "LOG_ENABLED": False,
            "ROBOTSTXT_OBEY": True,
        }

        def parse(self, response):
            if _path_allowed(response.url, include_paths):
                collected.append({"url": response.url, "html": response.text})
            for href in response.css("a::attr(href)").getall():
                next_url = response.urljoin(href)
                if urlparse(next_url).netloc == domain and _path_allowed(next_url, include_paths):
                    yield response.follow(next_url, callback=self.parse)

    process = CrawlerProcess(settings={"LOG_ENABLED": False})
    process.crawl(SiteSpider)
    process.start()

    result_queue.put(collected)


def _run_spider_and_collect(base_url: str, include_paths: list[str],
                             max_pages: int, max_depth: int) -> list[CrawledPage]:
    """Runs the Scrapy spider in a subprocess (Scrapy's Twisted reactor can only start once
    per OS process, and the caller runs inside uvicorn's asyncio loop) and collects results."""
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_spider_worker,
        args=(base_url, include_paths, max_pages, max_depth, result_queue),
    )
    proc.start()
    proc.join()

    if result_queue.empty():
        logger.warning(f"Crawl subprocess for {base_url} produced no results")
        return []

    raw_pages = result_queue.get()
    return [CrawledPage(url=p["url"], html=p["html"]) for p in raw_pages]


def crawl_site(base_url: str, include_paths: list[str], max_pages: int, max_depth: int) -> list[CrawledPage]:
    """Crawl base_url's domain (optionally restricted to include_paths prefixes), bounded by
    max_pages/max_depth, returning each matched page's URL and raw HTML."""
    return _run_spider_and_collect(base_url, include_paths, max_pages, max_depth)
