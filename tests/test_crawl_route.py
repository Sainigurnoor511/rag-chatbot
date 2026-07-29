from unittest.mock import MagicMock, patch

import httpx
import pytest

import app.routes.rag_routes as rag_routes
from main import app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_post_crawl_returns_job_id_and_queued_status(client):
    with patch.object(rag_routes, "create_job", return_value=True) as mock_create, \
         patch.object(rag_routes.asyncio, "create_task") as mock_create_task, \
         patch.object(rag_routes, "RAGUtilities") as MockUtils, \
         patch.object(rag_routes, "settings") as mock_settings:
        MockUtils.return_value.get_embedding_model.return_value = MagicMock()
        mock_settings.CRAWL_MAX_PAGES = 200
        mock_settings.CRAWL_MAX_DEPTH = 5
        mock_settings.api_keys_list.return_value = []  # auth disabled (dev default)

        response = await client.post(
            "/api/v1/rag-chatbot/crawl",
            json={"channel_id": "chan1", "base_url": "https://example.com", "include_paths": ["/docs"]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "job_id" in body["data"]
    assert body["data"]["status"] == "queued"
    mock_create.assert_called_once()
    mock_create_task.assert_called_once()


async def test_get_crawl_job_returns_status(client):
    fake_job = {
        "channel_id": "chan1",
        "base_url": "https://example.com",
        "status": "embedding",
        "pages_found": 10,
        "pages_processed": 4,
        "error": None,
    }
    with patch.object(rag_routes, "get_job", return_value=fake_job):
        response = await client.get("/api/v1/rag-chatbot/crawl/some-job-id")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["status"] == "embedding"
    assert body["data"]["pages_processed"] == 4


async def test_get_crawl_job_returns_404_when_missing(client):
    with patch.object(rag_routes, "get_job", return_value=None):
        response = await client.get("/api/v1/rag-chatbot/crawl/does-not-exist")

    assert response.status_code == 404
