from unittest.mock import MagicMock, patch

from app.ingestion.captioning import caption_figure


def test_caption_figure_returns_llm_text_on_success():
    mock_response = MagicMock()
    mock_response.content = "A bar chart showing quarterly revenue."

    with patch("app.ingestion.captioning._get_vision_llm") as mock_get_llm:
        mock_get_llm.return_value.invoke.return_value = mock_response
        result = caption_figure(b"fake-image-bytes")

    assert result == "A bar chart showing quarterly revenue."


def test_caption_figure_returns_empty_string_on_failure():
    with patch("app.ingestion.captioning._get_vision_llm") as mock_get_llm:
        mock_get_llm.return_value.invoke.side_effect = RuntimeError("API error")
        result = caption_figure(b"fake-image-bytes")

    assert result == ""
