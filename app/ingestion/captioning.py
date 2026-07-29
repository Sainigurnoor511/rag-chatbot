import base64

from langchain_core.messages import HumanMessage

from app.config.logger import logger
from app.config.settings import settings

_vision_llm_instance = None


def _get_vision_llm():
    global _vision_llm_instance
    if _vision_llm_instance is None:
        from langchain_groq import ChatGroq
        _vision_llm_instance = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            temperature=0.1,
            model_name=settings.GROQ_VISION_MODEL,
        )
    return _vision_llm_instance


def caption_figure(image_bytes: bytes) -> str:
    """Generate a short text description of a figure/diagram image via a vision-capable LLM.

    Returns an empty string on any failure so one bad image never fails a whole document ingest.
    """
    try:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this figure or diagram in 1-2 concise sentences, focusing on what information it conveys."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
            ]
        )
        response = _get_vision_llm().invoke([message])
        return response.content
    except Exception as e:
        logger.warning(f"Figure captioning failed: {e}")
        return ""
