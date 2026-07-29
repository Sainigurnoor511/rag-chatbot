"""Generate a synthetic golden Q&A set from a channel's chunks using the LLM."""
import json

from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate

from app.config.logger import logger


def _question_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are generating evaluation data. Given a passage, write ONE clear, "
             "specific question that is answerable SOLELY from the passage. Return only "
             "the question text, nothing else."),
            ("human", "Passage:\n{passage}"),
        ]
    )


def generate_golden_set(chunks: list[Document], llm, max_questions: int = 20) -> list[dict]:
    """For up to max_questions chunks, generate a question whose answer is in that chunk."""
    prompt = _question_prompt()
    items: list[dict] = []
    for chunk in chunks[:max_questions]:
        try:
            value = prompt.invoke({"passage": chunk.page_content})
            question = llm.invoke(value).content.strip()
        except Exception as e:
            logger.error(f"golden-set question generation failed: {e}")
            continue
        items.append({
            "question": question,
            "relevant_chunk_ids": [chunk.metadata.get("chunk_id")],
            "ground_truth_context": chunk.page_content,
            "source": chunk.metadata.get("source"),
        })
    return items


def save_golden_set(items: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def load_golden_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
