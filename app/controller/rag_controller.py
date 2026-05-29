import os
from operator import itemgetter
from concurrent.futures import ThreadPoolExecutor

from fastapi import HTTPException

from app.config.logger import logger
from app.config.settings import settings
from app.database.redis import save_session_to_redis, load_session_from_redis
from app.services.rag_service import RAGService
from app.utilities.rag_utilities import RAGUtilities
from app.utilities.timer import timer
from app.retrieval.chunking import chunk_text
from app.retrieval import bm25_index
from app.retrieval.hybrid_retriever import HybridRetriever

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory


EMBEDDING_DIR = settings.EMBEDDING_DIR

class RAGController:
    """Controller class to handle RAG API requests."""


    def __init__(self):
        """Initialize RAGUtilities and get the embedding model."""
        self.embedding_model = RAGUtilities().get_embedding_model()


    @timer
    def create_document_embeddings(self, channel_id: str, file_path: str):
        """Chunk a document and upsert it into the channel's Chroma collection."""
        try:
            if not os.path.isfile(file_path):
                logger.error("File upload error.")
                raise HTTPException(status_code=404, detail="File not found")

            filename = os.path.basename(file_path)
            logger.info(f"Embedding '{filename}' into channel '{channel_id}'")

            text = RAGService.get_text(file_path)
            if not text:
                logger.warning(f"No content extracted from file: {filename}.")
                return None

            docs = chunk_text(text, channel_id=channel_id, filename=filename)
            if not docs:
                logger.warning(f"No chunks produced for: {filename}.")
                return None

            doc_id = docs[0].metadata["doc_id"]
            persist_directory = os.path.join(EMBEDDING_DIR, channel_id)
            os.makedirs(persist_directory, exist_ok=True)

            Chroma.from_documents(
                documents=docs,
                embedding=self.embedding_model,
                persist_directory=persist_directory,
                collection_name=settings.CHROMA_COLLECTION_NAME,
            )

            bm25_index.add_documents(channel_id, docs)

            logger.info(f"Embedded {len(docs)} chunks for '{filename}'")
            return {"message": "Embeddings created", "doc_id": doc_id,
                    "path": persist_directory, "chunks": len(docs)}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in create_document_embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create document embeddings")


    @timer
    def chat_with_document(self, request: dict):
        """Hybrid RAG chat: contextualize -> hybrid retrieve -> answer, with Redis history."""
        try:
            channel_id = request.get("channel_id")
            message = request.get("message")
            filename = request.get("filename")  # optional: restricts to one doc in the channel

            if not channel_id or not message:
                logger.warning("Invalid request payload")
                return {
                    "success": False,
                    "message": "Invalid request payload",
                    "data": {},
                    "error": {"code": 400,
                              "message": "Missing required fields: channel_id or message"},
                }

            logger.info(f"Processing chat for channel: {channel_id}")
            user_input = message.strip()

            utils = RAGUtilities()
            vectorstore = utils.load_embeddings(channel_id)
            if vectorstore is None:
                logger.error(f"No embeddings for channel {channel_id}")
                return {
                    "success": False,
                    "message": "No documents found for this channel",
                    "data": {},
                    "error": {"code": 404,
                              "message": "Please upload a document first to generate embeddings"},
                }

            session_data = load_session_from_redis(channel_id)
            chat_history = (
                session_data.get(channel_id, ChatMessageHistory(messages=[]))
                if session_data else ChatMessageHistory(messages=[])
            )

            standalone_query = utils.contextualize_question(user_input, chat_history.messages)

            retriever = HybridRetriever(channel_id, vectorstore)
            docs = retriever.retrieve(standalone_query, filename=filename)
            context = "\n\n".join(d.page_content for d in docs)

            output = utils.answer(user_input, context, chat_history.messages, filename or channel_id)

            chat_history.messages.append(HumanMessage(content=user_input))
            chat_history.messages.append(AIMessage(content=output))
            save_session_to_redis(channel_id, {channel_id: chat_history})

            logger.info("Chat response generated successfully.")
            return {
                "success": True,
                "message": "Response generated successfully",
                "data": {"user_input": user_input, "bot_output": output},
                "error": None,
            }

        except Exception as e:
            logger.error(f"Error in chat_with_document: {str(e)}")
            return {
                "success": False,
                "message": "Internal server error during chat processing",
                "data": {},
                "error": {"code": 500, "message": str(e)},
            }
