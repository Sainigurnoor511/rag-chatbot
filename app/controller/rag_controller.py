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

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory


EMBEDDING_PATH = settings.EMBEDDING_PATH

class RAGController:
    """Controller class to handle RAG API requests."""


    def __init__(self):
        """Initialize RAGUtilities and get the embedding model."""
        self.embedding_model = RAGUtilities().get_embedding_model()


    @timer
    def create_document_embeddings(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """Splits the text and creates embeddings for RAG."""

        try:
            if not os.path.isfile(file_path):
                logger.error(f"File upload error")
                raise HTTPException(status_code=404, detail="File not found")
            
            filename = file_path.split("/")[-1]
            logger.info(f"Preparing RAG documents for file: {filename}")

            text = RAGService.get_text(file_path)
            if not text:
                logger.warning(f"No content extracted from {file_path}.")
                return []

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n"]
            )

            # Create Document objects with metadata directly
            docs = text_splitter.create_documents([text])
            formatted_docs = [
                Document(page_content=doc.page_content, metadata={"source": file_path})
                for doc in docs
            ]

            # Create vector store
            persist_directory = f"{EMBEDDING_PATH}{os.path.basename(file_path)}"
            os.makedirs(persist_directory, exist_ok=True)

            vectorstore = Chroma.from_documents(
                documents=formatted_docs,
                embedding=self.embedding_model,
                persist_directory=persist_directory,
                collection_name="rag"
            )

            logger.info(f"Embeddings created and stored successfully")
            return {"message": "Embeddings created successfully", "path": persist_directory}

        except Exception as e:
            logger.error(f"Error in create_document_embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create document embeddings")



    @timer
    def chat_with_document(self, request: dict):
        """Handles the RAG chat flow."""
        try:
            # Input validation
            channel_id = request.get("channel_id")
            message = request.get("message")
            filename = request.get("filename")

            if not channel_id or not message or not filename:
                logger.warning("Invalid request payload")
                raise HTTPException(status_code=400, detail="Invalid request payload")

            logger.info(f"Processing chat for channel: {channel_id}")

            user_input = message.strip()

            session_data = load_session_from_redis(channel_id)

            retriever, vectorstore = RAGUtilities().create_retriever(filename)

            if retriever is None or vectorstore is None:
                logger.error("Retriever or VectorStore not found")
                raise HTTPException(status_code=404, detail="Retriever or VectorStore not found")

            # Load or initialize chat history
            chat_history = session_data.get(channel_id, ChatMessageHistory(messages=[])) if session_data else ChatMessageHistory(messages=[])
            
            session_dict = {channel_id: chat_history}
            
            conversational_chain = RAGUtilities().create_conversational_chain_history(
                retriever,
                session_dict,
                filename
            )

            chat_history.messages.append(HumanMessage(content=user_input))

            context = ""
            relevant_docs = []

            # Perform similarity search
            results = vectorstore.similarity_search_with_score(user_input, k=2)

            if results:
                most_similar = min(results, key=itemgetter(1))
                most_similar_doc, highest_score = most_similar
                metadata = most_similar_doc.metadata
                source = metadata.get("source", "Unknown")
                relevant_docs.append(source)
                context = most_similar_doc.page_content.replace("\n", " ")

            input_data = {
                "input": user_input,
                "context": context,
                "links": "",
                "chat_history": chat_history.messages,
            }

            # Invoke the conversational chain
            output = conversational_chain.invoke(
                input_data,
                config={"configurable": {"session_id": channel_id}},
            )["answer"]

            chat_history.messages.append(AIMessage(content=output))

            # Save the updated chat history back to Redis
            save_session_to_redis(channel_id, {channel_id: chat_history})

            response_data = {
                "success": True,
                "message": "Response generated successfully",
                "data": {
                    "user_input": user_input,
                    "bot_output": output,
                },
                "error": None,
            }

            logger.info("Chat response generated successfully.")
            return response_data

        except Exception as e:
            logger.error(f"Error in chat_with_bot: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal Server Error")
