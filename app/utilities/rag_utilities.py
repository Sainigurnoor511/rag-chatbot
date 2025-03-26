import os

from fastapi import HTTPException

from app.config.logger import logger
from app.config.settings import settings
from app.utilities.optimum_embeddings import OptimumEmbeddingWrapper, FastEmbedWrapper

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from fastembed import TextEmbedding


LOCAL_EMBEDDING_MODEL = settings.LOCAL_EMBEDDING_MODEL
FAST_EMBEDDING_MODEL = settings.FAST_EMBEDDING_MODEL
EMBEDDING_PATH = settings.EMBEDDING_PATH

# Cache to hold loaded vector stores
VECTOR_STORE_CACHE = {}

# In-memory session storage
SESSION_HISTORY = {}    

# Store the global RAG instance
rag_utilities = None


class RAGUtilities:
    """Class for handling RAG (Retrieval-Augmented Generation) utilities."""

    # Class-level caching to prevent reloading
    _model_instance = None
    _llm_instance = None

    def __init__(self):
        """Initialize the LLM and embedding model only once."""
        try:
            # Initialize LLM only once
            if RAGUtilities._llm_instance is None:
                RAGUtilities._llm_instance = ChatGroq(
                    api_key=settings.GROQ_API_KEY,
                    temperature=0.1,
                    model_name="llama-3.2-3b-preview",
                )
                logger.info("LLM initialized successfully")

            self.llm = RAGUtilities._llm_instance

            # Use cached model if it exists
            if RAGUtilities._model_instance is None:
                RAGUtilities._model_instance = self._load_local_or_fallback()

            self.embedding_model = RAGUtilities._model_instance

        except Exception as e:
            logger.error(f"Failed to initialize RAGUtilities: {str(e)}")
            raise

    def _load_local_or_fallback(self):
        """Attempts to load the local embedding model, falls back to FAST_EMBEDDING_MODEL on failure."""
        try:
            embedding_model = OptimumEmbeddingWrapper(folder_name=settings.LOCAL_EMBEDDING_MODEL)
            logger.info("Local model loaded successfully")
            return embedding_model

        except Exception as e:
            logger.warning("Local model not found, falling back to FastEmbed model.")
            embedding_model = FastEmbedWrapper(
                TextEmbedding(model_name="BAAI/bge-base-en-v1.5", providers=["CUDAExecutionProvider"])
            )
            logger.info("Fallback model loaded successfully")
            return embedding_model


    def get_embedding_model(self):
        """Returns the initialized embedding model."""
        
        return self.embedding_model


    def create_retriever(self, filename: str):
        """Create retriever from the document's embeddings."""
        try:
            # logger.info(f"Creating retriever for: {filename}")
            
            vectorstore = self.load_embeddings(filename)

            if vectorstore is None:
                logger.error(f"Vector store not found for file: {filename}")
                return None, None

            # logger.debug(f"Retriever created successfully for {filename}")
            return vectorstore.as_retriever(), vectorstore

        except Exception as e:
            logger.error(f"Error in create_retriever: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create retriever")


    def create_conversational_chain_history(self, retriever, store, filename) -> RunnableWithMessageHistory:
        """Creates conversational chain with message history."""
        try:
            # logger.info(f"Creating conversational chain for: {filename}")

            history_aware_retriever = self.create_contextualize_question_prompt(self.llm, retriever)
            question_answer_chain = self.create_question_answer_chain(self.llm, filename)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            # logger.debug("Conversational chain created successfully.")

            return RunnableWithMessageHistory(
                rag_chain,
                lambda session_id: self.get_session_history(session_id, store),
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )

        except Exception as e:
            logger.error(f"Error in create_conversational_chain_history: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create conversational chain")


    def load_embeddings(self, filename: str):
        """Load embeddings from the specified document filename with caching."""
        try:
            persist_directory = os.path.join(EMBEDDING_PATH, filename)
            os.makedirs(persist_directory, exist_ok=True)

            # Use cached vector store if it exists
            if filename in VECTOR_STORE_CACHE:
                logger.info(f"Using cached vector store for {filename}")
                return VECTOR_STORE_CACHE[filename]

            if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
                logger.warning(f"No embeddings found for {filename}")
                raise HTTPException(status_code=404, detail="No embeddings found")

            logger.info(f"Loading embeddings from {persist_directory}")

            # Load and cache the vector store
            vectorstore = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=persist_directory,
                collection_name=f"{filename}_collection"
            )

            VECTOR_STORE_CACHE[filename] = vectorstore
            return vectorstore

        except Exception as e:
            logger.error(f"Error in load_embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to load embeddings")


    def create_question_answer_chain(self, llm, filename):
        """Create QA chain using the document filename in the prompt."""
        try:
            # logger.info(f"Creating QA chain for: {filename}")
            
            qa_prompt = self.create_qa_prompt(filename)
            qa_chain = create_stuff_documents_chain(llm, qa_prompt)

            # logger.debug("QA chain created successfully.")
            return qa_chain

        except Exception as e:
            logger.error(f"Error in create_question_answer_chain: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create QA chain")


    def create_contextualize_question_prompt(self, llm, retriever):
        """Creates contextualized question prompt."""
        try:
            # logger.info("Creating contextualized question prompt...")

            system_prompt = (
                "Given a chat history and the latest user question, formulate a standalone question "
                "that is clear without the chat history. Only rewrite the question if necessary, "
                "otherwise return it as is."
            )
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ]
            )

            contextual_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )

            # logger.debug("Contextualized question prompt created successfully.")
            return contextual_retriever

        except Exception as e:
            logger.error(f"Error in create_contextualize_question_prompt: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create contextualized question prompt")


    def get_session_history(self, session_id: str, store: dict) -> BaseChatMessageHistory:
        """Retrieve or create a new chat message history for the given session ID."""
        try:
            if session_id not in store:
                # logger.info(f"Creating new session history for: {session_id}")
                store[session_id] = ChatMessageHistory()

            # logger.debug(f"Session history retrieved for: {session_id}")
            return store[session_id]

        except Exception as e:
            logger.error(f"Error in get_session_history: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to get session history")


    def create_qa_prompt(self, filename) -> ChatPromptTemplate:
        """Create a QA prompt template for the document-based RAG chatbot."""
        try:
            # logger.info(f"Creating QA prompt for: {filename}")

            system_prompt = (
                f"You are an AI assistant answering questions based on the document: {filename}. "
                "Use the provided context from the document to generate accurate and concise answers. "
                "Do not add any extra information or speculate. "
                "Strictly provide answers based on the document content. "
                "Keep the answer short, clear, and factual."
                "\n\n{context}"
            )

            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ]
            )

            # logger.debug("QA prompt created successfully.")
            return qa_prompt

        except Exception as e:
            logger.error(f"Error in create_qa_prompt: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create QA prompt")
