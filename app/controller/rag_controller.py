import os
from operator import itemgetter

from fastapi import HTTPException

from app.services.rag_service import RAGService
from app.config.logger import logger
from app.config.settings import settings
from app.database.redis import save_session_to_redis, load_session_from_redis
from app.mypackage.myfunction import timer

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import (
    create_history_aware_retriever as langchain_create_history_aware_retriever,
    create_retrieval_chain
)
from langchain.chains.combine_documents import create_stuff_documents_chain

# In-memory session storage
SESSION_HISTORY = {}


class RAGController:
    """Controller class to handle RAG API requests."""


    def __init__(self):
        """Initialize the LLM and embedding model in the constructor."""
        try:
            logger.info("Initializing RAGController...")
            
            # Initialize LLM
            self.llm = ChatGroq(
                api_key=settings.GROQ_API_KEY,
                temperature=0.1,
                model_name="llama3-8b-8192",
            )
            logger.debug("LLM initialized successfully.")

            # Initialize embedding model
            self.embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
            logger.debug("Embedding model initialized successfully.")
        
        except Exception as e:
            logger.error(f"Failed to initialize RAGController: {str(e)}")
            raise

    
    def logger_test():
        """Test logging levels."""
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        logger.critical("This is a critical message")
        return {"message": "Logs created successfully"}


    @timer
    def prepare_rag_documents(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """Splits the text and creates embeddings for RAG."""
        try:
            if not os.path.isfile(file_path):
                logger.error(f"File not found: {file_path}")
                raise HTTPException(status_code=404, detail="File not found")

            logger.info(f"Preparing RAG documents for: {file_path}")

            text = RAGService.get_text(file_path)

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            docs = text_splitter.create_documents([text])

            # Create document objects with metadata
            formatted_docs = [
                Document(page_content=doc.page_content, metadata={"source": file_path})
                for doc in docs
            ]

            # Create vector store
            persist_directory = f"E:/Projects/rag-chatbot/data/database/{os.path.basename(file_path)}"
            os.makedirs(persist_directory, exist_ok=True)

            vectorstore = Chroma.from_documents(
                documents=formatted_docs,
                embedding=self.embedding_model,
                persist_directory=persist_directory,
                collection_name="rag"
            )

            logger.info(f"Embeddings stored in {persist_directory}")
            return {"message": "Embeddings created successfully", "path": persist_directory}

        except Exception as e:
            logger.error(f"Error in prepare_rag_documents: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to prepare RAG documents")


    def chat_with_rag(self, request: dict):
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

            retriever, vs = self.create_retriever(filename)

            if retriever is None or vs is None:
                logger.error("Retriever or VectorStore not found")
                raise HTTPException(status_code=404, detail="Retriever or VectorStore not found")

            # Load or initialize chat history
            chat_history = session_data.get(channel_id, ChatMessageHistory(messages=[])) if session_data else ChatMessageHistory(messages=[])
            
            session_dict = {channel_id: chat_history}
            
            conversational_chain = self.create_conversational_chain_history(
                self.llm,
                retriever,
                session_dict,
                filename
            )

            chat_history.messages.append(HumanMessage(content=user_input))

            context = ""
            relevant_docs = []

            # Perform similarity search
            results = vs.similarity_search_with_score(user_input, k=2)

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
            logger.error(f"Error in chat_with_rag: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal Server Error")


    def load_embeddings(self, filename: str):
        """Load embeddings from the specified document filename."""
        try:
            persist_directory = f'E:/Projects/rag-chatbot/data/database/{filename}'
            
            if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
                logger.warning(f"No embeddings found for {filename}")
                raise HTTPException(status_code=404, detail="No embeddings found")

            logger.info(f"Loading embeddings from {persist_directory}")
            
            vs = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=persist_directory,
                collection_name=f"{filename}_collection"
            )
            
            return vs

        except Exception as e:
            logger.error(f"Error in load_embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to load embeddings")


    def create_retriever(self, filename: str):
        """Create retriever from the document's embeddings."""
        try:
            logger.info(f"Creating retriever for: {filename}")
            
            vectorstore = self.load_embeddings(filename)

            if vectorstore is None:
                logger.error(f"Vector store not found for file: {filename}")
                return None, None

            logger.debug(f"Retriever created successfully for {filename}")
            return vectorstore.as_retriever(), vectorstore

        except Exception as e:
            logger.error(f"Error in create_retriever: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create retriever")


    def create_conversational_chain_history(self, llm, retriever, store, filename) -> RunnableWithMessageHistory:
        """Creates conversational chain with message history."""
        try:
            logger.info(f"Creating conversational chain for: {filename}")

            history_aware_retriever = self.create_contextualize_question_prompt(llm, retriever)
            question_answer_chain = self.create_question_answer_chain(llm, filename)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            logger.debug("Conversational chain created successfully.")

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


    def create_question_answer_chain(self, llm, filename):
        """Create QA chain using the document filename in the prompt."""
        try:
            logger.info(f"Creating QA chain for: {filename}")
            
            qa_prompt = self.create_qa_prompt(filename)
            qa_chain = create_stuff_documents_chain(llm, qa_prompt)

            logger.debug("QA chain created successfully.")
            return qa_chain

        except Exception as e:
            logger.error(f"Error in create_question_answer_chain: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create QA chain")


    def create_contextualize_question_prompt(self, llm, retriever):
        """Creates contextualized question prompt."""
        try:
            logger.info("Creating contextualized question prompt...")

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

            contextual_retriever = langchain_create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )

            logger.debug("Contextualized question prompt created successfully.")
            return contextual_retriever

        except Exception as e:
            logger.error(f"Error in create_contextualize_question_prompt: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create contextualized question prompt")


    def get_session_history(self, session_id: str, store: dict) -> BaseChatMessageHistory:
        """Retrieve or create a new chat message history for the given session ID."""
        try:
            if session_id not in store:
                logger.info(f"Creating new session history for: {session_id}")
                store[session_id] = ChatMessageHistory()

            logger.debug(f"Session history retrieved for: {session_id}")
            return store[session_id]

        except Exception as e:
            logger.error(f"Error in get_session_history: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to get session history")


    def create_qa_prompt(self, filename) -> ChatPromptTemplate:
        """Create a QA prompt template for the document-based RAG chatbot."""
        try:
            logger.info(f"Creating QA prompt for: {filename}")

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

            logger.debug("QA prompt created successfully.")
            return qa_prompt

        except Exception as e:
            logger.error(f"Error in create_qa_prompt: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create QA prompt")
