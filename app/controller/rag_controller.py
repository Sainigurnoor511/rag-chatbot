import os
from ..services.rag_service import RAGService
from ..config.logger import logger
from ..config.settings import settings
from ..mypackage.myfunction import timer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma  # Updated import
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# In-memory session storage
SESSION_HISTORY = {}

GROQ_API_KEY = settings.GROQ_API_KEY

class RAGController:
    """Controller class to handle RAG API requests."""

    def __init__(self):
        """Initialize the RAG controller with LLM and memory."""
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            temperature=0.1,
            model_name="llama3-8b-8192",
        )

    @staticmethod
    @timer
    def logger_test():
        """Test logging levels."""
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        logger.critical("This is a critical message")
        return {"message": "Logs created successfully"}

    @staticmethod
    @timer
    def prepare_rag_documents(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """Splits the text and creates embeddings for RAG."""

        text = RAGService.get_text(file_path)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        docs = text_splitter.create_documents([text])

        # Create document objects with metadata
        formatted_docs = [Document(page_content=doc.page_content, metadata={"source": file_path}) for doc in docs]

        # Embedding model
        embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

        # Create vector store
        persist_directory = f"E:/Projects/rag-chatbot/data/database/{os.path.basename(file_path)}"
        os.makedirs(persist_directory, exist_ok=True)

        vectorstore = Chroma.from_documents(
            documents=formatted_docs,
            embedding=embed_model,
            persist_directory=persist_directory,
            collection_name="rag"
        )

        logger.info(f"Embeddings stored in {persist_directory}")
        return {"message": "Embeddings created successfully", "path": persist_directory}

    @staticmethod
    def chat_with_rag(file_path: str, user_input: str, session_id: str):
        """
        Handles chat with RAG and maintains session-based chat history.

        Args:
            - file_path: Path to the PDF/DOCX file used for retrieval.
            - user_input: The message/question from the user.
            - session_id: Unique session ID for maintaining history.

        Returns:
            - dict: Contains the user input and RAG model response.
        """

        # Initialize session history if it doesn't exist
        if session_id not in SESSION_HISTORY:
            SESSION_HISTORY[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

        # Retrieve session memory
        memory = SESSION_HISTORY[session_id]

        # Load existing vector store
        persist_directory = f"E:/Projects/rag-chatbot/data/database/{os.path.basename(file_path)}"
        
        # **Pass embedding model to prevent the "embedding function missing" error**
        embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

        vectorstore = Chroma(
            persist_directory=persist_directory,
            collection_name="rag",
            embedding_function=embed_model  # ✅ Pass embedding model
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # Retrieve relevant documents
        docs = retriever.invoke(user_input)  # ✅ Updated to use .invoke() instead of deprecated .get_relevant_documents()
        
        context = "\n\n".join([doc.page_content for doc in docs])

        # Add the current message to the session history
        memory.save_context({"input": user_input}, {"output": ""})

        # Prepare the prompt
        prompt = f"""
        You are a document-based chatbot.
        Use the following pieces of retrieved context to answer the user's question.
        If the context is empty, say you don't know.
        Keep the answer concise.
        
        Context:
        {context}
        
        Question: {user_input}
        """

        # Generate the LLM response
        response = RAGController().llm.predict(prompt)

        # Update memory with the LLM's output
        memory.save_context({"input": user_input}, {"output": response})

        return {
            "user_input": user_input,
            "bot_output": response,
            # "chat_history": memory.load_memory_variables({})["chat_history"]
        }
