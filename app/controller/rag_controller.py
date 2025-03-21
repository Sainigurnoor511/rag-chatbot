import os
from ..services.rag_service import RAGService
from ..config.logger import logger
from ..mypackage.myfunction import timer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

class RAGController:
    """Controller class to handle RAG API requests."""

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

        # Embedding model #TODO : RND and select the best and fast model
        embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        # embed_model = FastEmbedEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

        # Create vector store #TODO : Change the path
        persist_directory = f"E:/Projects/rag-chatbot/data/database/vector_embeddings/{os.path.basename(file_path)}"
        os.makedirs(persist_directory, exist_ok=True)

        vectorstore = Chroma.from_documents(
            documents=formatted_docs,
            embedding=embed_model,
            persist_directory=persist_directory,
            collection_name="rag"
        )

        logger.info(f"Embeddings stored in {persist_directory}")
        return {"message": "Embeddings created successfully", "path": persist_directory}
