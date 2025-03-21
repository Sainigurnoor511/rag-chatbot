import os
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from loguru import logger


class RAGService:
    
    @staticmethod
    def pdf_to_text(pdf_path: str) -> str:
        """Extracts text from a PDF file."""
        text = ""
        with fitz.open(pdf_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)
                text += page.get_text()
        return text

    @staticmethod
    def docx_to_text(docx_path: str) -> str:
        """Extracts text from a DOCX file."""
        doc = DocxDocument(docx_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    @staticmethod
    def get_text(file_path: str) -> str:
        """Extracts text based on the file format."""
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension == ".pdf":
            return RAGService.pdf_to_text(file_path)
        elif file_extension == ".docx":
            return RAGService.docx_to_text(file_path)
        else:
            raise ValueError("Unsupported file format. Only PDF and DOCX are allowed.")
