import os
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from loguru import logger

class RAGService:
    
    @staticmethod
    def txt_to_text(txt_path: str) -> str:
        """Extracts text from a TXT file efficiently."""
        try:
            with open(txt_path, "r", encoding="utf-8") as file:
                content = file.read().strip()
                if not content:
                    logger.warning(f"Empty TXT file: {txt_path}")
                return content
        except FileNotFoundError:
            logger.error(f"TXT file not found: {txt_path}")
            return ""
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error in TXT file {txt_path}: {e}")
            return ""
        except Exception as e:
            logger.error(f"Error extracting TXT: {e}")
            return ""

    @staticmethod
    def pdf_to_text(pdf_path: str) -> str:
        """Efficiently extracts text from a PDF file."""
        try:
            with fitz.open(pdf_path) as pdf:
                if len(pdf) == 0:
                    logger.warning(f"Empty PDF file: {pdf_path}")
                    return ""
                text = "\n".join(page.get_text() for page in pdf if page.get_text().strip())
                if not text.strip():
                    logger.warning(f"No text content found in PDF: {pdf_path}")
                return text
        except FileNotFoundError:
            logger.error(f"PDF file not found: {pdf_path}")
            return ""
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            return ""

    @staticmethod
    def docx_to_text(docx_path: str) -> str:
        """Extracts text from a DOCX file efficiently."""
        try:
            doc = DocxDocument(docx_path)
            text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
            if not text.strip():
                logger.warning(f"No text content found in DOCX: {docx_path}")
            return text
        except FileNotFoundError:
            logger.error(f"DOCX file not found: {docx_path}")
            return ""
        except Exception as e:
            logger.error(f"Error extracting DOCX: {e}")
            return ""

    @staticmethod
    def get_text(file_path: str) -> str:
        """Extracts text based on the file format."""
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        if file_extension == ".txt":
            return RAGService.txt_to_text(file_path)
        elif file_extension == ".pdf":
            return RAGService.pdf_to_text(file_path)
        elif file_extension == ".docx":
            return RAGService.docx_to_text(file_path)
        else:
            logger.error("Unsupported file format.")
            raise ValueError("Unsupported file format. Only TXT, PDF, and DOCX are allowed.")
