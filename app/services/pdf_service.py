# app/services/pdf_service.py
"""
Born-digital PDF extractor.
Receives raw PDF bytes and returns clean Markdown ready for RAG/LLM.

• Fast (no OCR) – relies on embedded text layer
• Preserves layout, images, tables
"""
from fastapi import HTTPException
import fitz                    # PyMuPDF
import pymupdf4llm


def extract_markdown(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return pymupdf4llm.to_markdown(doc, write_images=True)
    except Exception as exc:
        raise HTTPException(500, f"PDF parsing failed: {exc}") from exc
