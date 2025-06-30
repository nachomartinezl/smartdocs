# app/services/document_ingestor.py
"""
Unified document ingestion layer.

UploadFile  -> str (markdown OR plain)

• Digital PDFs ............... PyMuPDF4LLM  (fast, 100 % accurate)
• Scan-PDFs (image-only) ..... Rasterise page → PaddleOCR
• Images (jpeg/png/bmp) ...... PaddleOCR
"""
from __future__ import annotations

import io, cv2, numpy as np, fitz
import asyncio # Import asyncio
from fastapi import HTTPException, UploadFile

# --- tunables carried over from old ocr_service.py ---
SCAN_CHAR_THRESHOLD = 20    # < 20 chars across PDF? assume it's a scan
RASTER_DPI          = 300   # raster resolution for OCR
# -----------------------------------------------------

from app.services import pdf_service, image_ocr_service
from app.utils.preprocess import enhance_for_ocr


def _process_scanned_pdf_page(page: fitz.Page) -> list[dict]:
    """Processes a single page of a scanned PDF for OCR."""
    pix = page.get_pixmap(matrix=fitz.Matrix(RASTER_DPI / 72, RASTER_DPI / 72), alpha=False)
    img_bytes = pix.tobytes("png")
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None: # Handle cases where image decoding might fail
        return []

    preprocessed_img = enhance_for_ocr(img)

    # Encode preprocessed image back to bytes for OCR service
    success, encoded_img_bytes = cv2.imencode(".png", preprocessed_img)
    if not success:
        return []

    return image_ocr_service.ocr_image(encoded_img_bytes.tobytes())


async def ingest(file: UploadFile) -> list[dict]:
    data   = await file.read()
    ctype  = file.content_type or ""
    blocks: list[dict] = []

    # ---------- images ----------
    if ctype.startswith("image/"):
        # Run synchronous OCR in a thread to avoid blocking the event loop
        return await asyncio.to_thread(image_ocr_service.ocr_image, data)

    # ---------- PDFs ----------
    if ctype == "application/pdf":
        try:
            # Run synchronous fitz.open in a thread
            doc = await asyncio.to_thread(fitz.open, stream=data, filetype="pdf")
        except Exception as e: # Handle potential fitz errors
            raise HTTPException(status_code=400, detail=f"Invalid or corrupted PDF file: {e}")

        # ➊ born-digital PDF
        # The text extraction sum can also be synchronous and potentially blocking
        page_texts = [p.get_text("text").strip() for p in doc]
        total_text_len = sum(len(text) for text in page_texts)

        if total_text_len >= SCAN_CHAR_THRESHOLD:
            # Run synchronous markdown extraction in a thread
            md = await asyncio.to_thread(pdf_service.extract_markdown, data)
            blocks = [
                {"type": "text", "text": line, "bbox": None} # bbox is None for markdown
                for line in md.splitlines() if line.strip()
            ]
            doc.close()
            return blocks

        # ➋ scan-PDF: raster → OCR
        for page_num in range(len(doc)):
            try:
                # doc.load_page is synchronous
                page = await asyncio.to_thread(doc.load_page, page_num)
                # _process_scanned_pdf_page is synchronous and CPU-bound
                page_blocks = await asyncio.to_thread(_process_scanned_pdf_page, page)
                blocks.extend(page_blocks)
            except Exception as e: # Log error for a specific page and continue
                print(f"Warning: Failed to process page {page_num + 1} of PDF '{file.filename}': {e}")
                continue # Optionally, add a block indicating page processing failure

        doc.close()
        return blocks

    raise HTTPException(400, f"Unsupported file type: {ctype}")

