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
from fastapi import HTTPException, UploadFile

# --- tunables carried over from old ocr_service.py ---
SCAN_CHAR_THRESHOLD = 20    # < 20 chars across PDF? assume it's a scan
RASTER_DPI          = 300   # raster resolution for OCR
# -----------------------------------------------------

from app.services import pdf_service, image_ocr_service
from app.utils.preprocess import enhance_for_ocr


async def ingest(file: UploadFile) -> list[dict]:
    data   = await file.read()
    ctype  = file.content_type or ""
    blocks: list[dict] = []

    # ---------- images ----------
    if ctype.startswith("image/"):
        return image_ocr_service.ocr_image(data)           # already list[dict]

    # ---------- PDFs ----------
    if ctype == "application/pdf":
        doc = fitz.open(stream=data, filetype="pdf")

        # ➊ born-digital
        if sum(len(p.get_text("text").strip()) for p in doc) >= SCAN_CHAR_THRESHOLD:
            md = pdf_service.extract_markdown(data)
            blocks = [
                {"type": "text", "text": line, "bbox": None}
                for line in md.splitlines() if line.strip()
            ]
            return blocks

        # ➋ scan-PDF: raster → OCR
        for page in doc:
            pix  = page.get_pixmap(matrix=fitz.Matrix(RASTER_DPI/72, RASTER_DPI/72), alpha=False)
            img  = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
            pre  = enhance_for_ocr(img)
            blocks.extend(image_ocr_service.ocr_image(cv2.imencode(".png", pre)[1].tobytes()))

        return blocks

    raise HTTPException(400, f"Unsupported file type: {ctype}")

