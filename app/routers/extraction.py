# app/routers/extraction.py
"""
Extraction router
=================
POST /extract_entities/
-----------------------
1. Passes the uploaded file to `document_ingestor.ingest()`  
2. (future) Classifies the document via vector DB / embeddings  
3. (future) Sends text + doc-type to an LLM for entity extraction
"""
import time
from fastapi import APIRouter, UploadFile, File
from app.models.response import DocumentResponse
from app.services import document_ingestor as ing

router = APIRouter()

@router.post("/extract_entities/", response_model=DocumentResponse)
async def extract_entities_from_document(file: UploadFile = File(...)):
    start = time.time()

    # 1️⃣  Unified ingest → returns markdown (digital pdf) or plain text
    structured_blocks = await ing.ingest(file)

    entities_payload = {
        # The key "text_blocks" now holds the structured list from the OCR service.
        "text_blocks": structured_blocks,
        # We can also include a simple "full_text" version for easy viewing or simple use cases.
        "full_text": "\n".join([block.get('text', '') for block in structured_blocks])
    }

    # 2️⃣  TODO: add vector-based doc-type classification here
    doc_type   = "TBD"
    confidence = 0.0

    # 3️⃣  TODO: call LLM to pull structured entities

    return DocumentResponse(
        document_type = doc_type,
        confidence     = confidence,
        entities       = entities_payload,
        processing_time= f"{time.time() - start:.2f}s"
    )
