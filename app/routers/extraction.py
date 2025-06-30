# app/routers/extraction.py
"""
Extraction router
=================
POST /extract_entities/
-----------------------
1. Passes the uploaded file to `document_ingestor.ingest()`  
2. Classifies the document using the trained SVC model.
3. Sends text + doc-type to an LLM for entity extraction.
"""
import time
from fastapi import APIRouter, UploadFile, File
from app.models.response import DocumentResponse

# Import the services that will do the work
from app.services import document_ingestor as ing
from app.services import classifier_service
from app.services import llm_service

router = APIRouter()

@router.post("/extract_entities/", response_model=DocumentResponse)
async def extract_entities_from_document(file: UploadFile = File(...)):
    start = time.time()

    # 1️⃣  Unified ingest -> returns text blocks and full text
    structured_blocks = await ing.ingest(file)
    full_text = "\n".join([block.get('text', '') for block in structured_blocks])

    # Handle cases where OCR fails to produce text
    if not full_text.strip():
        return DocumentResponse(
            document_type="unknown",
            confidence=0.0,
            entities={"error": "OCR did not extract any text from the document."},
            processing_time=f"{time.time() - start:.2f}s"
        )

    # 2️⃣  Classify the document type using our trained SVC model
    doc_type, confidence = await classifier_service.predict(full_text)

    # 3️⃣  Call the LLM to pull structured entities based on the predicted type
    extracted_entities = await llm_service.extract_entities(full_text, doc_type)

    # 4️⃣ Assemble the final response with all the pieces
    return DocumentResponse(
        document_type=doc_type,
        confidence=confidence,
        entities=extracted_entities, # This now holds the structured JSON from the LLM
        processing_time=f"{time.time() - start:.2f}s"
    )