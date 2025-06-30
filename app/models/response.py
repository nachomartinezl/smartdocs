    # app/models/response.py
from pydantic import BaseModel, Field

class DocumentResponse(BaseModel):
    document_type: str = Field(..., description="Detected document type")
    confidence: float = Field(..., ge=0, le=1, description="Classification confidence")
    entities: dict = Field(..., description="Extracted entities")
    processing_time: str = Field(..., description="Total processing duration")