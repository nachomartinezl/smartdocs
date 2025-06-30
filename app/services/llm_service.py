# app/services/llm_service.py
"""
This service interfaces with a Large Language Model (LLM) for the purpose
of structured entity extraction based on a provided schema.
"""
import os
import json
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# --- THE FIX ---
# Get the directory of the current file (app/services/)
CURRENT_DIR = Path(__file__).resolve().parent
# The schemas file is located in the 'app' directory, which is the parent of 'services'.
SCHEMAS_FILE = CURRENT_DIR.parent / "schemas.json"
# --- END OF FIX ---
EXTRACTION_MODEL = "gpt-3.5-turbo" 
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    with open(SCHEMAS_FILE, "r", encoding="utf-8") as f:
        SCHEMAS = json.load(f)
    print("✅ Extraction schemas loaded successfully.")
except FileNotFoundError:
    raise RuntimeError(f"FATAL: schemas.json not found. Please run generate_schemas.py first.")

# --- Extraction Function ---
async def extract_entities(text: str, doc_type: str) -> dict:
    """
    Extracts structured data from text based on its document type and a predefined schema.

    Args:
        text: The full text of the document.
        doc_type: The predicted document type (e.g., 'invoice', 'resume').

    Returns:
        A dictionary containing the extracted entities.
    """
    # Find the schema for the predicted document type
    schema_info = SCHEMAS.get(doc_type)
    if not schema_info:
        return {"error": f"No extraction schema is defined for document type: '{doc_type}'"}

    # Get the list of fields we need to extract from the schema
    field_list = list(schema_info.get("fields", {}).keys())
    if not field_list:
        return {"error": f"Schema for '{doc_type}' contains no fields to extract."}

    # This is the generic prompt template that works for any document type
    prompt = f"""
Given the following text from a '{doc_type}' document, extract these specific fields: {field_list}.
If a field's value is not found in the text, use a null value for it.
Your response must be a single, valid JSON object, with no additional text or explanations.

Document Text:
---
{text}
---

JSON Output:
"""

    try:
        response = await client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Set to 0 for factual, deterministic output
            response_format={"type": "json_object"}  # Use OpenAI's JSON mode
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"LLM extraction failed for document type '{doc_type}': {e}")
        return {"error": f"The LLM API call failed: {e}"}