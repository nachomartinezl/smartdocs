# generate_schemas.py
"""
Uses a powerful LLM to automatically generate a data extraction schema for each
document type by analyzing a few sample documents from a JSONL file.
"""
import os
import json
from pathlib import Path
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent 

TRAIN_EXAMPLES_JSONL = SCRIPT_DIR / "dataset" / "train_examples.jsonl"
ALL_EXAMPLES_JSONL = SCRIPT_DIR / "dataset" / "examples.jsonl" # Contains the full text for schema generation examples
OUTPUT_SCHEMAS_FILE = SCRIPT_DIR / "app" / "schemas.json"
NUM_EXAMPLES_PER_TYPE = 3
SCHEMA_GEN_MODEL = "gpt-4o"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OUTPUT_SCHEMAS_FILE.parent.mkdir(exist_ok=True)

# --- Logic ---

def get_document_types(train_jsonl_path: Path) -> list[str]:
    """Reads the training JSONL file to find all unique document labels."""
    if not train_jsonl_path.exists():
        raise FileNotFoundError(f"Training data file not found at: {train_jsonl_path}")
    
    with open(train_jsonl_path, 'r', encoding='utf-8') as f:
        labels = {json.loads(line).get('label') for line in f}
    # Remove None if any blank lines were processed and sort the list
    return sorted([label for label in labels if label])


def get_sample_texts(doc_type: str, all_examples_path: Path) -> list[str]:
    """Gets random sample texts for a given document type from the main JSONL file."""
    if not all_examples_path.exists():
        print(f"Warning: {all_examples_path} not found. Cannot get samples for {doc_type}.")
        return []

    with open(all_examples_path, 'r', encoding='utf-8') as f:
        all_records = [json.loads(line) for line in f]
    
    # Filter for records of the current document type that have text
    type_records = [
        r for r in all_records 
        if r.get('label') == doc_type and r.get('text', '').strip()
    ]
    
    if not type_records:
        return []
    
    sample_records = random.sample(type_records, min(len(type_records), NUM_EXAMPLES_PER_TYPE))
    # Truncate text to fit context window and avoid large costs
    texts = [r.get('text', '')[:3000] for r in sample_records]
    return texts


def generate_schema_for_type(doc_type: str, sample_texts: list[str]) -> dict:
    """Uses an LLM to generate a schema based on sample texts."""
    if not sample_texts:
        print(f"No sample texts provided for {doc_type}. Skipping.")
        return {}

    formatted_examples = "\n\n--- Next Example ---\n\n".join(sample_texts)
    
    prompt = f"""
You are an expert data architect. Your task is to define a JSON schema for extracting key information from a document.

The document type is: '{doc_type}'

Based on the following example texts from this document type, identify the most important, common, and useful fields to extract.

- Field names must be snake_case and descriptive (e.g., 'invoice_number', 'candidate_name').
- For each field, provide a brief, one-sentence description.
- Only include fields that are likely to be present in most documents of this type.
- Do not invent fields that are not supported by the text.

Respond with ONLY a valid JSON object. The JSON object should have a single key "fields", which is a dictionary of the fields you identified.

Example Texts:
---
{formatted_examples}
---

Your JSON response:
"""
    
    try:
        print(f"  > Sending request to {SCHEMA_GEN_MODEL} for schema...")
        response = client.chat.completions.create(
            model=SCHEMA_GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        schema_str = response.choices[0].message.content
        return json.loads(schema_str)
    except Exception as e:
        print(f"  > Error generating schema for {doc_type}: {e}")
        return {}


# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Get document types from the training JSONL file
        document_types = get_document_types(TRAIN_EXAMPLES_JSONL)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    all_schemas = {}

    if not document_types:
        print(f"Error: No document types found in {TRAIN_EXAMPLES_JSONL}")
        exit(1)

    print(f"Found {len(document_types)} document types. Starting schema generation...")

    for doc_type in document_types:
        print(f"\n🧠 Processing document type: '{doc_type}'")
        # Use the main examples file which should contain the OCR'd text
        texts = get_sample_texts(doc_type, ALL_EXAMPLES_JSONL)
        
        if not texts:
            print(f"  > Could not find any sample texts with content. Skipping.")
            continue

        schema = generate_schema_for_type(doc_type, texts)
        
        if schema and "fields" in schema and schema["fields"]:
            all_schemas[doc_type] = {
                "description": f"Schema for {doc_type} documents.",
                "fields": schema["fields"]
            }
            print(f"✅ Success! Generated schema with fields: {list(schema['fields'].keys())}")
        else:
            print(f"❌ Failed to generate a valid schema for {doc_type}.")

    with open(OUTPUT_SCHEMAS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_schemas, f, indent=2)
        
    print(f"\n\n🎉 All schemas have been generated and saved to: {OUTPUT_SCHEMAS_FILE}")