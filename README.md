# Intelligent Document Understanding API

## Overview

The Intelligent Document Understanding API is a powerful Python-based system designed to process, analyze, and extract structured information from various document types, including images and PDFs (both born-digital and scanned). It leverages a sophisticated pipeline involving OCR, document classification, and Large Language Model (LLM) based entity extraction.

The system can automatically determine the type of a document (e.g., invoice, resume, letter) and then extract relevant information based on dynamically generated or pre-defined schemas for that document type.

## Features

*   **Multi-Format Document Ingestion:** Supports common image formats (JPEG, PNG) and PDF files.
*   **Advanced OCR:** Utilizes PaddleOCR for text extraction from images and scanned PDFs.
*   **Smart PDF Processing:** Differentiates between born-digital PDFs (extracting text directly via PyMuPDF) and scanned PDFs (requiring OCR).
*   **Document Classification:** Employs a trained Support Vector Classifier (SVC) on text embeddings to automatically categorize documents.
*   **Dynamic Schema Generation:** Uses LLMs (e.g., GPT-4o) to generate data extraction schemas based on sample documents.
*   **LLM-Powered Entity Extraction:** Leverages LLMs (e.g., GPT-3.5 Turbo) to extract structured data from documents according to their classified type and corresponding schema.
*   **Vector-Based Indexing & Search:** Uses ChromaDB and OpenAI embeddings to create a searchable vector index of documents, enabling k-NN classification and other potential semantic search features.
*   **FastAPI Web Service:** Exposes its functionality through a robust and modern API built with FastAPI.
*   **Comprehensive Training Pipeline:** Includes scripts for dataset preparation, OCR, embedding generation, schema creation, and classifier training.

## Architecture Overview

The system follows a two-phase process:

1.  **Offline Training & Preparation Pipeline:**
    *   Raw documents are collected and organized.
    *   `prepare_dataset.py`: Organizes files, attempts auto-labeling for unknown types, and creates `dataset/examples.jsonl`.
    *   `build_index.py`: Performs OCR on documents, generates text embeddings (via OpenAI), and stores them with labels in ChromaDB.
    *   `split_dataset.py`: Splits `examples.jsonl` into training and testing sets (`train_examples.jsonl`, `test_examples.jsonl`).
    *   `generate_schemas.py`: Uses an LLM to create `app/schemas.json` for entity extraction based on document types found in the training data.
    *   `train_classifier.py`: Trains an SVC model (and others for comparison) using the embeddings and saves the best model and a label encoder to the `models/` directory.

2.  **Online API Runtime:**
    *   The FastAPI application serves an endpoint (`/extract_entities/`).
    *   **Ingestion:** Receives an uploaded document, extracts text (OCR if needed).
    *   **Classification:** Embeds the text and uses the trained SVC model to predict the document type.
    *   **Extraction:** Retrieves the appropriate schema for the document type and uses an LLM to extract structured data as JSON.
    *   **Response:** Returns the document type, classification confidence, and the extracted entities.

## Technology Stack

*   **Backend Framework:** FastAPI
*   **OCR Engine:** PaddleOCR (via `paddleocr`, `paddlepaddle-gpu`)
*   **PDF Processing:** PyMuPDF (`fitz`), `pymupdf4llm`
*   **Machine Learning:** Scikit-learn (`sklearn`) for classification
*   **LLM Interaction:** OpenAI API (`openai`)
*   **Text Embeddings:** OpenAI (e.g., `text-embedding-3-small`)
*   **Vector Database:** ChromaDB (`chromadb`)
*   **Programming Language:** Python 3
*   **Key Python Libraries:** Uvicorn, Pydantic, NumPy, OpenCV-Python-Headless, Joblib, python-dotenv

## Prerequisites

*   Python 3.8+
*   Access to OpenAI API (requires an API key)
*   Git (for cloning the repository)

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Environment Variables

Copy the example environment file and fill in your details, especially your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env` with your actual `OPENAI_API_KEY`. Other variables can be adjusted as needed:

```
# .env
OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY"
CHROMA_PATH="./chroma_db" # Directory for ChromaDB persistent storage
OCR_CONCURRENCY=4         # Concurrency for OCR in build_index.py
```

### 3. Python Dependencies

It's recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Note: `paddlepaddle-gpu` installation can be complex depending on your CUDA setup. For local setup, refer to PaddlePaddle's official installation guide.

## Dataset Preparation and Model Training Pipeline

This pipeline prepares your data, trains the classifier, generates schemas, and builds the vector index. **These steps must be run in order.**

**Important:**
*   Place your raw documents in the `raw_docs/` directory, organized by subfolders named after their respective labels (e.g., `raw_docs/invoice/doc1.pdf`, `raw_docs/resume/cv.png`).
*   For documents whose type is unknown, place them in `raw_docs/unknown/`. The `prepare_dataset.py` script will attempt to auto-label them.

**Running the Scripts:**

Run these scripts directly using Python from your activated virtual environment.

1.  **`prepare_dataset.py`**: Organizes raw documents, attempts auto-labeling, and creates `dataset/examples.jsonl`.
    ```bash
    python prepare_dataset.py
    ```
    *Check `raw_docs/_review/` for any documents that could not be confidently auto-labeled.*

2.  **`build_index.py`**: Performs OCR, generates embeddings, and populates ChromaDB. This can take a long time depending on the dataset size and OCR concurrency.
    ```bash
    python build_index.py
    ```

3.  **`split_dataset.py`**: Splits `dataset/examples.jsonl` into `train_examples.jsonl` and `test_examples.jsonl`.
    ```bash
    python split_dataset.py
    ```

4.  **`generate_schemas.py`**: Generates data extraction schemas using an LLM and saves them to `app/schemas.json`.
    ```bash
    python generate_schemas.py
    ```

5.  **`train_classifier.py`**: Trains the document classifier (SVC) and saves the model and label encoder to `models/`.
    ```bash
    python train_classifier.py
    ```

## Running the Application

Ensure all training steps are complete and you have an activated virtual environment.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*   `--reload`: Enables auto-reloading on code changes, useful for development.

The API will be available at `http://localhost:8000`.

## API Usage

### Endpoint: `POST /extract_entities/`

This endpoint accepts a file upload (image or PDF) and returns the classified document type, confidence score, and extracted entities.

**Request:**
*   Method: `POST`
*   URL: `/extract_entities/`
*   Body: `multipart/form-data` with a single field named `file` containing the document.

**Example using `curl`:**

```bash
curl -X POST "http://localhost:8000/extract_entities/" \
     -F "file=@/path/to/your/document.pdf"
```
Replace `/path/to/your/document.pdf` with the actual path to your test file.

**Sample Response (JSON):**

```json
{
  "document_type": "invoice",
  "confidence": 0.98765,
  "entities": {
    "invoice_number": "INV-2023-001",
    "vendor_name": "Supplier Corp",
    "total_amount": "150.75",
    "due_date": "2023-12-31"
    // ... other fields as per the schema for 'invoice'
  },
  "processing_time": "5.67s"
}
```
If OCR fails to extract text:
```json
{
  "document_type": "unknown",
  "confidence": 0.0,
  "entities": {
    "error": "OCR did not extract any text from the document."
  },
  "processing_time": "1.23s"
}
```
If no schema is found for a classified document type:
```json
{
  "document_type": "classified_type_without_schema",
  "confidence": 0.85,
  "entities": {
    "error": "No extraction schema is defined for document type: 'classified_type_without_schema'"
  },
  "processing_time": "2.34s"
}
```

## Directory Structure

```
.
├── app/                  # Main application code
│   ├── main.py           # FastAPI app definition and health check
│   ├── models/           # Pydantic models for API requests/responses (e.g., response.py)
│   ├── routers/          # API routers (e.g., extraction.py)
│   ├── services/         # Business logic (classifier, ingestor, llm, ocr, pdf, vector services)
│   ├── utils/            # Utility functions (e.g., preprocess.py)
│   └── schemas.json      # Auto-generated extraction schemas
├── dataset/              # Processed datasets
│   ├── examples.jsonl    # All processed documents metadata (path, label, id)
│   ├── train_examples.jsonl # Training set metadata
│   ├── test_examples.jsonl  # Test set metadata
│   ├── train/            # Training documents organized by label
│   └── test/             # Test documents organized by label
├── models/               # Trained ML models and encoders
│   ├── support_vector_machine_svc.joblib # Trained classifier
│   └── label_encoder.joblib            # Label encoder for document types
├── raw_docs/             # Place your raw documents here, organized by label
│   ├── invoice/
│   ├── resume/
│   ├── unknown/          # For documents with unknown types
│   └── _review/          # Documents needing manual review after auto-labeling
├── tests/                # Automated tests
│   ├── assets/           # Test documents
│   └── ...               # Test scripts (e.g., test_image_ocr.py)
├── .ocr_cache/           # Cache for OCR results to speed up re-runs
├── chroma_db/            # Persistent storage for ChromaDB vector index
├── .env                  # Local environment variables (ignored by Git)
├── .env.example          # Example environment variables file
├── requirements.txt      # Python dependencies
├── prepare_dataset.py    # Script to organize raw data and create examples.jsonl
├── build_index.py        # Script to OCR, embed, and index documents in ChromaDB
├── split_dataset.py      # Script to split examples.jsonl into train/test sets
├── generate_schemas.py   # Script to auto-generate extraction schemas
├── train_classifier.py   # Script to train the document classifier
└── README.md             # This file
```

## Testing

The `tests/` directory contains various tests for the application components. To run the tests (assuming `pytest` is the test runner, which is common):

1.  Ensure all dependencies, including development dependencies for testing, are installed.
2.  Navigate to the project root directory.
3.  Run:
    ```bash
    pytest
    ```
    (You might need to install pytest: `pip install pytest`)

Specific tests might require certain setup, like available test assets or a running ChromaDB instance if they test services interacting with it. Refer to individual test files for more details if needed.

## Future Enhancements / To-Do

*   More robust error handling and logging.
*   Support for more document formats.
*   Advanced schema management UI or API.
*   Fine-tuning embedding models for domain-specific vocabulary.
*   Integration with document management systems.
*   User authentication and authorization for the API.
*   More comprehensive test coverage.