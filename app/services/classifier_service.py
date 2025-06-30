# app/services/classifier_service.py
"""
This service loads the pre-trained SVC classifier and provides a function 
to predict the document type from a given text.
"""
import joblib
from pathlib import Path
import numpy as np

# Use relative imports for other services within the same 'app' package
from . import vector_service

# --- Configuration & Model Loading (on startup) ---
# Build a robust, absolute path to the models directory
# --- Configuration & Model Loading (on startup) ---
# --- THE FIX IS HERE ---
# Get the directory where this script ('classifier_service.py') is located
SCRIPT_DIR = Path(__file__).resolve().parent 
# Go UP one level from 'services' to 'app', then UP another level to the project root.
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# Now build the correct path to the models directory from the project root
MODELS_DIR = PROJECT_ROOT / "models"
# --- END OF FIX ---
SVC_MODEL_PATH = MODELS_DIR / "support_vector_machine_svc.joblib"
ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

# Load models and encoder when the application starts
try:
    CLASSIFIER = joblib.load(SVC_MODEL_PATH)
    LABEL_ENCODER = joblib.load(ENCODER_PATH)
    print("✅ Classifier and Label Encoder loaded successfully.")
except FileNotFoundError as e:
    raise RuntimeError(f"FATAL: A required model or encoder file was not found. Please run train_classifier.py. Details: {e}")

# --- Prediction Function ---
async def predict(text: str) -> tuple[str, float]:
    """
    Predicts the document type for a given text using the trained SVC model.

    Args:
        text: The text content of the document.

    Returns:
        A tuple containing the predicted label (str) and the confidence score (float).
    """
    if not text.strip():
        # If there's no text, we can't classify
        return "unknown", 0.0

    # 1. Get the embedding for the input text
    embedding = await vector_service.embed(text)
    embedding_array = np.array(embedding).reshape(1, -1) # Reshape for a single prediction

    # 2. Get the integer prediction from the scikit-learn pipeline
    prediction_encoded = CLASSIFIER.predict(embedding_array)
    
    # 3. Get the prediction probabilities to use as a confidence score
    probabilities = CLASSIFIER.predict_proba(embedding_array)
    confidence = float(np.max(probabilities))

    # 4. Decode the integer prediction back to a human-readable string label
    predicted_label = LABEL_ENCODER.inverse_transform(prediction_encoded)[0]

    return predicted_label, confidence