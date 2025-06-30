# train_classifier.py
"""
Train and evaluate multiple document-type classifiers using precomputed embeddings.
This script compares Logistic Regression, Random Forest, SVC, and XGBoost.
It now correctly handles label encoding for all models.
"""
import json
from pathlib import Path
import joblib
from collections import Counter
import time

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

from app.services.vector_service import get_vectors

# --- Configuration ---
# Paths
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "dataset"
TRAIN_FILE = DATA_DIR / "train_examples.jsonl"
TEST_FILE  = DATA_DIR / "test_examples.jsonl"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# --- Functions ---
def load_examples(path: Path) -> tuple[list[str], list[str]]:
    """Loads example IDs and labels from a JSONL file."""
    examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    ids    = [ex["id"] for ex in examples]
    labels = [ex["label"] for ex in examples]
    return ids, labels

def fetch_and_prepare_data(ids: list[str], labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fetches vectors and filters out any missing data, returning clean X and y."""
    print(f"🎯 Fetching {len(ids)} vectors...")
    raw_vectors = get_vectors(ids)

    if raw_vectors is None or len(raw_vectors) == 0:
        raise ValueError("get_vectors returned no data or an invalid result.")
        
    vector_map = {id: vec for id, vec in zip(ids, raw_vectors)}
    
    filtered_data = []
    for id, label in zip(ids, labels):
        vec = vector_map.get(id)
        if vec is not None:
            filtered_data.append((vec, label))
        else:
            print(f"⚠️ Missing embedding for ID {id}, skipping.")
            
    if not filtered_data:
        raise RuntimeError("No data remains after filtering for missing embeddings!")

    X, y = zip(*filtered_data)
    return np.array(X), np.array(y)

# --- Main Execution ---
if __name__ == "__main__":
    print("📂 Loading train and test examples...")
    train_ids, train_labels_str = load_examples(TRAIN_FILE)
    test_ids,  test_labels_str  = load_examples(TEST_FILE)

    print(f"🔢 Train samples: {len(train_ids)} | Test samples: {len(test_ids)}")
    print("-" * 30)

    X_train, y_train_str = fetch_and_prepare_data(train_ids, train_labels_str)
    X_test, y_test_str   = fetch_and_prepare_data(test_ids, test_labels_str)

    print(f"\nPrepared {len(X_train)} training samples and {len(X_test)} test samples.")
    print("-" * 30)

    # Encode string labels to integers for model training
    print("🏷️  Encoding string labels to integers...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_str)
    y_test_encoded = label_encoder.transform(y_test_str)
    
    # Save the encoder for later use (e.g., decoding predictions)
    encoder_path = MODELS_DIR / "label_encoder.joblib"
    joblib.dump(label_encoder, encoder_path)
    print(f"💾 Label encoder saved to {encoder_path}")
    print(f"Classes: {label_encoder.classes_}")

    # Define models to compare
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        ),
        "Random Forest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        ),
        "Support Vector Machine (SVC)": make_pipeline(
            StandardScaler(),
            SVC(random_state=42, probability=True) # probability=True is needed for some calibration/thresholding options
        ),
        "XGBoost": make_pipeline(
            StandardScaler(),
            # Now XGBoost will receive integer labels (0, 1, 2...)
            xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        )
    }

    # Loop, train, and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\n{'='*20}\n FITTING: {name} \n{'='*20}")
        start_time = time.time()
        
        # Fit the model using the ENCODED integer labels
        model.fit(X_train, y_train_encoded)
        train_time = time.time() - start_time
        
        # Get predictions (they will be integers)
        preds_encoded = model.predict(X_test)
        
        # We need the original string labels for the classification report
        acc = accuracy_score(y_test_encoded, preds_encoded)
        
        results[name] = {'accuracy': acc, 'train_time': train_time}
        
        print(f"✅ Test Accuracy: {acc:.4f}")
        print(f"⏱️ Training Time: {train_time:.2f} seconds")
        print("\n📊 Classification Report:")
        # Use the original string labels for a readable report
        print(classification_report(y_test_str, label_encoder.inverse_transform(preds_encoded), digits=3))

        model_path = MODELS_DIR / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        joblib.dump(model, model_path)
        print(f"💾 Model saved to {model_path}")
    
    # Print final summary
    print(f"\n{'='*30}\n🏆 FINAL MODEL COMPARISON 🏆\n{'='*30}")
    
    sorted_results = sorted(results.items(), key=lambda item: item[1]['accuracy'], reverse=True)
    
    print(f"{'Model':<30} | {'Accuracy':<10} | {'Train Time (s)':<15}")
    print(f"{'-'*30} | {'-'*10} | {'-'*15}")
    for name, metrics in sorted_results:
        print(f"{name:<30} | {metrics['accuracy']:.4f}     | {metrics['train_time']:.2f}")