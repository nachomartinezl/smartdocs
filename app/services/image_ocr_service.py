# app/services/image_ocr_service.py
"""
Image OCR service
-----------------
Bytes → A single block of ordered plain text using PPStructureV3.
This version uses the save/load JSON method for robust parsing.
"""
import cv2
import numpy as np
import logging
import json
import tempfile
from pathlib import Path
from paddleocr import PPStructureV3, PaddleOCR
from fastapi import HTTPException
from typing import List, Dict, Any

# --- PaddleOCR Model Initialization ---
ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False, # Disable document orientation classification
    use_doc_unwarping=False, # Disable text image unwarping
    use_textline_orientation=False, # Disable textline orientation classification
)
# -------- public API --------
def ocr_image(img_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Performs OCR and returns a structured list of text blocks by saving the
    result to a JSON file and then correctly parsing the top-level keys.
    """
    try:
        img_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image data, could not be decoded.")

        result = ocr.predict(input=img)

        if not result or not result[0]:
            logging.warning("OCR returned no result.")
            return []

        # Re-implementing the save/load JSON method as you instructed.
        data_dict = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_json_path = Path(temp_dir) / "result.json"
            result[0].save_to_json(save_path=str(temp_json_path))
            with open(temp_json_path, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)

        # --- THIS IS THE CORRECTED LOGIC ---

        # 1. Extract the recognized texts, polygons, and scores directly from the top level
        #    of the loaded dictionary, as shown in your latest error log.
        texts = data_dict.get('rec_texts')
        polys = data_dict.get('rec_polys')
        scores = data_dict.get('rec_scores')

        if not texts or not polys or not scores or len(texts) != len(polys):
            logging.warning(f"OCR data is inconsistent or empty in loaded JSON. Found keys: {data_dict.keys()}")
            return []

        # 2. Combine the data and sort by vertical position (top-left y-coordinate).
        combined_data = sorted(zip(polys, texts, scores), key=lambda item: item[0][0][1])

        # 3. Build the structured output that the router expects.
        structured_output = [
            {
                "type": "text",
                "text": text.strip(),
                "bbox": poly,
                "confidence": float(score)
            }
            for poly, text, score in combined_data
        ]

        return structured_output

    except Exception as exc:
        logging.exception("Image OCR process failed")
        raise HTTPException(status_code=500, detail=f"Image OCR failed: {exc}") from exc