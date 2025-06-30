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

# --- PPStructureV3 Model Initialization ---
ocr = PPStructureV3(
    text_recognition_model_name='en_PP-OCRv4_mobile_rec',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False, 
    use_textline_orientation=False, 
)
# -------- public API --------
def ocr_image(img_bytes: bytes) -> List[Dict[str, Any]]:
    try:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")

        result = ocr.predict(img)          # one page → len == 1
        if not result:
            logging.warning("OCR produced no result")
            return []

        # ---- JSON round-trip exactly like before ----
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "page.json"
            result[0].save_to_json(str(json_path))
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        blocks = data.get("parsing_res_list", [])
        if not blocks:
            logging.warning("No parsing_res_list found")
            return []

        extracted = []
        for blk in blocks:
            text  = (blk.get("block_content") or "").strip()
            bbox  = blk.get("block_bbox")    # [x1, y1, x2, y2]
            label = blk.get("block_label", "text")
            if not text or not bbox:
                continue

            x1, y1, x2, y2 = bbox
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            extracted.append(
                {
                    "type": label,
                    "text": text,
                    "bbox": poly,
                }
            )

        # sort blocks top-to-bottom
        extracted.sort(key=lambda b: b["bbox"][0][1])
        return extracted

    except Exception as exc:
        logging.exception("Image OCR failed")
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc
