import sys
import os
from pathlib import Path

# Fix import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.image_ocr_service import ocr_image

# Recursively find any image inside tests/assets/
ASSETS_ROOT = ROOT / "tests" / "assets"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def find_first_image():
    return next(
        (p for p in ASSETS_ROOT.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS),
        None
    )

def test_single_image_ocr():
    img_path = find_first_image()
    assert img_path, "❌ No image file found under tests/assets/."

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    result = ocr_image(img_bytes)

    assert isinstance(result, list), "❌ Result should be a list."
    assert all("text" in b and "bbox" in b and "confidence" in b for b in result), "❌ OCR output format invalid."

    print(f"✅ OCR extracted {len(result)} blocks from: {img_path.relative_to(ASSETS_ROOT)}")
    for block in result[:5]:
        print(f"- {block['text']} (conf={block['confidence']:.2f})")
