# tests/test_ocr_to_vector_single.py

import sys
from pathlib import Path
import asyncio

# Setup sys path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.image_ocr_service import ocr_image
from app.services.vector_service import index_examples, client, COLLECTION

ASSETS_DIR = ROOT / "tests" / "assets"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_first_image():
    return next(
        (p for p in ASSETS_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS),
        None
    )

def test_ocr_to_vector_single():
    asyncio.run(_test_ocr_to_vector_single())

async def _test_ocr_to_vector_single():
    img_path = find_first_image()
    assert img_path, "❌ No image found under tests/assets."

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    ocr_result = ocr_image(img_bytes)
    assert isinstance(ocr_result, list) and len(ocr_result) > 0, "❌ OCR returned no valid blocks."

    texts = [b["text"] for b in ocr_result if b.get("text")]
    examples = [{"id": f"test-{i}", "text": t, "metadata": {"label": "test"}} for i, t in enumerate(texts)]

    await index_examples(examples)

    collection = client.get_collection(COLLECTION)
    ids = [ex["id"] for ex in examples]
    res = collection.get(ids=ids)

    assert len(res["ids"]) == len(examples), "❌ Not all examples indexed in Chroma."

    # Cleanup
    collection.delete(ids=ids)
    print(f"\n✅ OCR and vector index test passed for: {img_path.name}")
