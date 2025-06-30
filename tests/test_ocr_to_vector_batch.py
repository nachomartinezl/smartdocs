# tests/test_ocr_to_vector_batch.py

import sys
from pathlib import Path
import asyncio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.image_ocr_service import ocr_image
from app.services.vector_service import index_examples, client, COLLECTION

ASSETS_DIR = ROOT / "tests" / "assets"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def get_all_images():
    return list(ASSETS_DIR.rglob("*")) if ASSETS_DIR.exists() else []

def test_ocr_to_vector_batch():
    asyncio.run(_test_ocr_to_vector_batch())

async def _test_ocr_to_vector_batch():
    images = [p for p in get_all_images() if p.suffix.lower() in IMAGE_EXTENSIONS]
    assert images, "❌ No images found under assets."

    examples = []
    total_blocks = 0

    for path in images:
        try:
            img_bytes = path.read_bytes()
            ocr_result = ocr_image(img_bytes)
            texts = [b["text"] for b in ocr_result if b.get("text")]

            for i, text in enumerate(texts):
                examples.append({
                    "id": f"{path.stem}-{i}",
                    "text": text,
                    "metadata": {"label": "test"}
                })

            total_blocks += len(texts)
            print(f"✅ {len(texts)} blocks from {path.relative_to(ASSETS_DIR)}")
        except Exception as e:
            print(f"❌ Failed on {path.name}: {e}")

    assert examples, "❌ No OCR results to embed."
    await index_examples(examples)

    collection = client.get_collection(COLLECTION)
    ids = [ex["id"] for ex in examples]
    res = collection.get(ids=ids)

    assert len(res["ids"]) == len(examples), "❌ Not all examples indexed."

    # Cleanup
    collection.delete(ids=ids)
    print(f"\n🎯 Total embedded and indexed: {len(ids)}")
