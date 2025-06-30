import sys
import os
from pathlib import Path
import asyncio

# Fix import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.image_ocr_service import ocr_image

ASSETS_DIR = Path("tests/assets")

def get_all_images(directory: Path):
    return list(directory.rglob("*.jpg")) + list(directory.rglob("*.png"))

async def main():
    images = get_all_images(ASSETS_DIR)
    print(f"\n🔍 Found {len(images)} images for OCR...\n")

    for img_path in images:
        try:
            img_bytes = img_path.read_bytes()
            result = ocr_image(img_bytes)

            if result:
                print(f"\n✅ OCR extracted {len(result)} blocks from: {img_path.relative_to(ASSETS_DIR)}")
                for block in result[:5]:  # show max 5
                    print(f"- {block['text']} (conf={block['confidence']:.2f})")
            else:
                print(f"\n⚠️  No text detected in: {img_path.relative_to(ASSETS_DIR)}")
        except Exception as e:
            print(f"\n❌ Error processing {img_path.name}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
