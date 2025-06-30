import json, io, asyncio, pathlib, hashlib
from asyncio import Semaphore, gather
from fastapi import UploadFile
from starlette.datastructures import Headers
from sklearn.metrics import classification_report

from app.services.document_ingestor import ingest
from app.services.vector_service     import classify

# point at the same JSONL that build_index.py wrote
EXAMPLES_JSONL = pathlib.Path("dataset/examples.jsonl")
OCR_CACHE_DIR  = pathlib.Path(".ocr_cache")
OCR_CACHE_DIR.mkdir(exist_ok=True)

CONCURRENCY = 16
SEM = Semaphore(CONCURRENCY)

def mime_from_path(fp: pathlib.Path) -> str:
    ext = fp.suffix.lower()
    return "application/pdf" if ext == ".pdf" else f"image/{ext.lstrip('.')}"

def cache_path(fp: pathlib.Path) -> pathlib.Path:
    h = hashlib.sha256(fp.read_bytes()).hexdigest()[:20]
    return OCR_CACHE_DIR / f"{h}.txt"

async def extract_text(fp: pathlib.Path) -> str:
    cfile = cache_path(fp)
    if cfile.exists():
        return cfile.read_text(encoding="utf-8")

    up = UploadFile(
        file=io.BytesIO(fp.read_bytes()),
        filename=fp.name,
        headers=Headers({'content-type': mime_from_path(fp)})
    )
    blocks = await ingest(up)
    text = "\n".join(b["text"] for b in blocks if b.get("text", "").strip())
    cfile.write_text(text, encoding="utf-8")
    return text

async def to_prediction(rec: dict):
    # only test set
    if "/test/" not in rec["file_path"].replace("\\", "/"):
        return None

    async with SEM:
        fp = pathlib.Path(rec["file_path"])
        text = await extract_text(fp)
        if not text:
            return None
        pred, _ = await classify(text)
        return rec["label"], pred

async def main():
    # load all records
    lines   = EXAMPLES_JSONL.read_text(encoding="utf-8").splitlines()
    records = [json.loads(l) for l in lines]

    # run OCR + classify in parallel
    tasks   = [to_prediction(r) for r in records]
    results = [r for r in await gather(*tasks) if r]

    if not results:
        print("❌ No test examples found under /test/.")
        return

    y_true, y_pred = zip(*results)
    print("\n📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, digits=3))

if __name__ == "__main__":
    asyncio.run(main())
