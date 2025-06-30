import json, io, asyncio, hashlib, pathlib, os
from asyncio import Semaphore, gather
from fastapi import UploadFile
from starlette.datastructures import Headers
from app.services.document_ingestor import ingest
from app.services.vector_service import index_examples

EXAMPLES_JSONL = pathlib.Path("dataset/examples.jsonl")
OCR_CACHE_DIR  = pathlib.Path(".ocr_cache")
OCR_CACHE_DIR.mkdir(exist_ok=True)

CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "32"))
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

    try:
        up = UploadFile(
            file=io.BytesIO(fp.read_bytes()),
            filename=fp.name,
            headers=Headers({'content-type': mime_from_path(fp)})
        )
        blocks = await ingest(up)
        text = "\n".join(b["text"] for b in blocks if b.get("text", "").strip())
        cfile.write_text(text, encoding="utf-8")
        return text
    except Exception as e:
        print(f"❌ OCR failed on {fp.name}: {e}")
        return ""

async def to_example(rec: dict) -> dict | None:
    path = rec.get("file_path", "")
    if "/train/" not in path.replace("\\", "/"):
        return None

    async with SEM:
        fp = pathlib.Path(path)
        text = await extract_text(fp)
        if not text.strip():
            return None

        # **Include label** for metadata
        return {
            "id":    rec["id"],
            "text":  text,
            "label": rec["label"]
        }

async def main():
    lines   = EXAMPLES_JSONL.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    print(f"🧠 Loaded {len(records)} records")

    tasks    = [to_example(rec) for rec in records]
    examples = [ex for ex in await gather(*tasks) if ex]

    print(f"📚 Indexing {len(examples)} training examples...")
    await index_examples(examples)

if __name__ == "__main__":
    asyncio.run(main())
