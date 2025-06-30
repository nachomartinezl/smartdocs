import json, io, asyncio, hashlib, pathlib, os
from asyncio import Semaphore, gather
from fastapi import UploadFile
from starlette.datastructures import Headers
from app.services.document_ingestor import ingest
from app.services.vector_service     import index_examples

EXAMPLES_JSONL = pathlib.Path("dataset/examples.jsonl")
OCR_CACHE_DIR  = pathlib.Path(".ocr_cache")
OCR_CACHE_DIR.mkdir(exist_ok=True)

# ---------- helper utilities -----------------------------------------------

def mime_from_path(fp: pathlib.Path) -> str:
    ext = fp.suffix.lower()
    return "application/pdf" if ext == ".pdf" else f"image/{ext.lstrip('.')}"

def cache_path(fp: pathlib.Path) -> pathlib.Path:
    h = hashlib.sha256(fp.read_bytes()).hexdigest()[:20]
    return OCR_CACHE_DIR / f"{h}.txt"

async def extract_text(fp: pathlib.Path) -> str:
    """Run ingest() once per file, with a disk cache."""
    print(f"[INFO] Processing file: {fp}")
    cfile = cache_path(fp)
    if cfile.exists():
        print(f"[CACHE] Hit for: {fp.name}")
        return cfile.read_text(encoding="utf-8")

    try:
        up = UploadFile(
            file=io.BytesIO(fp.read_bytes()),
            filename=fp.name,
            headers=Headers({'content-type': mime_from_path(fp)})
        )

        blocks = await ingest(up)
        if not blocks:
            print(f"[WARN] No OCR blocks for: {fp}")
            return ""

        text = "\n".join(b["text"] for b in blocks if "text" in b and b["text"].strip())
        cfile.write_text(text, encoding="utf-8")
        print(f"[OK] OCR success for: {fp.name} — {len(text)} chars")
        return text

    except Exception as e:
        print(f"[ERROR] Exception during OCR for: {fp}\n{e}")
        return ""

# ---------- main routine ----------------------------------------------------

CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "32"))
SEM = Semaphore(CONCURRENCY)

async def to_example(rec: dict) -> dict | None:
    """Return embedding-ready record or None (if not in train)."""
    path = rec.get("file_path", "")
    if "/train/" not in path.replace("\\", "/"):
        print(f"[SKIP] Not a training file: {path}")
        return None

    async with SEM:
        fp = pathlib.Path(path)
        text = await extract_text(fp)
        if not text.strip():
            print(f"[SKIP] Empty OCR result: {fp.name}")
            return None

        print(f"[DONE] Ready to index: {fp.name}")
        return {"id": rec["id"], "text": text}

async def main():
    print(f"📄 Reading: {EXAMPLES_JSONL}")
    lines = EXAMPLES_JSONL.read_text(encoding="utf-8").splitlines()
    print(f"🔎 Found {len(lines)} files to process...")

    tasks = [to_example(json.loads(line)) for line in lines]
    examples = [ex for ex in await gather(*tasks) if ex]

    print(f"💾 Indexing {len(examples)} valid training examples...")
    await index_examples(examples)
    print(f"✅ Done. Indexed {len(examples)} examples (concurrency={CONCURRENCY})")

if __name__ == "__main__":
    asyncio.run(main())
