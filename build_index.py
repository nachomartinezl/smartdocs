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
    cfile = cache_path(fp)
    if cfile.exists():
        # --- CORRECTED LINE ---
        # Specify UTF-8 when reading the cache to handle all Unicode characters.
        return cfile.read_text(encoding="utf-8")

    up = UploadFile(
        file=io.BytesIO(fp.read_bytes()),
        filename=fp.name,
        headers=Headers({'content-type': mime_from_path(fp)})
    )

    blocks = await ingest(up)                       # OCR / PDF parse here
    text   = "\n".join(b["text"] for b in blocks)
    
    # --- CORRECTED LINE ---
    # Specify UTF-8 when writing to the cache to prevent UnicodeEncodeError.
    cfile.write_text(text, encoding="utf-8")
    
    return text

# ---------- main routine ----------------------------------------------------

CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "4"))  # raise if GPU has headroom
SEM = Semaphore(CONCURRENCY)

async def to_example(rec: dict) -> dict | None:
    """Return embedding-ready record or None (if not in train)."""
    if "/train/" not in rec["file_path"].replace("\\", "/"):
        return None
    async with SEM:
        fp   = pathlib.Path(rec["file_path"])
        text = await extract_text(fp)
        return {"id": rec["id"], "text": text, "label": rec["label"]}

async def main():
    # read jsonl
    tasks = []
    # This line already correctly uses UTF-8, which is great.
    for line in EXAMPLES_JSONL.read_text(encoding="utf-8").splitlines():
        tasks.append(to_example(json.loads(line)))

    examples = [ex for ex in await gather(*tasks) if ex]

    await index_examples(examples)            # store in Chroma
    print(f"✓ Indexed {len(examples)} training examples "
          f"(concurrency={CONCURRENCY})")

if __name__ == "__main__":
    asyncio.run(main())