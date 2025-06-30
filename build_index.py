import json, io, asyncio, hashlib, pathlib, os
from asyncio import Semaphore, gather
from fastapi import UploadFile
from starlette.datastructures import Headers
from app.services.document_ingestor import ingest
from app.services.vector_service import index_examples

RAW_DOCS_DIR    = pathlib.Path("raw_docs")
EXAMPLES_JSONL  = pathlib.Path("dataset/examples.jsonl")
OCR_CACHE_DIR   = pathlib.Path(".ocr_cache")
OCR_CACHE_DIR.mkdir(exist_ok=True)
EXAMPLES_JSONL.parent.mkdir(exist_ok=True)

CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "32"))
SEM         = Semaphore(CONCURRENCY)
BATCH_SIZE  = 1000

# 🧠 Hash-based unique ID
def make_id(fp: pathlib.Path) -> str:
    return hashlib.sha256(fp.read_bytes()).hexdigest()[:16]

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

async def to_example(fp: pathlib.Path) -> dict | None:
    async with SEM:
        text = await extract_text(fp)
        if not text.strip():
            return None

        return {
            "id": make_id(fp),
            "file_path": str(fp),
            "label": fp.parent.name.lower(),
            "text": text
        }

def already_indexed_ids() -> set[str]:
    if not EXAMPLES_JSONL.exists():
        return set()
    return {
        json.loads(line)["id"]
        for line in EXAMPLES_JSONL.read_text(encoding="utf-8").splitlines()
    }

async def main():
    all_files = sorted([
        f for f in RAW_DOCS_DIR.rglob("*.*")
        if f.is_file() and f.parent.name != "_review"
    ])
    print(f"📂 Found {len(all_files)} total files")

    existing_ids = already_indexed_ids()
    print(f"🧠 Skipping {len(existing_ids)} already-indexed files")

    # Only keep new, uncached files
    files_to_process = [f for f in all_files if make_id(f) not in existing_ids]
    total_batches = (len(files_to_process) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(total_batches):
        start = i * BATCH_SIZE
        end   = start + BATCH_SIZE
        batch = files_to_process[start:end]
        if not batch:
            break

        print(f"\n🔥 Batch {i+1}/{total_batches} — {len(batch)} files")

        tasks = [to_example(fp) for fp in batch]
        examples = [ex for ex in await gather(*tasks) if ex]

        if not examples:
            print("⚠️ No new valid examples in this batch.")
            continue

        # 👉 Vector DB indexing
        print(f"📚 Indexing {len(examples)} to vector DB...")
        await index_examples(examples)

        # 👉 Append to examples.jsonl
        with EXAMPLES_JSONL.open("a", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"✅ Batch {i+1} done")

if __name__ == "__main__":
    asyncio.run(main())
