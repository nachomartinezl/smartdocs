#!/usr/bin/env python
"""
Organise raw_docs into:
dataset/
  train/{label}/...
  test/{label}/...

Also creates examples.jsonl for indexing.
"""
import shutil, hashlib, random, json, asyncio
from pathlib import Path
from collections import defaultdict
from app.services.vector_service import classify   # your earlier code
from app.services.document_ingestor import ingest  # gives list[dict]

ROOT_RAW    = Path("raw_docs")
ROOT_OUT    = Path("dataset")
TRAIN_PCT   = 0.8
MAX_FILES   = 1000
REVIEW_DIR  = ROOT_RAW / "_review"

random.seed(42)
REVIEW_DIR.mkdir(exist_ok=True)

async def auto_label(file_path: Path) -> str | None:
    "Return predicted label (>=0.6 confidence) or None."
    blocks = await ingest_dummy(file_path)   # sync wrapper below
    text   = "\n".join(b["text"] for b in blocks)
    label, conf = await classify(text)
    return label if conf >= 0.6 else None

def ingest_dummy(fp: Path):
    "Thin sync wrapper around async ingest() for CLI use."
    import io
    from fastapi import UploadFile
    data = fp.read_bytes()
    up   = UploadFile(filename=fp.name, file=io.BytesIO(data))
    up.content_type = ("application/pdf" if fp.suffix.lower() == ".pdf"
                       else "image/" + fp.suffix.strip("."))
    return asyncio.run(ingest(up))

async def main():
    files_by_label = defaultdict(list)

    for f in ROOT_RAW.rglob("*.*"):
        if f.parent.name == "_review" or f.is_dir():
            continue

        label = f.parent.name.lower()

        if label == "unknown":
            label = await auto_label(f) or "review"
            if label == "review":
                shutil.move(f, REVIEW_DIR / f.name)
                continue

        files_by_label[label].append(f)

    # ⚠️ Limit to MAX_FILES total, preserving label structure
    all_files = [(label, fp) for label, files in files_by_label.items() for fp in files]
    random.shuffle(all_files)
    all_files = all_files[:MAX_FILES]

    files_by_label = defaultdict(list)
    for label, fp in all_files:
        files_by_label[label].append(fp)

    # Organize into train/test and build JSONL
    out_records = []
    for label, files in files_by_label.items():
        random.shuffle(files)
        split_idx   = int(len(files) * TRAIN_PCT)
        train_files = files[:split_idx]
        test_files  = files[split_idx:]

        for subset, flist in (("train", train_files), ("test", test_files)):
            dest_dir = ROOT_OUT / subset / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            for fp in flist:
                dest = dest_dir / fp.name
                shutil.copy2(fp, dest)

                rec_id = hashlib.sha256(dest.read_bytes()).hexdigest()[:16]
                out_records.append({
                    "id":  rec_id,
                    "file_path": str(dest),
                    "label": label
                })

    with open(ROOT_OUT / "examples.jsonl", "w", encoding="utf-8") as fh:
        for rec in out_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✓ Done. {len(out_records)} docs organised.")
    if any(REVIEW_DIR.iterdir()):
        print(f"→ Check {REVIEW_DIR} for low-confidence docs.")

if __name__ == "__main__":
    asyncio.run(main())
