"""Slice data/markdown into overlapping token windows -> data/chunks.jsonl.

Uses the embedding model's own tokenizer so chunk sizes match what gets embedded.
"""
import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

MD_DIR = Path("data/markdown")
OUT_PATH = Path("data/chunks.jsonl")

TOKENIZER_NAME = "BAAI/bge-m3"
CHUNK_TOKENS = 700
OVERLAP_TOKENS = 100


def parse_frontmatter(text):
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        return {}, text
    fm_raw, body = m.groups()
    meta = {}
    for line in fm_raw.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        meta[k.strip()] = v
    return meta, body.strip()


def chunk_tokens(tokenizer, text, size, overlap):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    step = size - overlap
    pieces = []
    start = 0
    while start < len(ids):
        window = ids[start:start + size]
        if not window:
            break
        pieces.append(tokenizer.decode(window, skip_special_tokens=True))
        if start + size >= len(ids):
            break
        start += step
    return pieces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N markdown files (for quick preview)")
    args = parser.parse_args()

    print(f"loading tokenizer {TOKENIZER_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    md_files = sorted(MD_DIR.rglob("*.md"))
    out_path = OUT_PATH
    if args.limit is not None:
        md_files = md_files[:args.limit]
        # never overwrite the real corpus
        out_path = OUT_PATH.with_suffix(".preview.jsonl")
    print(f"found {len(md_files)} markdown files (processing {len(md_files)})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_chunks = 0
    with out_path.open("w", encoding="utf-8") as out:
        for md_path in md_files:
            raw = md_path.read_text(encoding="utf-8")
            meta, body = parse_frontmatter(raw)
            # the "# Title" heading duplicates meta["title"]
            body = re.sub(r"^#\s.*\n+", "", body, count=1)

            for i, piece in enumerate(chunk_tokens(tokenizer, body, CHUNK_TOKENS, OVERLAP_TOKENS)):
                row = {
                    "chunk_id": f"{meta.get('uid', md_path.parent.name)}_{md_path.stem}_{i}",
                    "uid": meta.get("uid"),
                    "title": meta.get("title"),
                    "speaker": meta.get("speaker"),
                    "date": meta.get("date"),
                    "source": meta.get("source"),
                    "record_url": meta.get("record_url"),
                    "source_file": meta.get("source_file"),
                    "text": piece,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_chunks += 1

    print(f"wrote {n_chunks} chunks -> {out_path}")


if __name__ == "__main__":
    main()
