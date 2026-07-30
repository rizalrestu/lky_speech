"""Stage 5: chunk data/markdown/**/*.md into ~700-token pieces (100-token
overlap) for embedding, saved to data/chunks.jsonl.

Tokenized with the same tokenizer family as the embedding model (bge-m3) so
chunk sizes line up with what actually gets embedded later.

Each output row:
{
  "chunk_id": "<uid>_<pdf-stem>_<index>",
  "uid", "title", "speaker", "date", "source", "record_url", "source_file",
  "text": "<chunk text>"
}

pip install transformers sentencepiece

Run from the project root:
    python chunk_markdown.py
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
    if args.limit:
        md_files = md_files[:args.limit]
    print(f"found {len(md_files)} markdown files (processing {len(md_files)})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_chunks = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for md_path in md_files:
            raw = md_path.read_text(encoding="utf-8")
            meta, body = parse_frontmatter(raw)
            # drop the leading "# Title" heading added in to_markdown.py —
            # it's redundant with meta["title"] and would eat token budget
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

    print(f"wrote {n_chunks} chunks -> {OUT_PATH}")


if __name__ == "__main__":
    main()
