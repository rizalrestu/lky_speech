"""Preview & sanity-check data/chunks.jsonl before spending time on embedding.

Shows:
- basic stats (total chunks, docs, chunks/doc, token-length distribution)
- flags suspiciously short chunks (likely OCR/pdftotext noise or empty pages)
- prints a handful of full sample chunks so you can eyeball the text quality

pip install transformers sentencepiece   (same tokenizer used by chunk_markdown.py)

Run:
    python preview_chunks.py               # random samples + stats
    python preview_chunks.py --n 10         # show 10 samples instead of 5
    python preview_chunks.py --uid <uid>    # show all chunks for one record
    python preview_chunks.py --short 50     # list chunks under 50 tokens
"""
import argparse
import json
import random
import statistics
from pathlib import Path

CHUNKS_PATH = Path("data/chunks.jsonl")
TOKENIZER_NAME = "BAAI/bge-m3"


def load_chunks():
    rows = []
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="number of random samples to print")
    parser.add_argument("--uid", type=str, default=None, help="show all chunks for this uid")
    parser.add_argument("--short", type=int, default=None,
                         help="list chunks with fewer than N tokens (likely noise)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not CHUNKS_PATH.exists():
        print(f"{CHUNKS_PATH} not found — run chunk_markdown.py first")
        return

    rows = load_chunks()
    print(f"total chunks: {len(rows)}")

    n_docs = len({r["uid"] for r in rows})
    print(f"total unique records (uid): {n_docs}")
    print(f"avg chunks/record: {len(rows) / max(n_docs, 1):.1f}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    lengths = [len(tokenizer.encode(r["text"], add_special_tokens=False)) for r in rows]
    print(f"token length — min: {min(lengths)}, max: {max(lengths)}, "
          f"mean: {statistics.mean(lengths):.0f}, median: {statistics.median(lengths):.0f}")

    if args.short is not None:
        short_rows = [(r, l) for r, l in zip(rows, lengths) if l < args.short]
        print(f"\n{len(short_rows)} chunk(s) under {args.short} tokens:")
        for r, l in short_rows[:30]:
            print(f"\n--- {r['chunk_id']} ({l} tokens) ---")
            print(r["text"])
        return

    if args.uid:
        matches = [r for r in rows if r["uid"] == args.uid]
        print(f"\n{len(matches)} chunk(s) for uid={args.uid}:")
        for r in matches:
            print(f"\n--- {r['chunk_id']} ---")
            print(f"title: {r['title']}")
            print(r["text"])
        return

    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))
    print(f"\n=== {len(sample)} random sample chunk(s) ===")
    for r in sample:
        print(f"\n--- {r['chunk_id']} ---")
        print(f"title:   {r['title']}")
        print(f"speaker: {r['speaker']} | date: {r['date']} | source: {r['source']}")
        print(f"text ({len(tokenizer.encode(r['text']))} tokens):")
        print(r["text"][:1500] + ("..." if len(r["text"]) > 1500 else ""))


if __name__ == "__main__":
    main()
