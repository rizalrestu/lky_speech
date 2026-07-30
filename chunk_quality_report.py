"""Automated heuristic quality checks for data/chunks.jsonl.

There is no single canonical "chunk quality score". What this script measures
are proxy heuristics that catch obvious problems early. The metric that
actually matters — whether chunking retrieves the right passage for a real
question — can only be measured after embedding; see test_retrieval.py for
that.

Run:
    python chunk_quality_report.py
"""
import json
import statistics
from pathlib import Path

CHUNKS_PATH = Path("data/chunks.jsonl")


def load_chunks():
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def starts_mid_sentence(text):
    # heuristic: chunk starting with a lowercase letter was probably cut
    # mid-sentence by the token-window slicer
    stripped = text.lstrip()
    return bool(stripped) and stripped[0].islower()


def ends_mid_sentence(text):
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] not in ".!?\"'”)"


def garbage_ratio(text):
    if not text:
        return 1.0
    alnum_or_space = sum(c.isalnum() or c.isspace() for c in text)
    return 1 - (alnum_or_space / len(text))


def main():
    if not CHUNKS_PATH.exists():
        print(f"{CHUNKS_PATH} not found — run chunk_markdown.py first")
        return

    rows = load_chunks()
    n = len(rows)
    print(f"total chunks: {n}\n")

    mid_start = sum(starts_mid_sentence(r["text"]) for r in rows)
    mid_end = sum(ends_mid_sentence(r["text"]) for r in rows)
    print(f"chunks starting mid-sentence (lowercase start): {mid_start} ({mid_start/n:.1%})")
    print(f"chunks ending mid-sentence (no closing punctuation): {mid_end} ({mid_end/n:.1%})")
    print("  (some of this is expected/fine — chunks overlap by design. A HIGH")
    print("   number across the board, e.g. >50-60%, means the token-window")
    print("   slicer is cutting too aggressively; paragraph-aware chunking")
    print("   would read more naturally.)\n")

    garbage_scores = [garbage_ratio(r["text"]) for r in rows]
    noisy = [(r, g) for r, g in zip(rows, garbage_scores) if g > 0.15]
    print(f"chunks with >15% non-alphanumeric junk characters: {len(noisy)} ({len(noisy)/n:.1%})")
    if noisy:
        worst = max(noisy, key=lambda t: t[1])
        print(f"  worst offender: {worst[0]['chunk_id']} ({worst[1]:.0%} junk)")

    lengths = [len(r["text"].split()) for r in rows]
    print(f"\nword-count length — min {min(lengths)}, max {max(lengths)}, "
          f"mean {statistics.mean(lengths):.0f}, median {statistics.median(lengths):.0f}")
    print("  (compare against your CHUNK_TOKENS=700 setting — mean word count")
    print("   should land somewhere around 400-550 words for ~700 tokens,")
    print("   since 1 token ≈ 0.7-0.8 words for English/Indonesian text)")


if __name__ == "__main__":
    main()
