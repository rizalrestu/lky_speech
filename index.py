"""Stage 6: embed chunks with BAAI/bge-m3 and index them into a local Qdrant
collection (embedded/file-based mode — no server process needed).

Resumable: progress (how many chunks have been successfully indexed) is
tracked in data/embed_progress.json. Safe to stop with Ctrl+C at any time —
already-upserted chunks are not lost, and re-running this script picks up
right where it left off instead of starting over.

pip install qdrant-client sentence-transformers torch transformers

Run from the project root:
    python embed_and_index.py
"""
import json
from pathlib import Path

import torch
import transformers  # noqa: F401  (import order workaround, see chat notes)
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/chunks.jsonl")
QDRANT_PATH = "data/qdrant_db"
COLLECTION = "lky_speeches"
EMBED_MODEL = "BAAI/bge-m3"
BATCH_SIZE = 8  # keep small for 4GB VRAM GPUs; bump up if you have more headroom

PROGRESS_PATH = Path("data/embed_progress.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_chunks():
    rows = []
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_progress():
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8-sig"))["processed"]
    return 0


def save_progress(n):
    PROGRESS_PATH.write_text(json.dumps({"processed": n}), encoding="utf-8")


def main():
    if not CHUNKS_PATH.exists():
        print(f"{CHUNKS_PATH} not found — run chunk_markdown.py first")
        return

    rows = load_chunks()
    total = len(rows)
    print(f"loaded {total} chunks")

    client = QdrantClient(path=QDRANT_PATH)

    collection_exists = client.collection_exists(COLLECTION)
    start = load_progress() if collection_exists else 0

    if not collection_exists:
        print("no existing collection found — starting fresh")
        start = 0
    elif start >= total:
        print(f"already fully indexed ({start}/{total}) — nothing to do")
        return
    else:
        print(f"resuming from chunk {start}/{total}")

    print(f"loading embedding model {EMBED_MODEL} on {DEVICE} ...")
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    if DEVICE == "cuda":
        model = model.half()  # fp16: ~2x less VRAM, faster on Turing+ GPUs
    dim = model.get_sentence_embedding_dimension()

    if not collection_exists:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

    try:
        for i in range(start, total, BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            texts = [r["text"] for r in batch]
            vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

            client.upsert(
                collection_name=COLLECTION,
                points=[
                    qmodels.PointStruct(
                        id=i + j,
                        vector=vectors[j].tolist(),
                        payload=batch[j],
                    )
                    for j in range(len(batch))
                ],
            )
            done = min(i + BATCH_SIZE, total)
            save_progress(done)
            print(f"  indexed {done}/{total}")
    except KeyboardInterrupt:
        print("\nstopped early — progress saved, re-run this script later to resume")
        return

    print(f"\ndone -> collection '{COLLECTION}' stored at {QDRANT_PATH}")


if __name__ == "__main__":
    main()
