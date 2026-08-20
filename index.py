"""Embed data/chunks.jsonl and upsert it into Qdrant.

Point IDs come from chunk_id, so upserts are idempotent: re-running, stopping
early, or adding a speech never shifts anyone else's ID.
"""
import json
import uuid
from pathlib import Path

import torch
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from core import COLLECTION, EMBED_MODEL, get_client

CHUNKS_PATH = Path("data/chunks.jsonl")
BATCH_SIZE = 8  # sized for 4GB VRAM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# changing this renames every point
NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def point_id(chunk_id):
    return str(uuid.uuid5(NS, chunk_id))


def load_chunks():
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    if not CHUNKS_PATH.exists():
        print(f"{CHUNKS_PATH} not found — run chunk.py first")
        return

    rows = load_chunks()
    total = len(rows)
    print(f"loaded {total} chunks")

    client = get_client()

    if client.collection_exists(COLLECTION):
        # old collections numbered points by position; uuid5 upserts on top of
        # those would double the corpus instead of replacing it
        old, _ = client.scroll(COLLECTION, limit=1, with_payload=False, with_vectors=False)
        if old and isinstance(old[0].id, int):
            print(f"collection '{COLLECTION}' still uses positional integer IDs.")
            print("Indexing on top of it would duplicate every chunk, not replace it.")
            print("Drop the old collection first, then re-run this script:")
            print("  python -c \"from core import get_client, COLLECTION;"
                  " get_client().delete_collection(COLLECTION)\"")
            return

    print(f"loading embedding model {EMBED_MODEL} on {DEVICE} ...")
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    dim = model.get_embedding_dimension()

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

    done = skipped = 0
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        ids = [point_id(r["chunk_id"]) for r in batch]

        # resume without a progress file: ask the DB what it already has
        if len(client.retrieve(COLLECTION, ids=ids, with_payload=False)) == len(ids):
            skipped += len(batch)
            done += len(batch)
            continue

        vectors = model.encode([r["text"] for r in batch],
                               normalize_embeddings=True, show_progress_bar=False)
        client.upsert(
            collection_name=COLLECTION,
            points=[
                qmodels.PointStruct(id=ids[j], vector=vectors[j].tolist(), payload=batch[j])
                for j in range(len(batch))
            ],
        )
        done += len(batch)
        print(f"  indexed {done}/{total}")

    print(f"\ndone -> collection '{COLLECTION}' ({skipped} chunks already present, skipped)")


if __name__ == "__main__":
    main()
