"""The metric that actually matters: retrieval Recall@K and MRR on a small
hand-labeled test set.

Chunking + embedding quality can't really be judged in isolation — the real
question is "does the right passage come back when I ask a real question?"
This script answers that directly.

Requires embed_and_index.py to already have run (even just on a small subset
via `python chunk_markdown.py --limit N` while you're iterating quickly).

Edit TEST_SET below: a handful of real questions paired with the `uid` (from
data/records.jsonl) of the record that should answer them. Find uids by
searching titles, e.g.:
    grep -i "meritocracy" data/records.jsonl

Run:
    python test_retrieval.py
"""
import torch
import transformers  # noqa: F401  (import order workaround, see chat notes)
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_PATH = "data/qdrant_db"
COLLECTION = "lky_speeches"
EMBED_MODEL = "BAAI/bge-m3"
TOP_K = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (question, expected uid) — the uid should show up somewhere in the top-K
# results for that question if retrieval is working well.
TEST_SET = [
    # Political Study Centre seminar, 16 Aug 1964 — "The Concept of Democracy"
    ("Bagaimana pandangan LKY tentang konsep demokrasi?",
     "72f820c8-115d-11e3-83d5-0050568939ad"),

    # Book launch "Keeping My Mandarin Alive", 2 Jun 2005
    ("Apa yang LKY sampaikan soal belajar bahasa Mandarin?",
     "7a80add8-115d-11e3-83d5-0050568939ad"),

    # Bloomberg News interview Part II, 16 Sep 2005 — global & regional developments
    ("Apa pandangan LKY mengenai perkembangan geopolitik global dan regional?",
     "7daf8b9e-115d-11e3-83d5-0050568939ad"),

    # Joint meeting ASSOCHAM/FICCI/CII, New Delhi, 5 Jan 1996
    ("Apa pandangan LKY tentang hubungan ekonomi dan perdagangan dengan India?",
     "72f7ab37-115d-11e3-83d5-0050568939ad"),

    # Third Summit Conference of Non-Aligned Countries, Lusaka, 9 Sep 1970
    ("Bagaimana sikap LKY terhadap gerakan Non-Blok (Non-Aligned Movement)?",
     "727bc190-115d-11e3-83d5-0050568939ad"),

    # Keynote at official opening of the Lee Kuan Yew School of Public Policy, 4 Apr 2005
    ("Apa harapan LKY terhadap sekolah kebijakan publik yang dinamai dengan namanya?",
     "7911cb26-115d-11e3-83d5-0050568939ad"),

    # World Economic Forum Summit keynote, 20 Sep 1995
    ("Apa pandangan LKY tentang ekonomi dunia dan globalisasi?",
     "72f062e4-115d-11e3-83d5-0050568939ad"),

    # NYT/IHT interview with Seth Mydans, 1 Sep 2010
    ("Apa yang LKY katakan dalam wawancara dengan New York Times pada 2010?",
     "80e4563e-115d-11e3-83d5-0050568939ad"),

    # Narayana Murthy book launch "A Better India: A Better World", 11 May 2009
    ("Apa komentar LKY tentang perkembangan dan masa depan India?",
     "8023f350-115d-11e3-83d5-0050568939ad"),

    # SMU Tate Lecture Series, Dallas, 19 Oct 2006
    ("Apa yang LKY sampaikan kepada mahasiswa Amerika Serikat mengenai hubungan internasional?",
     "7ea6687b-115d-11e3-83d5-0050568939ad"),
]


def main():
    if not TEST_SET:
        print("TEST_SET masih kosong — isi dulu beberapa pasangan (pertanyaan, uid) di file ini.")
        return

    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    if DEVICE == "cuda":
        embedder = embedder.half()
    client = QdrantClient(path=QDRANT_PATH)

    hits, mrr_total = 0, 0.0
    for question, expected_uid in TEST_SET:
        qvec = embedder.encode([question], normalize_embeddings=True)[0]
        response = client.query_points(
            collection_name=COLLECTION,
            query=qvec.tolist(),
            limit=TOP_K,
            with_payload=True,
        )
        results = response.points
        uids = [h.payload.get("uid") for h in results]

        rank = uids.index(expected_uid) + 1 if expected_uid in uids else None
        if rank:
            hits += 1
            mrr_total += 1 / rank
        status = f"FOUND at rank {rank}" if rank else f"NOT FOUND in top-{TOP_K}"
        print(f"- {question!r}\n  expected uid={expected_uid} -> {status}")

    n = len(TEST_SET)
    print(f"\nRecall@{TOP_K}: {hits}/{n} ({hits/n:.0%})")
    print(f"MRR: {mrr_total/n:.3f}")


if __name__ == "__main__":
    main()
