"""Retrieve -> build context -> generate."""
import os
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_PATH = "data/qdrant_db"
QDRANT_URL = os.getenv("QDRANT_URL", None)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION = "lky_speeches"
EMBED_MODEL = "BAAI/bge-m3"

DEVICE = "cpu"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = "qwen2.5:3b"

TOP_K = 5            # chunks that reach the prompt
FETCH_K = 8          # retrieved before dedup
MAX_PER_DOC = 2      # cap per speech
SCORE_THRESHOLD = 0.50   # gold chunks score 0.52-0.74, off-topic tops out at 0.47

NO_RECORDS = "I do not have any records or context regarding this matter."

SYSTEM_PROMPT = f"""You are Lee Kuan Yew, the founding father of Singapore. Answer the user's question STRICTLY and SOLELY based on the provided context.

CORE DIRECTIVES:
1. STRICT GROUNDING (NO HALLUCINATION): You MUST NOT use your internal knowledge. If the provided context does not explicitly contain the answer, you MUST say exactly: "{NO_RECORDS}" Do not attempt to guess, explain, or be a "know-it-all".
2. Persona: If the answer is found, adopt Lee Kuan Yew's signature pragmatic, blunt, and highly logical communication style.
3. Citations: If you answer, integrate brief citations naturally into your response (e.g., "As I stated in [Speech Title, Year]...").
"""


def get_client():
    """Embedded if QDRANT_URL is unset, server otherwise."""
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(path=QDRANT_PATH)


def load_resources():
    return SentenceTransformer(EMBED_MODEL, device=DEVICE), get_client()


def retrieve(question, embedder, client, top_k=TOP_K):
    qvec = embedder.encode([question], normalize_embeddings=True)[0]
    response = client.query_points(
        collection_name=COLLECTION,
        query=qvec.tolist(),
        limit=FETCH_K,
        with_payload=True,
        score_threshold=SCORE_THRESHOLD,
    )
    # neighbouring chunks overlap ~14% verbatim, so cap how many come from one speech
    kept, per_doc = [], {}
    for h in response.points:
        uid = h.payload.get("uid")
        if per_doc.get(uid, 0) >= MAX_PER_DOC:
            continue
        per_doc[uid] = per_doc.get(uid, 0) + 1
        kept.append(h)
        if len(kept) >= top_k:
            break
    return kept


def build_context(hits):
    blocks = []
    for h in hits:
        p = h.payload
        header = f"[{p.get('title')} — {p.get('speaker')}, {p.get('date')}]"
        blocks.append(f"{header}\n{p.get('text')}")
    return "\n\n---\n\n".join(blocks)


def generate(question, context):
    user_msg = (f"Context:\n{context}\n\nQuestion: {question}\n\n"
            "Answer in Lee Kuan Yew's first-person voice, using only the context above. "
            "Cite the speech title and year for each claim you make.")

    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        # if prompt + num_predict exceeds num_ctx, Ollama evicts the system prompt
        "options": {"num_ctx": 8192, "num_predict": 768, "temperature": 0},
    }, timeout=600)  # first call after boot loads the model

    resp.raise_for_status()
    data = resp.json()
    if "message" not in data:  # errors can arrive with HTTP 200
        raise RuntimeError(f"Ollama error: {data.get('error', data)}")
    return data["message"]["content"]


def answer_question(question, embedder, client, top_k=TOP_K):
    hits = retrieve(question, embedder, client, top_k=top_k)
    if not hits:
        return NO_RECORDS, hits
    return generate(question, build_context(hits)), hits
