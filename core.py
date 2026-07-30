"""Shared RAG logic"""
import os
import requests
import torch
import transformers
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_PATH = "data/qdrant_db"
QDRANT_URL = os.getenv("QDRANT_URL", None)
COLLECTION = "lky_speeches"
EMBED_MODEL = "BAAI/bge-m3"

DEVICE = "cpu"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = "qwen2.5:3b"
TOP_K = 5

SYSTEM_PROMPT = """You are Lee Kuan Yew, the founding father of Singapore. Answer the user's question STRICTLY and SOLELY based on the provided context.

CORE DIRECTIVES:
1. STRICT GROUNDING (NO HALLUCINATION): You MUST NOT use your internal knowledge. If the provided context does not explicitly contain the answer, you MUST say exactly: "I do not have any records or context regarding this matter." Do not attempt to guess, explain, or be a "know-it-all".
2. Persona: If the answer is found, adopt Lee Kuan Yew's signature pragmatic, blunt, and highly logical communication style.
3. Citations: If you answer, integrate brief citations naturally into your response (e.g., "As I stated in [Speech Title, Year]...").
"""

def load_resources():
    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    if DEVICE == "cuda":
        embedder = embedder.half()
        
    if QDRANT_URL:
        client = QdrantClient(url=QDRANT_URL)
    else:
        client = QdrantClient(path=QDRANT_PATH)
        
    return embedder, client


def retrieve(question, embedder, client, top_k=TOP_K):
    qvec = embedder.encode([question], normalize_embeddings=True)[0]
    response = client.query_points(
        collection_name=COLLECTION,
        query=qvec.tolist(),
        limit=top_k,
        with_payload=True,
    )
    return response.points


def build_context(hits):
    blocks = []
    for h in hits:
        p = h.payload
        header = f"[{p.get('title')} — {p.get('speaker')}, {p.get('date')}]"
        blocks.append(f"{header}\n{p.get('text')}")
    return "\n\n---\n\n".join(blocks)


def generate(question, context):
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"
    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"num_ctx": 4096} 
    }, timeout=180)

    resp.raise_for_status()
    return resp.json()["message"]["content"]


def answer_question(question, embedder, client, top_k=TOP_K):
    hits = retrieve(question, embedder, client, top_k=top_k)
    if not hits:
        return "I'm sorry, I couldn't find any relevant records regarding this matter.", hits
    context = build_context(hits)
    return generate(question, context), hits
