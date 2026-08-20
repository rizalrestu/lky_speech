"""Does the generated answer stay inside the retrieved context?

The judge is a different model family from the one under test, so they don't
share blind spots.
"""
import os

import requests

from core import OLLAMA_URL, build_context, generate, load_resources, retrieve

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "phi4-mini:latest")

TEST_QUESTIONS = [
    "What is your view on democracy in developing countries?",
    "Why did Singapore separate from Malaysia?",
    "What is the role of the English language in Singapore?",
    "What is the best recipe for chocolate chip cookies?",  # must be refused
]


def llm_judge(context, answer):
    judge_prompt = f"""You are an evaluator.
Analyze the following Context and Answer.
Your task is to determine if the Answer is factually supported by the Context.
It is okay if the Answer uses a different persona or rephrases the text, as long as the core facts are in the Context.

Context: {context}

Answer: {answer}

First, write a 1-sentence analysis. Then, on a new line, you MUST output exactly:
VERDICT: YES (if supported) or VERDICT: NO (if not supported)."""

    resp = requests.post(OLLAMA_URL, json={
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": judge_prompt}],
        "stream": False,
        "options": {"num_ctx": 8192, "temperature": 0},
    }, timeout=600)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip().upper()


def evaluate():
    embedder, client = load_resources()
    grounded = 0
    try:
        for q in TEST_QUESTIONS:
            hits = retrieve(q, embedder, client)
            if not hits:
                print(f"[refused ] no context above threshold: {q}")
                grounded += 1
                continue

            context = build_context(hits)
            answer = generate(q, context)
            if "VERDICT: YES" in llm_judge(context, answer):
                print(f"[grounded] {q}")
                grounded += 1
            else:
                print(f"[HALLUCINATED] {q}\n    {answer[:300]}")
        print(f"\n{grounded}/{len(TEST_QUESTIONS)} grounded (judge: {JUDGE_MODEL})")
    finally:
        client.close()


if __name__ == "__main__":
    evaluate()
