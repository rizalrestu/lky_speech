import time
import requests
from core import load_resources, retrieve, build_context, generate

OLLAMA_URL = "http://localhost:11434/api/chat"
JUDGE_MODEL = "qwen2.5:3b"

embedder, client = load_resources()

TEST_QUESTIONS = [
    "What is your view on democracy in developing countries?",
    "Why did Singapore separate from Malaysia?",
    "What is the role of the English language in Singapore?"
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
        "options": {"num_ctx": 4096, "temperature": 0.0}
    }, timeout=180)
    
    return resp.json()["message"]["content"].strip().upper()

def evaluate():
    for q in TEST_QUESTIONS:
        hits = retrieve(q, embedder, client)
        context = build_context(hits)
        answer = generate(q, context)
        
        verdict = llm_judge(context, answer)
        
        if "VERDICT: YES" in verdict:
            pass
        else:
            print(f"Hallucination detected for question: {q}")
            
    client.close()

if __name__ == "__main__":
    evaluate()
