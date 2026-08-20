"""Ask one question from the terminal."""
import sys

print("Loading AI libraries, please wait...")
from core import answer_question, load_resources


def main():
    if len(sys.argv) < 2:
        print('usage: python cli.py "your question"')
        return
    question = " ".join(sys.argv[1:])

    print("loading embedder + qdrant ...")
    embedder, client = load_resources()

    answer, hits = answer_question(question, embedder, client)

    print("\n=== ANSWER ===\n")
    print(answer)

    print("\n=== SOURCES ===")
    for h in hits:
        p = h.payload
        print(f"- {p.get('title')} ({p.get('date')}) — score {h.score:.3f}")


if __name__ == "__main__":
    main()
