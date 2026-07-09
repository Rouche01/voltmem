"""
Batteries-included quickstart — the simplest way to use VoltMem.
===============================================================

You hand VoltMem raw statements with remember(); it infers the domain, finds any
related existing memory, and lets the volatility engine decide whether to update
or keep it. You read memories back with recall(). No manual domain / mismatch.

Retrieval + matching use a real embedding model when one is available
(sentence-transformers or a local Ollama embed model), otherwise a deterministic
offline fallback so this always runs.

Run:
    .venv/bin/python examples/quickstart_batteries.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voltmem import MemoryLayer, EmbeddingSimilarity   # noqa: E402


def main():
    sim = EmbeddingSimilarity(verbose=True)
    mem = MemoryLayer(":memory:", similarity_fn=sim)

    print("\n--- remember() : one call, no domain/mismatch needed ---")
    statements = [
        "I live in Berlin",
        "I prefer concise, direct answers",
        "I'm working on the Postgres migration",
        "I feel pretty stressed this week",
        "Actually I moved to Paris last month",   # updates the location memory
        "I really like short, to-the-point replies",  # restates a stable pref
    ]
    for s in statements:
        r = mem.remember(s)
        print(f"  {r.action:<15} [{r.item.domain}]  {s!r}")

    print("\n--- recall() : fresh, relevant memories for a prompt ---")
    for q in ["Where does the user live?",
              "How should I format my answers?",
              "What are they working on?",
              "How are they feeling?"]:
        answers = mem.recall(q, top_k=1)
        print(f"  Q: {q}")
        print(f"     -> {answers[0] if answers else '(nothing relevant)'}")

    mem.close()


if __name__ == "__main__":
    main()
