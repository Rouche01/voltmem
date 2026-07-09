"""
Contradiction demo — VoltMem vs always-add memory.
==================================================

Shows the product wedge: when facts change, ADD-only memory accumulates
contradictions; VoltMem keeps current truth.

Run:
    python examples/contradiction_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voltmem import create_memory  # noqa: E402


class AlwaysAddMemory:
    """Naive baseline: every statement is a new fact (Mem0 ADD-only style)."""

    def __init__(self) -> None:
        self._facts: list[str] = []

    def add(self, text: str) -> None:
        self._facts.append(text)

    def search(self, query: str, limit: int = 3) -> list[str]:
        q = set(query.lower().split())
        scored = []
        for fact in self._facts:
            words = set(fact.lower().split())
            overlap = len(q & words) / max(len(q), 1)
            scored.append((overlap, fact))
        scored.sort(reverse=True)
        return [f for _, f in scored[:limit]]


TURNS = [
    ("I live in Berlin", "where does the user live"),
    ("I prefer concise, direct answers", "how should I format replies"),
    ("I live in Paris now", "where does the user live"),
    ("I'm feeling pretty stressed this week", "how is the user feeling"),
    ("I really like short replies", "how should I format replies"),
]


def top_answer(mem, query: str) -> str:
    if isinstance(mem, AlwaysAddMemory):
        hits = mem.search(query, limit=1)
        return hits[0] if hits else "(nothing)"
    hits = mem.search(query, limit=1)
    return hits[0]["memory"] if hits else "(nothing)"


def location_facts(mem) -> list[str]:
    if isinstance(mem, AlwaysAddMemory):
        return [f for f in mem._facts if "live" in f.lower() or "paris" in f.lower() or "berlin" in f.lower()]
    return [m["memory"] for m in mem.get_all() if m["domain"] == "location"]


def main() -> None:
    naive = AlwaysAddMemory()
    volt = create_memory(":memory:", user_id="demo", embeddings=True)

    print("=" * 72)
    print("CONTRADICTION DEMO — VoltMem vs always-add memory")
    print("=" * 72)

    for i, (statement, probe) in enumerate(TURNS, 1):
        naive.add(statement)
        volt.add(statement)
        print(f"\n--- after turn {i}: {statement!r} ---")
        print(f"  Q: {probe}?")
        n = top_answer(naive, probe)
        v = top_answer(volt, probe)
        print(f"  always-add : {n}")
        print(f"  voltmem    : {v}")

    print("\n" + "=" * 72)
    print("VERDICT:")
    loc_naive = location_facts(naive)
    loc_volt = location_facts(volt)
    pref_naive = top_answer(naive, "how should I format replies")
    pref_volt = top_answer(volt, "how should I format replies")

    print(f"  Location facts stored — always-add: {len(loc_naive)}  voltmem: {len(loc_volt)}")
    if len(loc_naive) > len(loc_volt):
        print("    always-add keeps stale Berlin *and* Paris; VoltMem keeps one current fact.")
    print(f"  Preference query — always-add: {pref_naive!r}")
    print(f"                     voltmem:    {pref_volt!r}")
    if "concise" in pref_volt and "concise" not in pref_naive:
        print("    VoltMem protected a stable preference from a paraphrase blip.")
    print("=" * 72)


if __name__ == "__main__":
    main()
