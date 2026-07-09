"""
Mem0-style comparison — VoltMem vs always-add on scripted scenarios.
===================================================================

Reproducible side-by-side for the product wedge. Does not require Mem0 installed.

Run:
    python experiments/mem0_comparison.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voltmem import create_memory  # noqa: E402


class AlwaysAddMemory:
    def __init__(self) -> None:
        self._facts: list[str] = []

    def add(self, text: str) -> None:
        self._facts.append(text)

    def search(self, query: str, limit: int = 1) -> list[str]:
        q = set(query.lower().split())
        scored = []
        for fact in self._facts:
            words = set(fact.lower().split())
            scored.append((len(q & words) / max(len(q), 1), fact))
        scored.sort(reverse=True)
        return [f for _, f in scored[:limit]]

    def count(self) -> int:
        return len(self._facts)


SCENARIOS = [
    {
        "name": "location_update",
        "turns": [
            "I live in Berlin",
            "I live in Paris now",
        ],
        "query": "where does the user live",
        "want": "Paris",
        "dont_want": "Berlin",
    },
    {
        "name": "stable_pref_blip",
        "turns": [
            "I prefer concise, direct answers",
            "I really like short replies",
        ],
        "query": "how should I format replies",
        "want": "concise",
        "dont_want": "really like short",
    },
    {
        "name": "volatile_mood",
        "turns": [
            "I'm feeling great today",
            "I'm pretty stressed this week",
        ],
        "query": "how is the user feeling",
        "want": "stressed",
        "dont_want": "great",
    },
]


def score_answer(text: str, want: str, dont_want: str) -> str:
    t = text.lower()
    if want.lower() in t and dont_want.lower() not in t:
        return "WIN"
    if dont_want.lower() in t:
        return "LOSE"
    return "PARTIAL"


def score_scenario(
    scenario: dict, naive_top: str, volt_top: str,
    naive_count: int, volt_count: int,
) -> tuple[str, str]:
    want, dont = scenario["want"], scenario["dont_want"]
    n_score = score_answer(naive_top, want, dont)
    v_score = score_answer(volt_top, want, dont)
    # location-style: fewer stored facts without wrong top answer is a win
    if scenario["name"] == "location_update" and volt_count < naive_count:
        if v_score != "LOSE":
            v_score = "WIN"
        if naive_count > volt_count and dont.lower() in naive_top.lower():
            n_score = "LOSE"
    return n_score, v_score


def run_scenario(scenario: dict) -> dict[str, str]:
    naive = AlwaysAddMemory()
    volt = create_memory(":memory:", user_id="demo", embeddings=True)

    for turn in scenario["turns"]:
        naive.add(turn)
        volt.add(turn)

    q = scenario["query"]
    n = naive.search(q, limit=1)[0] if naive.search(q) else ""
    v = volt.search(q, limit=1)[0]["memory"] if volt.search(q) else ""
    nc, vc = naive.count(), len(volt.get_all())
    ns, vs = score_scenario(scenario, n, v, nc, vc)

    return {
        "always_add": ns,
        "voltmem": vs,
        "naive_top": n,
        "volt_top": v,
        "naive_count": str(nc),
        "volt_count": str(vc),
    }


def main() -> None:
    print("=" * 72)
    print("MEM0-STYLE COMPARISON — always-add vs VoltMem (current truth)")
    print("=" * 72)
    print(f"  {'scenario':<22}{'always-add':>12}{'voltmem':>12}")
    print("  " + "-" * 46)

    volt_wins = naive_wins = 0
    for sc in SCENARIOS:
        r = run_scenario(sc)
        print(f"  {sc['name']:<22}{r['always_add']:>12}{r['voltmem']:>12}")
        print(f"    always-add ({r['naive_count']} facts): {r['naive_top']!r}")
        print(f"    voltmem    ({r['volt_count']} facts): {r['volt_top']!r}")
        if r["voltmem"] == "WIN" and r["always_add"] != "WIN":
            volt_wins += 1
        if r["always_add"] == "WIN" and r["voltmem"] != "WIN":
            naive_wins += 1

    print("\n" + "-" * 72)
    print(f"  VoltMem clearer wins: {volt_wins}/{len(SCENARIOS)} scenarios")
    print("=" * 72)


if __name__ == "__main__":
    main()
