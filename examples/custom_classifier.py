"""
Custom classifier + domain registry demo.
=======================================

Shows how to bring your own domain vocabulary (e.g. fashion / stylist)
without pre-classifying messages before ``add()``.

Run:
    python examples/custom_classifier.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voltmem import (  # noqa: E402
    ChainedClassifier,
    DomainRegistry,
    HeuristicClassifier,
    KeywordClassifier,
    create_memory,
)


def main() -> None:
    domains = DomainRegistry()
    domains.register("style_preference", 0.08)
    domains.register("style_constraint", 0.25)
    domains.register("session_occasion", 0.80, slot=True)

    classifier = ChainedClassifier([
        KeywordClassifier({
            "style_preference": ["prefer", "darker colors", "minimal", "loose fits"],
            "style_constraint": ["no wool", "tight budget", "must be formal"],
            "session_occasion": ["wedding", "job interview", "date night"],
        }),
        HeuristicClassifier(),
    ])

    with create_memory(
        ":memory:",
        user_id="stylist-demo",
        domains=domains,
        classifier=classifier,
        embeddings=False,
    ) as mem:
        turns = [
            "I prefer darker colors and minimal fits",
            "No wool — I'm allergic",
            "I'm dressing for a summer wedding",
        ]
        for text in turns:
            row = mem.add(text)
            print(f"  {row['action']:<15} [{row['domain']}]  {text!r}")

        print("\nSearch: what are the user's style preferences?")
        for hit in mem.search("style preferences colors fits", limit=3):
            print(f"  - {hit['memory']} (score={hit['score']})")


if __name__ == "__main__":
    main()
