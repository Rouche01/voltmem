#!/usr/bin/env python3
"""Print HeuristicClassifier accuracy on the classification corpus.

Usage::

    python experiments/classification_baseline.py
    python experiments/classification_baseline.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voltmem import HeuristicClassifier  # noqa: E402
from voltmem.classification_eval import (  # noqa: E402
    evaluate_classifier,
    format_report,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable summary instead of the text report",
    )
    args = ap.parse_args()

    clf = HeuristicClassifier()
    result = evaluate_classifier(clf.classify_domain)

    if args.json:
        payload = {
            "n": result.n,
            "correct": result.correct,
            "accuracy": round(result.accuracy, 4),
            "by_domain": {
                d: {
                    "n": s["n"],
                    "correct": s["correct"],
                    "accuracy": round(s["correct"] / s["n"], 4) if s["n"] else 0.0,
                }
                for d, s in sorted(result.by_domain.items())
            },
            "collision_misses": len(result.collisions),
            "n_errors": len(result.errors),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
