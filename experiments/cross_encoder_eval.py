"""
Cross-encoder as stage 2 — first matching suggestion from the surprise-detection chat
====================================================================================

Embeddings still recall. A MiniLM cross-encoder scores the pair. The threshold
is fitted on the 24-pair DEV split only, then frozen for held-out.

Pass (held-out 56): ≤2 false merges and ≥21/28 must-link, or close enough that
only a mid-band needs the LLM.

Fail: Python/Japanese still outranks career change. Stop. Do not tune.

Run:
    .venv/bin/python experiments/cross_encoder_eval.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voltmem.embeddings import EmbeddingSimilarity             # noqa: E402
from voltmem.verify import CrossEncoderVerifier, fit_score_threshold  # noqa: E402
from linking_eval import run_pairs                             # noqa: E402
from linking_pairs import SPLITS                               # noqa: E402

CAREER = (
    "User works as a data analyst",
    "User explicitly said they changed careers and now work as a nurse",
)
PYTHON_JA = (
    "User is proficient in Python",
    "User is proficient in Japanese",
)

PASS_MERGES = 2
PASS_LINKS = 21


def _score_split(ce: CrossEncoderVerifier, pos_pairs, neg_pairs):
    pos = ce.score_pairs([(st, nt) for _sd, st, _nd, nt, _n in pos_pairs])
    neg = ce.score_pairs([(st, nt) for _sd, st, _nd, nt, _n in neg_pairs])
    return pos, neg


def _report(name, pos, neg, pos_pairs, neg_pairs, t):
    linked = sum(1 for s in pos if s >= t)
    merged = sum(1 for s in neg if s >= t)
    print(f"  {name:<14} linked {linked:>2}/{len(pos)}   false merges {merged:>2}/{len(neg)}")
    misses = [
        (s, note) for s, (*_, note) in zip(pos, pos_pairs) if s < t
    ]
    kills = [
        (s, note) for s, (*_, note) in zip(neg, neg_pairs) if s >= t
    ]
    if misses:
        print("    missed links:")
        for s, note in misses:
            print(f"      {s:8.3f}  {note}")
    if kills:
        print("    false merges:")
        for s, note in kills:
            print(f"      {s:8.3f}  {note}")
    return linked, merged


def main():
    print("=" * 84)
    print("CROSS-ENCODER STAGE 2 — ms-marco-MiniLM-L-6-v2")
    print("=" * 84)

    t0 = time.perf_counter()
    ce = CrossEncoderVerifier()
    load_s = time.perf_counter() - t0
    print(f"  loaded {ce.model_name} in {load_s:.1f}s")

    dev_pos, dev_neg = SPLITS["dev (fitted)"]
    held_pos, held_neg = SPLITS["held-out"]

    t1 = time.perf_counter()
    dpos, dneg = _score_split(ce, dev_pos, dev_neg)
    hpos, hneg = _score_split(ce, held_pos, held_neg)
    n_pairs = len(dpos) + len(dneg) + len(hpos) + len(hneg)
    elapsed = time.perf_counter() - t1
    ms = (elapsed / n_pairs) * 1000.0 if n_pairs else 0.0
    print(f"  scored {n_pairs} pairs in {elapsed:.2f}s  ({ms:.1f} ms/pair)")

    career = ce.score(CAREER[1], CAREER[0])
    trap = ce.score(PYTHON_JA[1], PYTHON_JA[0])
    inverted = trap > career
    print("\n  ranking check (do not retune if inverted):")
    print(f"    career change (must-link)     {career:8.3f}")
    print(f"    Python vs Japanese (must-not) {trap:8.3f}")
    print(f"    trap outranks career? {inverted}")

    t, missed, merged = fit_score_threshold(dpos, dneg)
    ce.threshold = t
    print(f"\n  DEV-fitted threshold t={t:.4f}  "
          f"(missed {missed}/{len(dpos)}, merges {merged}/{len(dneg)})")
    print("  (frozen for held-out — no sweep there)")

    print("\n  pair scorer (no MemoryLayer, no recall):")
    _report("dev", dpos, dneg, dev_pos, dev_neg, t)
    h_link, h_merge = _report("held-out", hpos, hneg, held_pos, held_neg, t)

    print("\n  remember() path — embeddings recall @ 0.30 + CE verifier:")
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        sim = EmbeddingSimilarity(backend="sentence-transformers")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [skip] embeddings unavailable: {exc}")
        sim = None
    if sim is not None:
        pos = run_pairs(held_pos, sim, verifier=ce)
        neg = run_pairs(held_neg, sim, verifier=ce)
        linked = sum(1 for r in pos if r["linked"])
        merged_n = sum(1 for r in neg if r["linked"])
        print(f"  held-out remember()  linked {linked}/{len(pos)}   "
              f"false merges {merged_n}/{len(neg)}")
        for r in pos:
            if not r["linked"]:
                print(f"    miss  sim={r['sim']:.2f}  {r['note']}")
        for r in neg:
            if r["linked"]:
                print(f"    MERGE sim={r['sim']:.2f}  {r['note']}")

    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    pass_bar = (not inverted) and h_merge <= PASS_MERGES and h_link >= PASS_LINKS
    print(f"  held-out pair scorer: {h_link}/28 must-link, {h_merge}/28 false merges")
    print(f"  pass bar: ≥{PASS_LINKS}/28 links and ≤{PASS_MERGES} merges, "
          "and the trap must not outrank career change")
    if inverted:
        print("  FAIL — ranking inverted. Similarity-shaped scores again. Stop. Do not tune.")
    elif pass_bar:
        print("  PASS — cheap stage 2 exists. LLM can stay optional or grey-zone only.")
    else:
        print("  FAIL — not enough signal after a DEV-only cut. Do not sweep held-out.")
        print("         Next matching suggestion is structured extract-then-join.")


if __name__ == "__main__":
    main()
