"""
Synthetic Problem 3 probe — similarity plateau vs clear gap.

Constructs near-equal candidate similarities (under-specified query shape) and
shows that freshness_mix dampens V_d·staleness so cross-domain score gaps shrink.
Also checks a clear similarity gap keeps mix=1 (full freshness).

Run:
    python experiments/retrieval_plateau_probe.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voltmem import MemoryLayer  # noqa: E402
from voltmem.scoring import (  # noqa: E402
    MIX_MIN,
    freshness_mix,
    retrieval_score,
    similarity_spread,
)


def main() -> None:
    now = time.time()
    old = now - 14 * 86400
    sims_flat = [0.705, 0.700]
    sims_clear = [0.90, 0.40]
    spread_flat = similarity_spread(sims_flat)
    spread_clear = similarity_spread(sims_clear)
    mix_flat = freshness_mix(spread_flat)
    mix_clear = freshness_mix(spread_clear)

    print("=" * 60)
    print("Retrieval plateau probe (Problem 3)")
    print("=" * 60)
    print(f"  flat sims {sims_flat} → spread={spread_flat:.3f} mix={mix_flat:.3f}")
    print(f"  clear sims {sims_clear} → spread={spread_clear:.3f} mix={mix_clear:.3f}")
    assert mix_flat == MIX_MIN
    assert mix_clear == 1.0

    def plateau_sim(query, content):
        return {"volatile fact": 0.705, "stable fact": 0.700}.get(content, 0.0)

    with MemoryLayer(":memory:", similarity_fn=plateau_sim) as mem:
        mem.write("volatile fact", domain="current_task", at_time=old)
        mem.write("stable fact", domain="core_preference", at_time=old)
        for it in mem._active():
            it.last_confirmed_at = old
            it.created_at = old
            mem._store.update(it)
        result = mem.retrieve("what was I working on", top_k=2, now=now)
        print("\n  plateau retrieve ranking:")
        for item, score in zip(result.items, result.scores):
            print(f"    {score:.4f}  [{item.domain}] {item.content}")

        vol = next(i for i in mem._active() if "volatile" in i.content)
        gap_full = abs(
            retrieval_score(vol, 0.705, now=now, mix=1.0)
            - retrieval_score(
                next(i for i in mem._active() if "stable" in i.content),
                0.700, now=now, mix=1.0,
            )
        )
        gap_damp = abs(
            retrieval_score(vol, 0.705, now=now, mix=MIX_MIN)
            - retrieval_score(
                next(i for i in mem._active() if "stable" in i.content),
                0.700, now=now, mix=MIX_MIN,
            )
        )
        print(f"\n  score gap full mix={gap_full:.4f}  dampened={gap_damp:.4f}")
        assert gap_damp < gap_full
        print("\n  PASS. Plateau dampening shrinks freshness-driven gaps; "
              "clear gaps keep mix=1.")


if __name__ == "__main__":
    main()
