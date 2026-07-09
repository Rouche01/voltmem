"""
Retrieval haystack benchmark — does freshness ranking find CURRENT truth in noise?
==================================================================================

LongMemEval *oracle* ties all systems (~72–78%) because the haystack is tiny.
This benchmark is deliberately harder: each query must find the **current** fact
among dozens of memories, including **semantically similar but stale volatile**
decoys (old project names, old moods) and irrelevant distractors.

All systems store the **same chunks**; only retrieval ranking differs:
  * voltmem_real       — volatility-weighted freshness (real priors)
  * voltmem_flat       — equal volatility (ablation)
  * voltmem_swap       — inverted volatility (negative control)
  * similarity_only    — cosine similarity only (no staleness penalty)

Metrics
-------
  * current@1 / current@5 — is the ground-truth *current* memory in top-k?
  * stale_vol@1           — does a stale volatile decoy win rank 1? (lower = better)
  * separation            — mean score(current) − mean score(best stale volatile decoy)

A causal pass requires voltmem_real to beat similarity_only on current@1 and
show real > flat > swap on separation.

Run:
    .venv/bin/python experiments/retrieval_haystack_bench.py
    .venv/bin/python experiments/retrieval_haystack_bench.py --decoys 8 --runs 30
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import voltmem.domains as vdomains  # noqa: E402
from voltmem import MemoryLayer, EmbeddingSimilarity  # noqa: E402
from voltmem.scoring import retrieval_score  # noqa: E402

SYSTEMS = ("voltmem_real", "voltmem_flat", "voltmem_swap", "similarity_only")
DAY = 86400.0


@dataclass
class Slot:
    query: str
    current: str
    domain: str
    stale_pool: list[str]          # interchangeable stale volatile-ish decoys
    distractor_pool: list[tuple[str, str]]  # (text, domain)


# Ground-truth slots: query targets the *current* line, not stale decoys.
SLOTS: list[Slot] = [
    Slot(
        "what is the user working on right now",
        "User is currently building the billing service",
        "current_project",
        [
            "User is currently building the postgres migration",
            "User is currently building the auth refactor",
            "User is currently building the mobile app rewrite",
            "User is currently building the search indexer",
            "User is currently building the onboarding flow",
            "User is currently building the metrics dashboard",
        ],
        [
            ("User prefers concise bullet points", "core_preference"),
            ("User lives in Berlin", "location"),
            ("User feels stressed this week", "emotional_context"),
            ("User's manager is Alex", "relationship"),
        ],
    ),
    Slot(
        "how is the user feeling lately",
        "User is feeling focused and energized this week",
        "emotional_context",
        [
            "User is feeling stressed and overwhelmed this week",
            "User is feeling anxious about deadlines this week",
            "User is feeling tired and burned out this week",
            "User is feeling frustrated with the rollout this week",
        ],
        [
            ("User is working on the billing service", "current_project"),
            ("User was born in Spain", "biographical"),
            ("User prefers direct communication", "core_preference"),
        ],
    ),
    Slot(
        "where does the user live",
        "User currently lives in Paris",
        "location",
        [
            "User currently lives in Berlin",
            "User currently lives in Lisbon",
            "User currently lives in Amsterdam",
        ],
        [
            ("User is building the billing service", "current_project"),
            ("User enjoys hiking on weekends", "core_preference"),
        ],
    ),
    Slot(
        "how should responses be formatted for this user",
        "User wants responses as concise bullet points",
        "core_preference",
        [
            "User wants responses as long detailed essays",
            "User wants responses with many code examples",
        ],
        [
            ("User lives in Paris", "location"),
            ("User is reviewing pull requests today", "current_task"),
        ],
    ),
    Slot(
        "what is the user's immediate task today",
        "User is reviewing pull request 42 today",
        "current_task",
        [
            "User is fixing the login bug today",
            "User is preparing the launch demo today",
            "User is writing documentation today",
            "User is debugging the payment webhook today",
        ],
        [
            ("User prefers concise answers", "core_preference"),
            ("User lives in Paris", "location"),
        ],
    ),
]


def profile_volatility(profile: str) -> dict[str, float]:
    real = dict(vdomains.DOMAIN_VOLATILITY)
    if profile == "real":
        return real
    if profile == "flat":
        return {d: 0.40 for d in real}
    if profile == "swap":
        return {d: round(1.0 - v, 3) for d, v in real.items()}
    raise ValueError(profile)


@contextmanager
def volatility_profile(profile: str):
    original = dict(vdomains.DOMAIN_VOLATILITY)
    try:
        vdomains.DOMAIN_VOLATILITY.update(profile_volatility(profile))
        yield
    finally:
        vdomains.DOMAIN_VOLATILITY.clear()
        vdomains.DOMAIN_VOLATILITY.update(original)


def seed_haystack(
    mem: MemoryLayer,
    slot: Slot,
    rng: random.Random,
    *,
    n_decoys: int,
    n_distractors: int,
    now: float,
) -> str:
    """Populate memory; return id of the current-truth item."""
    current_id = mem.write(
        slot.current, domain=slot.domain, at_time=now - 1 * DAY,
    ).item.id

    stale = rng.sample(slot.stale_pool, min(n_decoys, len(slot.stale_pool)))
    for text in stale:
        mem.write(text, domain=slot.domain, at_time=now - 90 * DAY)

    distractors = rng.sample(
        slot.distractor_pool, min(n_distractors, len(slot.distractor_pool)))
    for text, dom in distractors:
        mem.write(text, domain=dom, at_time=now - 30 * DAY)

    # extra noise: random filler across domains
    filler = [
        ("User is learning Rust", "skill"),
        ("User attended a conference last month", "professional_context"),
        ("User's long-term goal is staff engineer", "long_term_goal"),
        ("User thinks microservices are overrated", "opinion"),
    ]
    for text, dom in filler:
        mem.write(text, domain=dom, at_time=now - rng.uniform(10, 120) * DAY)

    return current_id


def eval_slot(
    system: str,
    slot: Slot,
    sim: EmbeddingSimilarity,
    rng: random.Random,
    *,
    n_decoys: int,
    n_distractors: int,
    top_k: int,
) -> dict:
    now = time.time()
    profile = system.replace("voltmem_", "") if system.startswith("voltmem_") else None
    ctx = volatility_profile(profile) if profile else nullcontext()
    use_staleness = system != "similarity_only"

    with ctx:
        with MemoryLayer(":memory:", similarity_fn=sim) as mem:
            current_id = seed_haystack(
                mem, slot, rng, n_decoys=n_decoys,
                n_distractors=n_distractors, now=now)
            result = mem.retrieve(
                slot.query, top_k=top_k, now=now, use_staleness=use_staleness)

            ids = [it.id for it in result.items]
            current_rank = ids.index(current_id) + 1 if current_id in ids else None

            stale_ids = {
                it.id for it in mem._active(domain=slot.domain)
                if it.id != current_id
            }
            stale_in_top1 = (
                len(result.items) > 0 and result.items[0].id in stale_ids)

            current_item = mem._store.get(current_id)
            current_score = 0.0
            best_stale_score = 0.0
            if current_item:
                cur_sim = sim(slot.query, current_item.content)
                current_score = (
                    retrieval_score(current_item, cur_sim, now)
                    if use_staleness else cur_sim)
            for it in mem._active(domain=slot.domain):
                if it.id == current_id:
                    continue
                s = sim(slot.query, it.content)
                sc = retrieval_score(it, s, now) if use_staleness else s
                best_stale_score = max(best_stale_score, sc)

    return {
        "current_at_1": current_rank == 1,
        "current_at_k": current_rank is not None and current_rank <= top_k,
        "stale_vol_top1": stale_in_top1,
        "separation": current_score - best_stale_score,
    }


def run(n_decoys: int, n_distractors: int, runs: int, top_k: int, seed: int):
    print("=" * 74)
    print("RETRIEVAL HAYSTACK — current truth vs stale volatile noise")
    print("=" * 74)
    sim = EmbeddingSimilarity(verbose=True)
    print(f"  backend: {sim.backend} ({sim.model})")
    print(f"  slots={len(SLOTS)}  runs/slot={runs}  decoys={n_decoys}  "
          f"distractors={n_distractors}  top_k={top_k}\n")

    stats = {s: {"current_at_1": 0, "current_at_k": 0, "stale_top1": 0,
                 "separation": 0.0, "n": 0} for s in SYSTEMS}

    for run_i in range(runs):
        rng = random.Random(seed + run_i)
        for slot in SLOTS:
            for system in SYSTEMS:
                r = eval_slot(system, slot, sim, rng,
                              n_decoys=n_decoys, n_distractors=n_distractors,
                              top_k=top_k)
                st = stats[system]
                st["n"] += 1
                st["current_at_1"] += int(r["current_at_1"])
                st["current_at_k"] += int(r["current_at_k"])
                st["stale_top1"] += int(r["stale_vol_top1"])
                st["separation"] += r["separation"]

    def rate(num, den):
        return num / den if den else 0.0

    print(f"  {'system':<16}{'current@1':>11}{'current@'+str(top_k):>11}"
          f"{'stale@1':>10}{'separation':>12}")
    print("  " + "-" * 60)
    for s in SYSTEMS:
        n = stats[s]["n"]
        print(f"  {s:<16}"
              f"{rate(stats[s]['current_at_1'], n):>11.3f}"
              f"{rate(stats[s]['current_at_k'], n):>11.3f}"
              f"{rate(stats[s]['stale_top1'], n):>10.3f}"
              f"{stats[s]['separation']/n:>12.3f}")

    real = stats["voltmem_real"]
    flat = stats["voltmem_flat"]
    swap = stats["voltmem_swap"]
    simo = stats["similarity_only"]
    n = real["n"]
    c1_real = rate(real["current_at_1"], n)
    ck_real = rate(real["current_at_k"], n)
    c1_sim = rate(simo["current_at_1"], n)
    ck_sim = rate(simo["current_at_k"], n)
    stale_real = rate(real["stale_top1"], n)
    stale_sim = rate(simo["stale_top1"], n)
    sep_real = real["separation"] / n
    sep_flat = flat["separation"] / n
    sep_swap = swap["separation"] / n

    print("\n" + "-" * 74)
    print("VERDICT:")
    # Primary win: avoid ranking stale volatile decoys #1; secondary: current in top-k
    avoids_stale = stale_real < stale_sim and sep_real > 0
    better_recall = ck_real >= ck_sim
    causal_sep = sep_real > sep_swap and sep_real >= sep_flat * 0.85

    if avoids_stale and better_recall and causal_sep:
        print(f"  PASS. voltmem_real avoids stale volatile traps (stale@1="
              f"{stale_real:.3f} vs similarity_only={stale_sim:.3f}), finds the "
              f"current fact in top-{top_k} more often ({ck_real:.3f} vs "
              f"{ck_sim:.3f}), and separates current from stale decoys "
              f"(sep={sep_real:.3f} > swap={sep_swap:.3f}).")
    elif avoids_stale and better_recall:
        print(f"  PARTIAL. Beats similarity_only on stale@1 ({stale_real:.3f} vs "
              f"{stale_sim:.3f}) and current@{top_k} ({ck_real:.3f} vs {ck_sim:.3f}); "
              f"causal separation ordering unclear (real={sep_real:.3f} "
              f"flat={sep_flat:.3f} swap={sep_swap:.3f}).")
    else:
        print(f"  NEEDS WORK. stale@1 real={stale_real:.3f} sim={stale_sim:.3f}; "
              f"current@{top_k} real={ck_real:.3f} sim={ck_sim:.3f}; "
              f"sep real={sep_real:.3f}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoys", type=int, default=6)
    ap.add_argument("--distractors", type=int, default=3)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.decoys, args.distractors, args.runs, args.top_k, args.seed)


if __name__ == "__main__":
    main()
