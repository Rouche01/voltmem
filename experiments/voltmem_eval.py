"""
End-to-end evaluation of the VoltMem memory layer
=================================================

The neural experiments (ewc_volatility_*.py) validated the volatility PRINCIPLE
on weights. This script evaluates the actual PRODUCT — the VoltMem library — on
the two behaviours it promises, and checks that those behaviours are genuinely
caused by the per-domain volatility signal rather than by chance.

Two capabilities are measured:

  A) SELECTIVE UPDATING (escalation).  When a new observation contradicts a
     stored memory, VoltMem should UPDATE volatile facts readily (job, mood,
     current task) but RETAIN stable facts under weak evidence (personality,
     core preferences), while still updating even a stable fact given strong,
     reliable evidence.

  B) FRESHNESS-AWARE RETRIEVAL.  When ranking memories, a stale VOLATILE memory
     should be pushed down (it has probably gone out of date), while a stable
     memory of the same age should be barely penalised.

Each battery is run under three volatility PROFILES to establish causality — the
same control idea as the neural --sabotage test:

  real  : the library's true per-domain volatilities.
  flat  : every domain forced to the same volatility (0.5) — the "treat all
          memories equally" baseline. If VoltMem's value is real, this loses.
  swap  : each domain's volatility inverted (v -> 1 - v) — stable treated as
          volatile and vice versa. This should be WORST; if it isn't, the
          behaviour is not actually driven by the volatility signal.

Run:
    .venv/bin/python experiments/voltmem_eval.py
"""

import contextlib
import os
import sys
import time

# make the repo root importable regardless of where this is launched from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import voltmem.domains as vdomains          # noqa: E402
from voltmem import MemoryLayer              # noqa: E402
from voltmem.domains import MemoryItem       # noqa: E402
from voltmem.scoring import retrieval_score, staleness  # noqa: E402

DAY = 86400.0


# ── volatility profile switch (mutates the shared dict in place) ───────────────

@contextlib.contextmanager
def volatility_profile(profile):
    original = dict(vdomains.DOMAIN_VOLATILITY)
    try:
        if profile == "flat":
            for k in vdomains.DOMAIN_VOLATILITY:
                vdomains.DOMAIN_VOLATILITY[k] = 0.5
        elif profile == "swap":
            for k in list(vdomains.DOMAIN_VOLATILITY):
                vdomains.DOMAIN_VOLATILITY[k] = 1.0 - original[k]
        elif profile != "real":
            raise ValueError(profile)
        yield
    finally:
        vdomains.DOMAIN_VOLATILITY.clear()
        vdomains.DOMAIN_VOLATILITY.update(original)


# ── Battery A: selective updating (escalation) ────────────────────────────────
# Each probe: bootstrap a base fact, feed one contradicting observation, and
# check whether the layer updated ("audited") or kept it ("confirmed"/
# "logged_mismatch"). expected: "U" = should update, "R" = should retain.

ESCALATION_PROBES = [
    # realistic volatile updates (legit, strong, reliable) -> should UPDATE
    ("current_project", "User is job hunting",
     "User accepted a job offer and is no longer job hunting", 0.90,
     "explicit_statement", "U", "volatile: legit job change"),
    ("current_task", "User is preparing the Monday slides",
     "User finished the slides and is now writing the report", 0.90,
     "explicit_statement", "U", "volatile: task moved on"),
    ("emotional_context", "User is feeling stressed",
     "User says they feel calm and relaxed now", 0.85,
     "explicit_statement", "U", "volatile: mood changed"),
    ("location", "User lives in Lagos",
     "User moved to Berlin last week", 0.90,
     "explicit_statement", "U", "volatile: relocation"),

    # realistic stable facts under WEAK evidence -> should RETAIN
    ("personality_trait", "User is deeply introverted",
     "User was talkative at one event", 0.60,
     "weak_inference", "R", "stable: weak counter-signal"),
    ("core_preference", "User prefers concise, direct answers",
     "User asked one unusually detailed question today", 0.55,
     "weak_inference", "R", "stable: one-off exception"),
    ("biographical", "User grew up in Lagos",
     "Someone mentioned Nairobi in passing", 0.50,
     "weak_inference", "R", "stable: noisy hearsay"),

    # stable fact under STRONG reliable evidence -> should still UPDATE
    ("professional_context", "User works as a data analyst",
     "User explicitly said they changed careers and now work as a nurse", 0.90,
     "explicit_statement", "U", "stable but real change, strong evidence"),

    # matched-pressure discriminators: SAME mismatch/source, extreme domains.
    # Only the domain volatility differs, so these isolate the volatility signal.
    ("current_task", "User is working on task Alpha",
     "User is now working on task Beta instead", 0.75,
     "strong_inference", "U", "matched: volatile should yield"),
    ("personality_trait", "User is a careful, risk-averse planner",
     "User made one impulsive decision", 0.75,
     "strong_inference", "R", "matched: stable should hold"),
    ("emotional_context", "User is anxious about the deadline",
     "User seems upbeat in this message", 0.75,
     "strong_inference", "U", "matched: volatile should yield"),
    ("core_preference", "User strongly prefers dark mode",
     "User used light mode once on a shared screen", 0.75,
     "strong_inference", "R", "matched: stable should hold"),
]


def action_is_update(action):
    return action == "audited"


def run_escalation(profile):
    correct = 0
    rows = []
    with volatility_profile(profile):
        for (domain, base, obs, mm, src, expected, note) in ESCALATION_PROBES:
            with MemoryLayer(":memory:") as mem:
                mem.write(base, domain=domain)
                res = mem.observe(obs, domain=domain, mismatch_magnitude=mm,
                                  source=src)
                got = "U" if action_is_update(res.action) else "R"
                ok = (got == expected)
                correct += ok
                rows.append((domain, expected, got, ok, res.action, note))
    return correct, len(ESCALATION_PROBES), rows


# ── Battery B: freshness-aware retrieval ──────────────────────────────────────
# Score identical-similarity memories that differ only in domain and age, then
# check the ranking separates "still trustworthy" from "probably stale".
# Ground truth: a memory is trustworthy unless it is VOLATILE and OLD.

STABLE_DOMAINS = ["personality_trait", "core_preference", "biographical"]
VOLATILE_DOMAINS = ["current_project", "emotional_context", "current_task"]
AGES = {"fresh": 1.0, "old": 60.0}


def _item(domain, age_days, now):
    return MemoryItem(
        id=f"{domain}-{age_days}", content="x", domain=domain,
        source="explicit_statement",
        created_at=now - age_days * DAY,
        last_confirmed_at=now - age_days * DAY,
    )


def run_retrieval(profile):
    now = time.time()
    cells = []  # (label, domain_class, age_label, trusted, mean_score)
    with volatility_profile(profile):
        for age_label, age in AGES.items():
            for dclass, domains in (("stable", STABLE_DOMAINS),
                                    ("volatile", VOLATILE_DOMAINS)):
                scores = [retrieval_score(_item(d, age, now),
                                          semantic_similarity=1.0, now=now)
                          for d in domains]
                trusted = not (dclass == "volatile" and age_label == "old")
                cells.append((f"{dclass:8s} {age_label:5s}", dclass, age_label,
                              trusted, sum(scores) / len(scores)))
    trusted_scores = [c[4] for c in cells if c[3]]
    untrusted_scores = [c[4] for c in cells if not c[3]]
    separation = (sum(trusted_scores) / len(trusted_scores)
                  - sum(untrusted_scores) / len(untrusted_scores))
    # key contrast: stable-old (should stay) vs volatile-old (should drop)
    stable_old = next(c[4] for c in cells if c[1] == "stable" and c[2] == "old")
    volatile_old = next(c[4] for c in cells if c[1] == "volatile" and c[2] == "old")
    return cells, separation, stable_old - volatile_old


# ── report ────────────────────────────────────────────────────────────────────

def main():
    profiles = ["real", "flat", "swap"]

    print("=" * 76)
    print("VoltMem end-to-end eval — real vs flat (equal) vs swap (inverted)")
    print("=" * 76)

    # Battery A
    print("\nBATTERY A — SELECTIVE UPDATING (higher accuracy = better)")
    print("-" * 76)
    esc = {}
    for p in profiles:
        c, n, rows = run_escalation(p)
        esc[p] = (c, n, rows)
        print(f"  {p:5s}: {c}/{n} probes correct  ({c / n:.0%})")

    print("\n  per-probe detail (real profile):")
    for (domain, expected, got, ok, action, note) in esc["real"][2]:
        flag = "ok " if ok else "XX "
        print(f"    [{flag}] {domain:20s} want={expected} got={got} "
              f"({action:14s}) {note}")

    # Battery B
    print("\nBATTERY B — FRESHNESS-AWARE RETRIEVAL")
    print("-" * 76)
    print("  retrieval score by (domain class x age); similarity fixed at 1.0")
    print(f"  {'profile':7s}{'stable fresh':>14}{'stable old':>13}"
          f"{'volatile fresh':>16}{'volatile old':>14}{'  sep':>8}")
    ret = {}
    for p in profiles:
        cells, sep, key = run_retrieval(p)
        ret[p] = (cells, sep, key)
        by = {(c[1], c[2]): c[4] for c in cells}
        print(f"  {p:7s}{by[('stable','fresh')]:>14.3f}{by[('stable','old')]:>13.3f}"
              f"{by[('volatile','fresh')]:>16.3f}{by[('volatile','old')]:>14.3f}"
              f"{sep:>8.3f}")
    print("\n  'sep' = mean score of trustworthy memories minus mean score of")
    print("  stale (volatile+old) memories. Higher = retrieval correctly favours")
    print("  memories that are still valid.")

    # ── verdict ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)

    a_real, a_flat, a_swap = (esc[p][0] / esc[p][1] for p in profiles)
    s_real = ret["real"][1]
    s_flat = ret["flat"][1]
    s_swap = ret["swap"][1]

    print(f"  A (updating)  accuracy   real={a_real:.0%}  flat={a_flat:.0%}  swap={a_swap:.0%}")
    print(f"  B (retrieval) separation real={s_real:+.3f} flat={s_flat:+.3f} swap={s_swap:+.3f}")

    a_ok = a_real > a_flat and a_real >= a_swap and a_real > 0.5
    b_ok = s_real > s_flat and s_real > s_swap

    print()
    if a_ok and b_ok:
        print("  PASS. On BOTH capabilities the true volatility profile beats the")
        print("  equal-treatment baseline, and beats (or ties) the inverted profile.")
        print("  VoltMem's behaviour is genuinely driven by per-domain volatility —")
        print("  not an accident of thresholds. The 'swap' control degrading is the")
        print("  causal evidence: flipping which domains are 'volatile' flips the")
        print("  behaviour in the wrong direction.")
    else:
        print("  MIXED / FAIL — read per-battery:")
        print(f"    Battery A causal (real>flat and real>=swap): {a_ok}")
        print(f"    Battery B causal (real>flat and real>swap):  {b_ok}")
        print("  Where a battery fails, the corresponding mechanism is not clearly")
        print("  driven by the volatility signal in this setup (worth investigating,")
        print("  e.g. the observe() EMA update or source-reliability dominating).")


if __name__ == "__main__":
    main()
