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
     core preferences). Medium-stable domains (e.g. professional_context) update
     on high-M explicit evidence via the V_d band θ-cap; very-stable domains
     (biographical, core_preference) retain on one-shot explicit and only update
     after cumulative strong evidence.

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
# Single-shot probe:
#   (domain, base, obs, mm, src, expected, note)
#   expected: "U" = should update (audited), "R" = should retain
#
# Cumulative probe (multi-turn):
#   (domain, base, [(obs, mm, src, expected), ...], note)
#   every step must match; scores as one probe.

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

    # medium-stable band: high-M explicit -> UPDATE (θ-cap)
    ("professional_context", "User works as a data analyst",
     "User explicitly said they changed careers and now work as a nurse", 0.90,
     "explicit_statement", "U", "medium-band: career change, explicit"),
    ("skill", "User is proficient in Python",
     "User explicitly said they no longer use Python and work in Rust now", 0.90,
     "explicit_statement", "U", "medium-band: skill change, explicit"),
    ("relationship", "User works closely with Alice",
     "User explicitly said they no longer work with Alice", 0.90,
     "explicit_statement", "U", "medium-band: relationship change, explicit"),
    ("long_term_goal", "User wants to become a research scientist",
     "User explicitly said their goal is now to start a company", 0.90,
     "explicit_statement", "U", "medium-band: goal change, explicit"),

    # below-band (very stable): high-M explicit one-shot -> RETAIN
    ("core_preference", "User prefers concise, direct answers",
     "User explicitly said they now prefer long, detailed explanations", 0.90,
     "explicit_statement", "R", "below-band: pref flip one-shot retains"),
    ("biographical", "User grew up in Lagos",
     "User explicitly said they grew up in Nairobi", 0.90,
     "explicit_statement", "R", "below-band: bio flip one-shot retains"),
    ("personality_trait", "User is deeply introverted",
     "User explicitly said they are now highly extroverted", 0.90,
     "explicit_statement", "R", "below-band: trait flip one-shot retains"),

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

# Below-band real changes: N weak logged mismatches, then strong evidence updates.
CUMULATIVE_ESCALATION_PROBES = [
    ("biographical", "User grew up in Lagos",
     [
         ("Someone mentioned Nairobi in passing", 0.65, "weak_inference", "R"),
         ("Another acquaintance said Nairobi", 0.65, "weak_inference", "R"),
         ("A third mention of growing up in Nairobi", 0.65, "weak_inference", "R"),
         ("User confirmed they grew up in Nairobi", 0.75, "strong_inference", "U"),
     ],
     "below-band: cumulative then strong updates bio"),
    ("core_preference", "User prefers concise, direct answers",
     [
         ("User asked for more detail once", 0.65, "weak_inference", "R"),
         ("User lingered on a long explanation", 0.65, "weak_inference", "R"),
         ("User praised a verbose reply", 0.65, "weak_inference", "R"),
         ("User said they now prefer detailed answers", 0.75, "strong_inference", "U"),
     ],
     "below-band: cumulative then strong updates pref"),
]


def action_is_update(action):
    return action == "audited"


def _got_ur(action):
    return "U" if action_is_update(action) else "R"


def run_escalation(profile, escalation_mode="composite", v_exp=None, mode_scale=None):
    correct = 0
    rows = []
    layer_kwargs = dict(
        escalation_mode=escalation_mode,
        escalation_v_exp=v_exp,
        escalation_mode_scale=mode_scale,
    )
    with volatility_profile(profile):
        for (domain, base, obs, mm, src, expected, note) in ESCALATION_PROBES:
            with MemoryLayer(":memory:", **layer_kwargs) as mem:
                mem.write(base, domain=domain)
                res = mem.observe(obs, domain=domain, mismatch_magnitude=mm,
                                  source=src)
                got = _got_ur(res.action)
                ok = (got == expected)
                correct += ok
                rows.append((domain, expected, got, ok, res.action, note))

        for (domain, base, steps, note) in CUMULATIVE_ESCALATION_PROBES:
            with MemoryLayer(":memory:", **layer_kwargs) as mem:
                mem.write(base, domain=domain)
                step_ok = True
                last_action = last_expected = last_got = None
                for obs, mm, src, expected in steps:
                    res = mem.observe(
                        obs, domain=domain, mismatch_magnitude=mm, source=src)
                    got = _got_ur(res.action)
                    last_action, last_expected, last_got = res.action, expected, got
                    if got != expected:
                        step_ok = False
                        break
                ok = step_ok
                correct += ok
                rows.append(
                    (domain, last_expected, last_got, ok, last_action, note))

    n = len(ESCALATION_PROBES) + len(CUMULATIVE_ESCALATION_PROBES)
    return correct, n, rows


def run_calibration_footprint():
    """Replay Battery A probes into one layer; return domain_stats() telemetry."""
    with volatility_profile("real"):
        with MemoryLayer(":memory:") as mem:
            for (domain, base, obs, mm, src, _expected, _note) in ESCALATION_PROBES:
                mem.write(base, domain=domain)
                mem.observe(obs, domain=domain, mismatch_magnitude=mm, source=src)
            for (domain, base, steps, _note) in CUMULATIVE_ESCALATION_PROBES:
                mem.write(base, domain=domain)
                for obs, mm, src, _expected in steps:
                    mem.observe(
                        obs, domain=domain, mismatch_magnitude=mm, source=src)
            return mem.domain_stats()


# ── Battery C: recency-shift (allostatic diagnostic) ──────────────────────────
# Entrench with confirms, optionally log below-threshold mismatches, then a real
# explicit change. Current misses high-C professional_context (E_t is crushed
# before the N=3 cliff). Allostatic recovers it by dropping V_d from E_t and
# lowering θ via s(m), without flipping core_preference (negative control).

RECENCY_SHIFT_PROBES = [
    {
        "name": "noisy-then-shift professional",
        "domain": "professional_context",
        "base": "User works as a data analyst",
        "n_confirms": 5,
        "n_mismatch": 2,
        "final_obs": "User explicitly said they changed careers and now work as a nurse",
        "final_mm": 0.90,
        "final_src": "explicit_statement",
        "want_homeostatic": "R",
        "want_allostatic": "U",
        "note": "diagnostic: reopen after recent surprise",
    },
    {
        "name": "noisy-then-shift preference",
        "domain": "core_preference",
        "base": "User prefers concise, direct answers",
        "n_confirms": 5,
        "n_mismatch": 2,
        "final_obs": "User explicitly said they now prefer long, detailed explanations",
        "final_mm": 0.90,
        "final_src": "explicit_statement",
        "want_homeostatic": "R",
        "want_allostatic": "R",
        "note": "negative control: very-stable stays closed",
    },
    {
        "name": "quiet-then-shift professional",
        "domain": "professional_context",
        "base": "User works as a data analyst",
        "n_confirms": 5,
        "n_mismatch": 0,
        "final_obs": "User explicitly said they changed careers and now work as a nurse",
        "final_mm": 0.90,
        "final_src": "explicit_statement",
        "want_homeostatic": "R",
        "want_allostatic": "U",
        "note": "quiet then explicit: dropping V_d from E_t recovers the change",
    },
]


def _run_recency_probe(probe, escalation_mode, v_exp=None, mode_scale=None):
    with MemoryLayer(
        ":memory:",
        escalation_mode=escalation_mode,
        escalation_v_exp=v_exp,
        escalation_mode_scale=mode_scale,
    ) as mem:
        mem.write(probe["base"], domain=probe["domain"])
        for i in range(probe["n_confirms"]):
            mem.observe(
                f"{probe['base']} (still true {i})",
                domain=probe["domain"],
                mismatch_magnitude=0.05,
                source="explicit_statement",
            )
        for i in range(probe["n_mismatch"]):
            mem.observe(
                f"weak counter-signal {i}",
                domain=probe["domain"],
                mismatch_magnitude=0.65,
                source="weak_inference",
            )
        res = mem.observe(
            probe["final_obs"],
            domain=probe["domain"],
            mismatch_magnitude=probe["final_mm"],
            source=probe["final_src"],
        )
        return _got_ur(res.action), res.action


def run_recency_shift():
    rows = []
    for probe in RECENCY_SHIFT_PROBES:
        got_c, act_c = _run_recency_probe(probe, "homeostatic")
        got_a, act_a = _run_recency_probe(probe, "allostatic")
        ok_c = got_c == probe["want_homeostatic"]
        ok_a = got_a == probe["want_allostatic"]
        rows.append((probe, got_c, act_c, ok_c, got_a, act_a, ok_a))
    return rows


# ── Battery D: weak slow burn (isolates s(m)) ─────────────────────────────────
# Battery C could not test s(m): every probe ends in an explicit M=0.90
# statement, which clears the medium-band θ-cap whatever s(m) does. The
# ablation confirmed s(m) changed no Battery C outcome.
#
# This battery removes both shortcuts. Every observation is a WEAK inference
# (R=0.4), so:
#   * M < EXPLICIT_OVERRIDE_M and the source is not explicit → no θ-cap, and
#   * R < SOURCE_RELIABILITY["strong_inference"] → the cumulative N-strike
#     override can never fire either.
# Nothing except s(m) can reopen the bar. After residual surprise (r_t =
# |M − Ê| / σ) replaced raw M in the EMA, a constant weak stream becomes
# *predicted* and no longer reopens — that is the first-test finding, not a
# regression of the controls. Catching the pile is a cumulative-belief job.

# Three controls keep it honest: a very-stable domain must never reopen,
# interleaved confirms must keep it settled, and the same number of mentions
# spread over months must NOT accumulate (surprise has to be recent, not just
# numerous — that is what the decay half-life buys).

SLOW_BURN_START = 1_700_000_000.0

SLOW_BURN_PROBES = [
    {
        "name": "weak burn, daily",
        "domain": "professional_context",
        "base": "User works as a data analyst",
        "obs": "User mentioned a nursing shift in passing",
        "mm": 0.70,
        "src": "weak_inference",
        "turns": 16,
        "gap_days": 1.0,
        "confirm_between": False,
        "expect": {"homeostatic": "never", "allostatic": "never",
                   "allostatic_noscale": "never"},
        "note": "first-test: a predicted weak stream must not reopen via s(m)",
    },
    {
        "name": "weak burn, very stable",
        "domain": "core_preference",
        "base": "User prefers concise, direct answers",
        "obs": "User lingered on a long explanation",
        "mm": 0.70,
        "src": "weak_inference",
        "turns": 16,
        "gap_days": 1.0,
        "confirm_between": False,
        "expect": {"homeostatic": "never", "allostatic": "never",
                   "allostatic_noscale": "never"},
        "note": "control: deep preference must not yield to weak evidence",
    },
    {
        "name": "weak burn + confirms",
        "domain": "professional_context",
        "base": "User works as a data analyst",
        "obs": "User mentioned a nursing shift in passing",
        "confirm_obs": "User referred to their analyst work again",
        "mm": 0.70,
        "src": "weak_inference",
        "turns": 16,
        "gap_days": 1.0,
        "confirm_between": True,
        "expect": {"homeostatic": "never", "allostatic": "never",
                   "allostatic_noscale": "never"},
        "note": "control: confirms in between must keep the bar settled",
    },
    {
        "name": "weak burn, monthly",
        "domain": "professional_context",
        "base": "User works as a data analyst",
        "obs": "User mentioned a nursing shift in passing",
        "mm": 0.70,
        "src": "weak_inference",
        "turns": 16,
        "gap_days": 30.0,
        "confirm_between": False,
        "expect": {"homeostatic": "never", "allostatic": "never",
                   "allostatic_noscale": "never"},
        "note": "control: same evidence spread thin must decay, not accumulate",
    },
]

SLOW_BURN_CONFIGS = {
    # label -> (escalation_mode, v_exp, mode_scale)
    "homeostatic": ("homeostatic", None, None),
    "allostatic": ("allostatic", None, None),
    "allostatic_noscale": ("allostatic", None, False),
}


def _run_slow_burn(probe, escalation_mode, v_exp=None, mode_scale=None):
    """Return the 1-based turn that first escalated, or None if never."""
    start = SLOW_BURN_START
    gap = probe["gap_days"] * DAY
    with MemoryLayer(
        ":memory:",
        escalation_mode=escalation_mode,
        escalation_v_exp=v_exp,
        escalation_mode_scale=mode_scale,
    ) as mem:
        mem.write(probe["base"], domain=probe["domain"], at_time=start)
        for i in range(probe["turns"]):
            at = start + (i + 1) * gap
            if probe["confirm_between"] and i % 2 == 1:
                mem.observe(
                    probe["confirm_obs"],
                    domain=probe["domain"],
                    mismatch_magnitude=0.05,
                    source="explicit_statement",
                    at_time=at,
                )
                continue
            res = mem.observe(
                f"{probe['obs']} ({i})",
                domain=probe["domain"],
                mismatch_magnitude=probe["mm"],
                source=probe["src"],
                at_time=at,
            )
            if res.action == "audited":
                return i + 1
    return None


def run_slow_burn():
    rows = []
    for probe in SLOW_BURN_PROBES:
        got, ok = {}, {}
        for label, (mode, v_exp, mode_scale) in SLOW_BURN_CONFIGS.items():
            turn = _run_slow_burn(probe, mode, v_exp=v_exp, mode_scale=mode_scale)
            got[label] = turn
            want = probe["expect"][label]
            ok[label] = (turn is None) if want == "never" else (turn is not None)
        rows.append((probe, got, ok))
    return rows


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

    print("\n  per-probe detail (real profile, homeostatic trigger):")
    for (domain, expected, got, ok, action, note) in esc["real"][2]:
        flag = "ok " if ok else "XX "
        print(f"    [{flag}] {domain:20s} want={expected} got={got} "
              f"({action:14s}) {note}")

    print("\n  allostatic trigger on Battery A (same expected labels):")
    esc_allo = {}
    for p in profiles:
        c, n, rows = run_escalation(p, escalation_mode="allostatic")
        esc_allo[p] = (c, n, rows)
        print(f"  {p:5s}: {c}/{n} probes correct  ({c / n:.0%})")
    print("  per-probe detail (real profile, allostatic):")
    for (domain, expected, got, ok, action, note) in esc_allo["real"][2]:
        flag = "ok " if ok else "XX "
        print(f"    [{flag}] {domain:20s} want={expected} got={got} "
              f"({action:14s}) {note}")

    print("\n  prior calibration footprint (Battery A replay, real profile):")
    print(f"  {'domain':<22}{'ins':>5}{'conf':>6}{'mm':>5}{'aud':>5}"
          f"{'aud_rate':>10}{'prior':>8}")
    for domain, row in sorted(run_calibration_footprint().items()):
        print(f"  {domain:<22}{row['inserted']:>5}{row['confirmed']:>6}"
              f"{row['logged_mismatch']:>5}{row['audited']:>5}"
              f"{row['audit_rate']:>10.2f}{row['prior']:>8.2f}")

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

    # Battery C
    print("\nBATTERY C — RECENCY-SHIFT (allostatic diagnostic)")
    print("-" * 76)
    recency = run_recency_shift()
    for probe, got_c, act_c, ok_c, got_a, act_a, ok_a in recency:
        flag_c = "ok " if ok_c else "XX "
        flag_a = "ok " if ok_a else "XX "
        print(f"  [{flag_c}] homeostatic want={probe['want_homeostatic']} "
              f"got={got_c} ({act_c:14s}) {probe['name']}")
        print(f"  [{flag_a}] allostatic  want={probe['want_allostatic']} "
              f"got={got_a} ({act_a:14s}) {probe['note']}")

    # Battery D
    print("\nBATTERY D — WEAK SLOW BURN (isolates s(m); no θ-cap, no N-strike)")
    print("-" * 76)
    print(f"  {'probe':<24}{'homeostatic':>12}{'allostatic':>12}{'allo no-s(m)':>14}"
          f"{'ok':>5}")
    burn = run_slow_burn()
    for probe, got, ok in burn:
        def _fmt(label):
            turn = got[label]
            return "never" if turn is None else f"turn {turn}"
        all_ok = all(ok.values())
        print(f"  {probe['name']:<24}{_fmt('homeostatic'):>12}"
              f"{_fmt('allostatic'):>12}{_fmt('allostatic_noscale'):>14}"
              f"{('ok' if all_ok else 'XX'):>5}")
        print(f"    {probe['note']}")
    print("\n  Every observation is weak_inference (R=0.4), below both the θ-cap")
    print("  and the cumulative N-strike override — so only s(m) can reopen the")
    print("  bar. 'allo no-s(m)' is the same law with the surprise term disabled:")
    print("  if it matches plain allostatic everywhere, s(m) does nothing.")

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
    aa_real = esc_allo["real"][0] / esc_allo["real"][1]
    aa_flat = esc_allo["flat"][0] / esc_allo["flat"][1]
    aa_swap = esc_allo["swap"][0] / esc_allo["swap"][1]
    aa_ok = aa_real > aa_flat and aa_real >= aa_swap and aa_real > 0.5
    recency_ok = all(ok_c and ok_a for _, _, _, ok_c, _, _, ok_a in recency)
    recency_diag = next(
        (row for row in recency if row[0]["name"] == "noisy-then-shift professional"),
        None,
    )
    diag_win = (
        recency_diag is not None
        and recency_diag[1] == "R"
        and recency_diag[4] == "U"
    )

    print()
    print(f"  A allostatic   accuracy   real={aa_real:.0%}  "
          f"flat={aa_flat:.0%}  swap={aa_swap:.0%}")
    print(f"  C recency-shift probes matching expected: "
          f"{sum(ok_c and ok_a for _, _, _, ok_c, _, _, ok_a in recency)}"
          f"/{len(recency)}")
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
    print(f"  Allostatic Battery A causal: {aa_ok}")
    print(f"  Recency-shift expected labels: {recency_ok}")
    print(f"  Diagnostic (allostatic updates professional, homeostatic retains): {diag_win}")

    burn_ok = all(all(ok.values()) for _probe, _got, ok in burn)
    sm_earns_keep = any(
        got["allostatic"] is not None and got["allostatic_noscale"] is None
        for _probe, got, _ok in burn
    )
    sm_controls_held = all(
        got["allostatic"] is None
        for probe, got, _ok in burn
        if probe["expect"]["allostatic"] == "never"
    )
    print(f"  Battery D expected labels: {burn_ok}")
    print(f"  s(m) does something no exponent change does: {sm_earns_keep}")
    print(f"  s(m) controls held (stable / confirmed / spread-thin): {sm_controls_held}")
    if sm_earns_keep and sm_controls_held:
        print("  => s(m) EARNS ITS KEEP: it reopens on accumulated weak evidence,")
        print("     which no V_d exponent can do, and stays shut on all controls.")
    elif not sm_earns_keep:
        print("  => s(m) on r_t habituates to a constant weak stream. That is")
        print("     the first-test finding: online surprise is unexpected residual,")
        print("     not an EMA of M. Battery D's pile belongs to cumulative belief.")
    else:
        print("  => s(m) reopens, but a control also broke — it is too plastic.")


if __name__ == "__main__":
    main()
