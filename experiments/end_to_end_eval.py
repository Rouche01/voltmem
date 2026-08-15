"""
Battery F — end-to-end through remember(): where the error budget actually goes
==============================================================================

Batteries A–E all hand the system two things it does not have in production:
the domain label and the mismatch magnitude. That makes them measurements of
the escalation control law *given a perfect router*. Useful, and that is how
they were designed — but it means no battery in this repo can see the two
components that sit upstream of the decision:

  1. routing  — does a contradicting observation find the memory it contradicts?
                remember() tries a global semantic match (relate_threshold),
                then a within-slot match, then gives up and inserts a NEW item.
                An insert means the escalation law is never consulted at all.
  2. mismatch estimation — how strongly does the extractor think the new text
                contradicts the stored text? Batteries A–E supply this by hand.

This battery runs the same probes with neither hand-fed. Only `remember(text)`
is called, so classification, routing, mismatch estimation, and escalation all
have to work together.

The headline finding is that routing dominates everything else, and that most
of it is the DEFAULT SIMILARITY FUNCTION rather than the classifier: swapping
keyword overlap for embeddings moves end-to-end accuracy by ~50 points, while
label error is worth ~9. The escalation-mode choice that Batteries C/D/E exist
to settle is worth ~1 point in this context. Anyone tuning the control law
before fixing routing is optimising the smallest term in the budget.

Three similarity configurations, so the cost of the fix is visible too:

  keyword               the shipped default; dependency-free keyword overlap
  hashing               EmbeddingSimilarity(backend="hashing") — still
                        dependency-free, so whatever this recovers is FREE
  sentence-transformers real local model, if installed

Two label conditions:

  oracle    the extractor is forced to return each probe's true domain, which
            isolates routing and decision from classification error
  shipped   the real HeuristicExtractor exactly as it ships, no injection

Reported per row: how often the observation routed to the stored memory at all,
end-to-end correctness, and correctness *given* that it routed — which
separates "never found the memory" from "found it and decided wrong".

Run:
    .venv/bin/python experiments/end_to_end_eval.py
"""

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voltmem import MemoryLayer                                # noqa: E402
from voltmem.embeddings import EmbeddingSimilarity             # noqa: E402
from voltmem.extract import HeuristicExtractor                 # noqa: E402
from voltmem_eval import (                                     # noqa: E402
    ESCALATION_PROBES,
    _got_ur,
    volatility_profile,
)

# Distinct coexisting facts: (domain_a, text_a, domain_b, text_b, note).
#
# Why these exist. Every ESCALATION_PROBE is a same-fact update — a stored fact
# and a statement about that same fact — so a false merge is impossible in that
# set and preventing one earns nothing. Battery F could therefore only ever see
# the COST of a conservative verifier, never its benefit, and it duly scored
# two-stage linking as a regression while Battery G scored it as removing 12
# irreversible errors. Neither battery was wrong; each saw half the ledger.
#
# These probes supply the missing half in the same harness, so one run produces
# both numbers. They are written fresh rather than lifted from
# ``linking_pairs.py``: reusing those sentences would re-measure a corpus the
# verifier's prompt was already exercised against, and would tell us nothing
# about generalisation. The domain mix deliberately mirrors ESCALATION_PROBES so
# the two halves are comparable.
COEXIST_PROBES = [
    ("current_project", "User is building the billing service",
     "current_project", "User is redesigning the onboarding flow",
     "two live projects"),
    ("current_task", "User is writing the incident report",
     "current_task", "User needs to book the venue",
     "two concurrent to-dos"),
    ("emotional_context", "User is nervous about the demo",
     "emotional_context", "User is delighted with the new hire",
     "two feelings about different things"),
    ("location", "User lives in Lagos",
     "location", "User's brother lives in Accra",
     "same frame, different subject"),
    ("transient_fact", "User has a flight on Tuesday",
     "transient_fact", "User has a dentist appointment on Friday",
     "two unrelated near-term facts"),
    ("personality_trait", "User is deeply introverted",
     "personality_trait", "User is unusually persistent",
     "two distinct traits"),
    ("core_preference", "User prefers concise, direct answers",
     "core_preference", "User prefers metric units",
     "two distinct preferences"),
    ("core_preference", "User prefers concise, direct answers",
     "stated_preference", "User prefers morning meetings",
     "sibling domains are searched together"),
    ("biographical", "User grew up in Lagos",
     "biographical", "User trained as a pharmacist",
     "two distinct biographical facts"),
    ("biographical", "User's father is a doctor",
     "professional_context", "User works as a data analyst",
     "different subject, adjacent topic"),
    ("professional_context", "User works as a data analyst",
     "professional_context", "User sits on the ethics board",
     "two facts about one job"),
    ("skill", "User is comfortable with Kubernetes",
     "skill", "User is comfortable with Terraform",
     "near-identical frame, two skills"),
    ("relationship", "User works closely with Alice",
     "relationship", "User's mentor is Daniel",
     "same frame, different people and roles"),
    ("long_term_goal", "User wants to become a research scientist",
     "long_term_goal", "User wants to run a marathon",
     "two distinct goals"),
]

# Probe text -> its declared true domain, for the oracle condition.
TRUTH = {}
for _p in ESCALATION_PROBES:
    TRUTH[_p[1]] = _p[0]
    TRUTH[_p[2]] = _p[0]
for _c in COEXIST_PROBES:
    TRUTH.setdefault(_c[1], _c[0])
    TRUTH.setdefault(_c[3], _c[2])


class ProbeExtractor:
    """Wraps HeuristicExtractor; optionally forces the true domain, and records
    the mismatch magnitudes it produces so they can be compared with the
    hand-set values Batteries A–E rely on."""

    def __init__(self, oracle: bool):
        self._inner = HeuristicExtractor()
        self._oracle = oracle
        self.mismatches: list[float] = []
        self.label_errors = 0

    def classify_domain(self, text: str) -> str:
        pred = self._inner.classify_domain(text)
        true = TRUTH.get(text)
        if true is not None and pred != true:
            self.label_errors += 1
        if self._oracle and true is not None:
            return true
        return pred

    def mismatch(self, new_text: str, existing_text: str, similarity: float) -> float:
        m = self._inner.mismatch(new_text, existing_text, similarity)
        self.mismatches.append(float(m))
        return m


def build_similarities():
    """(label, similarity_fn, verifier) per backend. keyword => None."""
    out = [("keyword (default)", None)]
    try:
        out.append(("hashing (free)", EmbeddingSimilarity(backend="hashing")))
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [skip] hashing backend unavailable: {exc}")
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        out.append((
            "sentence-transformers",
            EmbeddingSimilarity(backend="sentence-transformers"),
        ))
    except Exception:                                          # noqa: BLE001
        print("  [skip] sentence-transformers unavailable "
              "(no local model cache) — keyword and hashing rows still run")
    out = [(label, fn, None) for label, fn in out]
    # LINK_VERIFIER=qwen2.5-coder:14b adds the two-stage row: same recall, but
    # the link decision moves from the threshold ladder to the verifier.
    from linking_eval import build_link_verifier
    verifier = build_link_verifier()
    if verifier is not None and len(out) > 1:
        best_label, best_fn, _ = out[-1]
        out.append((f"{best_label} + verify", best_fn, verifier))
    return out


def run_config(similarity_fn, oracle, escalation_mode="composite", verifier=None):
    routed = correct = correct_routed = false_update = missed_update = 0
    unrouted, mismatch_err, label_errors = [], [], 0

    for (domain, base, obs, hand_mm, _src, expected, note) in ESCALATION_PROBES:
        ex = ProbeExtractor(oracle=oracle)
        kwargs = {"extractor": ex}
        if similarity_fn is not None:
            kwargs["similarity_fn"] = similarity_fn
        if verifier is not None:
            kwargs["link_verifier"] = verifier
            kwargs["link_recall_bar"] = float(
                os.environ.get("LINK_RECALL_BAR", "0.30"))
        with MemoryLayer(":memory:", escalation_mode=escalation_mode,
                         **kwargs) as mem:
            mem.remember(base)
            res = mem.remember(obs)

        label_errors += ex.label_errors
        if ex.mismatches:
            mismatch_err.append(abs(ex.mismatches[-1] - hand_mm))

        if res.action == "inserted":
            unrouted.append((domain, note))
            continue
        routed += 1
        got = _got_ur(res.action)
        if got == expected:
            correct += 1
            correct_routed += 1
        elif expected == "R":
            false_update += 1
        else:
            missed_update += 1

    # ── the other half of the ledger: facts that must NOT be linked ──────────
    coexist_ok = 0
    merged = []
    for (da, ta, db, tb, note) in COEXIST_PROBES:
        ex = ProbeExtractor(oracle=oracle)
        kwargs = {"extractor": ex}
        if similarity_fn is not None:
            kwargs["similarity_fn"] = similarity_fn
        if verifier is not None:
            kwargs["link_verifier"] = verifier
            kwargs["link_recall_bar"] = float(
                os.environ.get("LINK_RECALL_BAR", "0.30"))
        with MemoryLayer(":memory:", escalation_mode=escalation_mode,
                         **kwargs) as mem:
            mem.remember(ta)
            mem.remember(tb)
            survivors = len(mem._active())
        if survivors >= 2:
            coexist_ok += 1
        else:
            merged.append((db, note))

    n = len(ESCALATION_PROBES)
    n_co = len(COEXIST_PROBES)
    false_merge = n_co - coexist_ok
    return {
        "routed": routed / n,
        "correct": correct / n,
        "correct_given_routed": (correct_routed / routed) if routed else 0.0,
        "false_update": false_update,
        "missed_update": missed_update,
        "unrouted": unrouted,
        "label_errors": label_errors,
        "mismatch_err": statistics.mean(mismatch_err) if mismatch_err else 0.0,
        "n": n,
        "coexist": coexist_ok / n_co,
        "false_merge": false_merge,
        "merged": merged,
        "n_co": n_co,
        # The weighting the rest of this document argues for: a merge or a
        # wrongly-superseded fact is gone, a duplicate or a stale value is not.
        "irreversible": false_update + false_merge,
        "recoverable": missed_update + len(unrouted),
        "total_correct": (correct + coexist_ok) / (n + n_co),
    }


def main():
    print("=" * 92)
    print("BATTERY F — END-TO-END through remember() (no hand-fed domain or mismatch)")
    print("=" * 92)
    sims = build_similarities()
    print(f"  probes: {len(ESCALATION_PROBES)} one-shot   "
          f"similarity backends: {len(sims)}")

    header = (
        f"  {'similarity':<32}{'labels':>9}{'routed':>9}{'update ok':>11}"
        f"{'coexist ok':>12}{'overall':>9}{'IRREVERSIBLE':>14}{'recoverable':>13}"
    )
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))

    results = {}
    with volatility_profile("real"):
        for sim_label, sim_fn, verifier in sims:
            for cond in ("oracle", "shipped"):
                r = run_config(sim_fn, oracle=(cond == "oracle"),
                               verifier=verifier)
                results[(sim_label, cond)] = r
                print(
                    f"  {sim_label:<32}{cond:>9}{r['routed']:>9.1%}"
                    f"{r['correct']:>11.1%}{r['coexist']:>12.1%}"
                    f"{r['total_correct']:>9.1%}"
                    f"{r['irreversible']:>14d}{r['recoverable']:>13d}"
                )

        print(f"\n  {len(ESCALATION_PROBES)} update probes (a stored fact and a "
              f"statement about it) + {len(COEXIST_PROBES)} coexist probes")
        print("  (two distinct facts that must both survive).")
        print("  IRREVERSIBLE = false updates + false merges: a true memory is gone.")
        print("  recoverable  = missed updates + duplicates: both facts still exist.")
        print("  Ranking on 'overall' alone hides that the two columns are not")
        print("  equally costly; the whole argument for verification is the")
        print("  IRREVERSIBLE column.")

        # ── error budget ──────────────────────────────────────────────────────
        best_label = sims[-1][0]
        base = results[(sims[0][0], "oracle")]
        best_oracle = results[(best_label, "oracle")]
        best_shipped = results[(best_label, "shipped")]

        print("\n" + "=" * 92)
        print("ERROR BUDGET")
        print("=" * 92)
        sim_gain = (best_oracle["correct"] - base["correct"]) * 100
        routing_loss = (1.0 - best_oracle["routed"]) * 100
        decision_loss = (best_oracle["routed"] - best_oracle["correct"]) * 100
        label_loss = (best_oracle["correct"] - best_shipped["correct"]) * 100

        print(f"  similarity function ({sims[0][0]} -> {best_label})"
              f"{sim_gain:>+22.1f} pts")
        print(f"  residual routing failure at best similarity"
              f"{-routing_loss:>+22.1f} pts")
        print(f"  decision error once routed"
              f"{-decision_loss:>+38.1f} pts")
        print(f"  classifier label error (oracle -> shipped)"
              f"{-label_loss:>+23.1f} pts")

        free = results[("hashing (free)", "oracle")]["correct"] if any(
            lbl == "hashing (free)" for lbl, _f, _v in sims) else None
        if free is not None:
            recovered = (free - base["correct"]) * 100
            total = max(sim_gain, 1e-9)
            print(f"\n  Of that similarity gain, {recovered:.1f} pts is recoverable "
                  f"with ZERO new\n  dependencies (hashing backend) — "
                  f"{recovered / total:.0%} of the total.")

        # ── what never routes ────────────────────────────────────────────────
        print(f"\n  Probes that never route even at best similarity "
              f"({len(best_oracle['unrouted'])} of {best_oracle['n']}):")
        for domain, note in best_oracle["unrouted"]:
            print(f"    {domain:<22} {note}")

        print(f"\n  Distinct facts wrongly merged at best similarity "
              f"({len(best_oracle['merged'])} of {best_oracle['n_co']}):")
        for domain, note in best_oracle["merged"]:
            print(f"    {domain:<22} {note}")
        if not best_oracle["merged"]:
            print("    (none)")

        print(f"\n  Mismatch estimation: mean |heuristic - hand-set| = "
              f"{best_oracle['mismatch_err']:.3f}")
        print("  (Batteries A-E supply the hand-set value; this is how far the")
        print("   shipped extractor's own estimate sits from it.)")

        # ── does the escalation mode even matter here? ────────────────────────
        print("\n" + "=" * 92)
        print("ESCALATION MODE, MEASURED END-TO-END")
        print("=" * 92)
        for mode in ("homeostatic", "allostatic"):
            r = run_config(sims[-1][1], oracle=True, escalation_mode=mode,
                           verifier=sims[-1][2])
            print(f"  {mode:<12} routed {r['routed']:.1%}  overall "
                  f"{r['total_correct']:.1%}  irreversible {r['irreversible']}"
                  f"  recoverable {r['recoverable']}")
        print("\n  Compare the spread here against the routing and similarity")
        print("  terms above before spending more effort on the control law.")


if __name__ == "__main__":
    main()
