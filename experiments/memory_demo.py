"""
VoltMem memory demo — a felt walkthrough of why volatility-aware memory matters
===============================================================================

A short, hand-scripted conversation history for an LLM assistant, run through
four memory policies. At the end we ask three natural-language questions and show
each policy's answer next to the ground truth.

The point (proven at scale in llm_memory_bench.py, shown concretely here):

  * VoltMem answers all three correctly.
  * always_overwrite gets corrupted by a confident-but-false blip on a STABLE
    fact (thinks the user moved cities).
  * never_overwrite goes stale on a VOLATILE fact (still thinks the user is on
    their old project).
  * reliability-threshold misses a weak-but-true update on a volatile fact.

Retrieval uses real embeddings when available (sentence-transformers / Ollama),
falling back to a deterministic offline scorer so the demo always runs.

Run:
    .venv/bin/python experiments/memory_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voltmem import MemoryLayer                    # noqa: E402
from voltmem.embeddings import EmbeddingSimilarity  # noqa: E402
from voltmem.domains import SOURCE_RELIABILITY     # noqa: E402

# ── scripted history ─────────────────────────────────────────────────────────
# Each event: (attr, domain, value, source, mismatch, true_after, note)
#   true_after = the ground-truth value of this attr AFTER the event
SEED = [
    ("home_city",  "biographical",      "Berlin",
     "explicit_statement", "User: I live in Berlin."),
    ("comm_style", "personality_trait", "concise bullet points",
     "explicit_statement", "User: Keep answers short — bullet points."),
    ("task",       "current_task",      "reviewing PR #42",
     "explicit_statement", "User: Right now I'm reviewing PR #42."),
]

TIMELINE = [
    # weak but TRUE update on a highly volatile fact — VoltMem adopts (low
    # threshold), reliability misses it (weak source below its trust cutoff)
    ("task", "current_task", "preparing the launch demo", "weak_inference", 0.85,
     True, "Behaviour suggests they've switched to preparing the launch demo."),
    # confident but FALSE blip on a stable fact — VoltMem resists, always adopts
    ("home_city", "biographical", "Lisbon", "explicit_statement", 0.85,
     False, "A meeting note claims 'user is based in Lisbon' (out of context)."),
    # weak FALSE blip on a stable fact — everyone except always resists
    ("comm_style", "personality_trait", "long detailed essays", "weak_inference",
     0.85, False, "One long message is misread as a style change."),
]

GROUND_TRUTH = {
    "home_city": "Berlin",
    "comm_style": "concise bullet points",
    "task": "preparing the launch demo",
}

QUESTIONS = [
    ("home_city",  "Where does the user live?"),
    ("task",       "What is the user's current task?"),
    ("comm_style", "How does the user like responses formatted?"),
]


# ── naive baselines (belief dict keyed by attr) ──────────────────────────────
class Naive:
    def __init__(self, policy):
        self.policy = policy
        self.belief = {}

    def seed(self, attr, value):
        self.belief[attr] = value

    def observe(self, attr, value, source, mismatch):
        if self.policy == "always":
            self.belief[attr] = value
        elif self.policy == "never":
            pass
        elif self.policy == "reliability":
            if SOURCE_RELIABILITY.get(source, 0.5) >= 0.7:
                self.belief[attr] = value

    def answer(self, attr):
        return self.belief.get(attr, "(unknown)")


def banner(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def main():
    sim = EmbeddingSimilarity(verbose=True)
    vmem = MemoryLayer(":memory:", similarity_fn=sim)
    attr_domain = {}
    naive = {p: Naive(p) for p in ("always", "never", "reliability")}

    banner("SEEDING what the assistant knows at the start")
    for attr, domain, value, source, note in SEED:
        vmem.write(value, domain=domain, source=source, tags=[attr])
        attr_domain[attr] = domain
        for n in naive.values():
            n.seed(attr, value)
        print(f"  [{domain}] {attr} = {value!r}")
        print(f"     {note}")

    banner("NEW OBSERVATIONS arrive over the following days")
    for attr, domain, value, source, mismatch, is_true, note in TIMELINE:
        res = vmem.observe(value, domain=domain, mismatch_magnitude=mismatch,
                           source=source, tags=[attr])
        for n in naive.values():
            n.observe(attr, value, source, mismatch)
        kind = "TRUE update" if is_true else "FALSE blip "
        print(f"\n  ({kind}, source={source}) {note}")
        print(f"     proposed: {attr} -> {value!r}")
        verdict = {"audited": "UPDATED memory",
                   "logged_mismatch": "kept old value (logged mismatch)",
                   "confirmed": "confirmed"}.get(res.action, res.action)
        print(f"     VoltMem decision: {verdict}")

    banner("FINAL Q&A — each policy's answer vs the ground truth")
    header = f"  {'question':<42}{'VoltMem':<22}{'always':<20}" \
             f"{'never':<20}{'reliability':<20}{'truth':<22}"
    print(header)
    print("  " + "-" * (len(header)))
    score = {"VoltMem": 0, "always": 0, "never": 0, "reliability": 0}
    for attr, q in QUESTIONS:
        r = vmem.retrieve(q, domain=attr_domain[attr], top_k=1)
        v_ans = r.items[0].content if r.items else "(unknown)"
        truth = GROUND_TRUTH[attr]
        row = {
            "VoltMem": v_ans,
            "always": naive["always"].answer(attr),
            "never": naive["never"].answer(attr),
            "reliability": naive["reliability"].answer(attr),
        }
        for k, ans in row.items():
            score[k] += int(ans == truth)
        print(f"  {q:<42}{_mark(row['VoltMem'], truth):<22}"
              f"{_mark(row['always'], truth):<20}{_mark(row['never'], truth):<20}"
              f"{_mark(row['reliability'], truth):<20}{truth:<22}")

    print("\n  score (correct / 3):")
    for k in ("VoltMem", "always", "never", "reliability"):
        print(f"    {k:<14} {score[k]}/3")

    print("\n  Takeaway: only volatility-aware memory answers all three — it "
          "protected the stable facts from confident-but-false blips AND tracked "
          "the volatile fact from a weak-but-true signal. Each naive policy fails "
          "at least one. (Quantified over many runs in llm_memory_bench.py.)")
    vmem.close()


def _mark(ans, truth):
    ok = "OK " if ans == truth else "XX "
    short = ans if len(ans) <= 15 else ans[:14] + "\u2026"
    return f"{ok}{short}"


if __name__ == "__main__":
    main()
