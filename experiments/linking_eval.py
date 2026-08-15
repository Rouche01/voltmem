"""
Battery G — linking: must-link vs must-not-link, and whether they are separable
==============================================================================

Battery F showed that routing, not the escalation law, dominates the error
budget: a contradicting observation usually never finds the memory it
contradicts, so the control law is never consulted. The obvious response is to
lower the link thresholds until it does. That would be a mistake to make blind.

Lowering a link bar does not only convert duplicates into updates. It also
converts unrelated statements into FALSE MERGES, and a false merge is strictly
worse than a duplicate:

  duplicate    two records of the same fact coexist. Both are retrievable, and
               a later consolidate pass can reconcile them. Recoverable.
  false merge  two DIFFERENT facts are treated as one. The escalation law
               supersedes one of them and a true, unrelated memory is
               destroyed. Not recoverable.

No test in this repo measures the false-merge rate with a real matcher. The
existing linking tests (test_remember_slot_fallback_*, cross_domain_no_false_link)
inject a mocked similarity function, so they verify the threshold ladder rather
than the scorer feeding it. That missing negative control is what makes
threshold tuning unsafe right now, and it is what this battery supplies.

The decisive output is not the link rate — it is SEPARABILITY. If the lowest
must-link similarity sits above the highest must-not-link similarity, some
threshold separates them cleanly and tuning is a solvable problem. If the two
ranges overlap, then no threshold can succeed at both and the scorer itself has
to improve; tuning would only be choosing which error to make.

The must-not-link set is deliberately adversarial in the way that matters:
pairs with HIGH lexical overlap that are nonetheless distinct coexisting facts
("proficient in Python" vs "proficient in Japanese"). Those are the cases where
a lower bar destroys data, and they are exactly what a naive scorer gets wrong.

The corpus (``linking_pairs.py``) is split into the original 24 pairs, which
every threshold and verifier so far was fitted to, and 56 held-out pairs written
afterwards from a structural grid. Both are reported separately: a result that
only holds on the fitted half is a result about hindsight.

Run:
    .venv/bin/python experiments/linking_eval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voltmem import MemoryLayer                                # noqa: E402
from voltmem.domains import (                                  # noqa: E402
    DOMAIN_SIBLINGS,
    DOMAIN_VOLATILITY,
    SLOT_DOMAINS,
    SLOT_LINK_FLOOR,
)
from voltmem.embeddings import EmbeddingSimilarity             # noqa: E402
from voltmem.extract import HeuristicExtractor                 # noqa: E402
from linking_pairs import (                                    # noqa: E402
    MUST_LINK,
    MUST_NOT_LINK,
    SPLITS,
    TRUTH,
)

class OracleExtractor:
    """Classifies by lookup so linking is measured without classifier noise."""

    def __init__(self):
        self._inner = HeuristicExtractor()

    def classify_domain(self, text):
        return TRUTH.get(text) or self._inner.classify_domain(text)

    def mismatch(self, new_text, existing_text, similarity):
        return self._inner.mismatch(new_text, existing_text, similarity)


def slot_threshold(domain, relate_threshold=0.55):
    V_d = DOMAIN_VOLATILITY.get(domain, 0.5)
    return max(SLOT_LINK_FLOOR, relate_threshold - 0.12 - 0.10 * V_d)


def _diagnose(mem, new_domain, new_text):
    """Which rung of the ladder fired, and at what similarity (diagnostic only)."""
    match, sim = mem._best_match_global(new_text, mem.relate_threshold)
    if match is not None:
        return sim, "global"
    slot_match, slot_sim = mem._best_match_in_slot(new_text, new_domain)
    if slot_match is not None:
        return slot_sim, "slot"
    return max(sim, slot_sim), "none"


def build_link_verifier():
    """LINK_VERIFIER=qwen2.5-coder:14b enables two-stage linking in remember().

    Wraps the shipped verifier with the same on-disk response cache Battery H
    uses. Because ``voltmem.verify.VERIFY_PROMPT`` is the prompt both sides
    send, the cache keys line up and re-measuring the integration costs nothing
    for pairs already judged.
    """
    model = os.environ.get("LINK_VERIFIER")
    if not model:
        return None
    from llm_verify_eval import Cache
    from voltmem.verify import LLMLinkVerifier

    class DiskCachedVerifier(LLMLinkVerifier):
        def __init__(self, model):
            super().__init__(model=model)
            self._disk = Cache(model.replace(":", "-").replace("/", "-"))

        def _generate(self, prompt):
            hit = self._disk.get(self.model, prompt)
            if hit is not None:
                return hit
            out = super()._generate(prompt)
            self._disk.put(self.model, prompt, out)
            return out

    return DiskCachedVerifier(model)


def run_pairs(pairs, similarity_fn, verifier=None):
    """Run both statements through remember() and count surviving memories.

    One surviving item means the two statements were treated as the same fact
    (whether superseded or logged as a mismatch — either way only one content
    remains). Two means they coexist. must-link wants 1, must-not-link wants 2.
    This measures actual data loss rather than modelling the threshold ladder.
    """
    out = []
    for (sd, st, nd, nt, note) in pairs:
        kwargs = {"similarity_fn": similarity_fn} if similarity_fn else {}
        if verifier is not None:
            kwargs["link_verifier"] = verifier
            kwargs["link_recall_bar"] = float(
                os.environ.get("LINK_RECALL_BAR", "0.30"))
        with MemoryLayer(":memory:", extractor=OracleExtractor(), **kwargs) as mem:
            mem.remember(st)
            sim, path = _diagnose(mem, nd, nt)
            res = mem.remember(nt)
            survivors = len(mem._active())
        out.append({
            "sim": sim, "path": path, "linked": survivors == 1,
            "survivors": survivors, "action": res.action,
            "stored_domain": sd, "new_domain": nd, "note": note,
            "bar": min(mem.relate_threshold, slot_threshold(nd)),
        })
    return out


def best_threshold(pos_sims, neg_sims):
    """Fewest total errors achievable by any single similarity cutoff."""
    cands = sorted(set(pos_sims) | set(neg_sims) | {0.0, 1.01})
    best = (None, len(pos_sims) + len(neg_sims))
    for t in cands:
        errs = sum(1 for s in pos_sims if s < t) + sum(1 for s in neg_sims if s >= t)
        if errs < best[1]:
            best = (t, errs)
    return best


def build_backends():
    out = [("keyword (default)", None)]
    try:
        out.append(("hashing", EmbeddingSimilarity(backend="hashing")))
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [skip] hashing: {exc}")
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        out.append(("sentence-transformers",
                    EmbeddingSimilarity(backend="sentence-transformers")))
    except Exception:                                          # noqa: BLE001
        print("  [skip] sentence-transformers unavailable (no local cache)")
    out = [(label, fn, None) for label, fn in out]
    verifier = build_link_verifier()
    if verifier is not None and len(out) > 1:
        # Pair the verifier with the strongest recall available: the point of
        # two stages is that recall and precision stop competing.
        best_label, best_fn, _ = out[-1]
        out.append((f"{best_label} + verify", best_fn, verifier))
    return out


def split_slices():
    """(label, pos_slice, neg_slice) — index ranges into the full result lists.

    ``MUST_LINK`` is DEV + HELDOUT in that order, so a slice is enough and every
    pair is only run through ``remember()`` once per backend.
    """
    n_dev_pos = len(SPLITS["dev (fitted)"][0])
    n_dev_neg = len(SPLITS["dev (fitted)"][1])
    return [
        ("dev (fitted)", slice(0, n_dev_pos), slice(0, n_dev_neg)),
        ("held-out", slice(n_dev_pos, None), slice(n_dev_neg, None)),
        ("all", slice(None), slice(None)),
    ]


def main():
    print("=" * 94)
    print("BATTERY G — LINKING: must-link vs must-not-link, and separability")
    print("=" * 94)
    backends = build_backends()
    n_dev = len(SPLITS["dev (fitted)"][0]) + len(SPLITS["dev (fitted)"][1])
    n_held = len(SPLITS["held-out"][0]) + len(SPLITS["held-out"][1])
    print(f"  {len(MUST_LINK)} must-link pairs, {len(MUST_NOT_LINK)} must-not-link "
          f"pairs, {len(backends)} backends")
    print(f"  {n_dev} dev (thresholds/verifiers were fitted here) + {n_held} held-out")
    print(f"  global bar {0.55}; slot bar = max({SLOT_LINK_FLOOR}, "
          f"0.55 - 0.12 - 0.10*V_d)  [lower V_d => HIGHER bar]")

    header = (
        f"  {'backend':<32}{'split':<14}{'linked':>9}{'false merge':>13}"
        f"{'min(pos)':>10}{'max(neg)':>10}{'separable':>11}{'best-case errs':>16}"
    )
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))

    summary = {}
    for label, fn, verifier in backends:
        pos = run_pairs(MUST_LINK, fn, verifier)
        neg = run_pairs(MUST_NOT_LINK, fn, verifier)
        for sname, pslice, nslice in split_slices():
            p, n = pos[pslice], neg[nslice]
            linked = sum(1 for r in p if r["linked"])
            merged = sum(1 for r in n if r["linked"])
            pos_sims = [r["sim"] for r in p]
            neg_sims = [r["sim"] for r in n]
            t, errs = best_threshold(pos_sims, neg_sims)
            separable = min(pos_sims) > max(neg_sims)
            if sname == "all":
                summary[label] = (pos, neg, t, errs, separable)
            print(
                f"  {label if sname == 'dev (fitted)' else '':<32}{sname:<14}"
                f"{linked:>4}/{len(p):<4}{merged:>8}/{len(n):<4}"
                f"{min(pos_sims):>10.2f}{max(neg_sims):>10.2f}"
                f"{str(separable):>11}{errs:>10} @ t={t:.2f}"
            )
        print()

    print("  'linked'      = must-link pairs the shipped ladder actually links.")
    print("  'false merge' = must-not-link pairs it wrongly links (destroys a fact).")
    print("  'separable'   = every must-link scores above every must-not-link, i.e.")
    print("                  some threshold gets both right. 'best-case errs' is the")
    print("                  fewest errors ANY single cutoff can achieve.")

    # ── detail per backend; the default row matters most, it is what ships ────
    for label, _fn, _v in backends:
        pos, neg, t, errs, separable = summary[label]
        print("\n" + "=" * 94)
        print(f"DETAIL — {label}")
        print("=" * 94)
        print("\n  must-link failures (duplicate created, escalation never consulted):")
        any_fail = False
        for r in pos:
            if not r["linked"]:
                any_fail = True
                print(f"    sim={r['sim']:.2f} bar={r['bar']:.2f} "
                      f"{r['new_domain']:<21} {r['note']}")
        if not any_fail:
            print("    (none)")

        print("\n  must-not-link failures (A TRUE MEMORY WAS LOST):")
        any_fail = False
        for r in neg:
            if r["linked"]:
                any_fail = True
                print(f"    sim={r['sim']:.2f} bar={r['bar']:.2f} via {r['path']:<7}"
                      f" {r['action']:<16} {r['note']}")
        if not any_fail:
            print("    (none)")

    print("\n" + "=" * 94)
    print("VERDICT")
    print("=" * 94)
    if separable:
        print(f"  SEPARABLE at t={t:.2f} with {errs} errors. The scorer carries enough")
        print("  signal to get both sides right — this is a threshold-placement")
        print("  problem, and retuning the bars is safe to attempt.")
    else:
        print(f"  NOT SEPARABLE. The best any single cutoff can do is {errs} errors")
        print(f"  (at t={t:.2f}), because must-link and must-not-link similarity")
        print("  ranges overlap. Lowering the bar to catch the career change WILL")
        print("  merge unrelated facts. Threshold tuning can only choose which")
        print("  error to make; closing this needs a better matching signal")
        print("  (e.g. entity/slot awareness, or an LLM judge on the candidate pair)")
        print("  rather than a different number.")


if __name__ == "__main__":
    main()
