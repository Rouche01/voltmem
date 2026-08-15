"""
Recall-then-verify prototype — is two-stage linking the right architecture?
==========================================================================

Battery G established that no single similarity cutoff can separate must-link
from must-not-link pairs: the false merges score HIGHER than the true links
("proficient in Python" vs "proficient in Japanese" = 0.80, while the career
change that must link = 0.25). One threshold cannot win, so the fix has to add
a second signal rather than move a number.

The candidate architecture is standard information retrieval: use similarity for
RECALL with a deliberately low bar, then apply a VERIFIER for precision.

    stage 1  sim(new, stored) >= recall_bar        -> candidate
    stage 2  verify(new, stored, domain) == True   -> link

This script answers three questions in order, because they have different
costs and only the third one requires spending anything:

  1. CEILING. With a perfect verifier, how much does stage 1 deliver? If
     recall at a low bar is not near-perfect, no verifier can rescue the
     design and the similarity function itself is the problem.
  2. FLOOR. How far does a free, dependency-free verifier get? Two signals
     are tested, both derived from what Battery G's traps actually look like:
       cardinality     — single-valued attributes (the mood, the location, the
                         current task) can only hold one value, so a competing
                         statement about them is necessarily a replacement.
                         VoltMem already encodes this as SLOT_DOMAINS.
       change marker   — replacement language ("no longer", "changed", "now",
                         "instead") distinguishes "I switched" from "here is
                         another one".
  3. GAP. What is left for an LLM verifier to earn? That is the number that
     justifies (or kills) paying for a model call per candidate pair.

Battery G's trap set includes pairs written specifically to defeat the
change-marker signal ("User no longer works with Bob" against a stored fact
about Alice) so the cheap verifier's weakness shows up here rather than in
production.

IMPORTANT — dev vs held-out. Both of the cheap verifier's signals were chosen
after reading which of the original 24 pairs failed, so its score on those is
fitted and can only be quoted as an upper bound. ``linking_pairs.py`` adds 56
held-out pairs written from a structural grid afterwards, and every table below
reports the splits side by side. The recall (stage-1) numbers are the exception:
nothing was ever tuned against them, so they are honest on both splits.

Run:
    .venv/bin/python experiments/link_verify_prototype.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voltmem.domains import SLOT_DOMAINS                       # noqa: E402
from voltmem.embeddings import EmbeddingSimilarity             # noqa: E402
from linking_pairs import SPLITS                               # noqa: E402

RECALL_BARS = [0.60, 0.50, 0.40, 0.30, 0.20, 0.10]

# Replacement language. Deliberately short and boring — if this needs to grow
# to work, that is evidence for the LLM verifier rather than against it.
CHANGE_MARKERS = (
    "no longer", "not anymore", "used to", "changed", "switched", "instead",
    "moved", "finished", "quit", "left", "now", "actually", "these days",
    "as of", "since then", "updated",
)
MARKER_RE = re.compile("|".join(re.escape(m) for m in CHANGE_MARKERS))


def has_change_marker(text: str) -> bool:
    return bool(MARKER_RE.search(text.lower()))


def single_valued(domain: str) -> bool:
    """Can this attribute hold only one value at a time?

    Reuses VoltMem's existing SLOT_DOMAINS rather than inventing a taxonomy:
    the mood, the location, the current task, a transient fact.
    """
    return domain in SLOT_DOMAINS


# ── verifiers ─────────────────────────────────────────────────────────────────

def verify_always(new, stored, domain, sim):
    """No verifier — stage 1 alone. Shows what the recall bar does unaided."""
    return True


def verify_oracle(new, stored, domain, sim, *, truth):
    """Perfect verifier. Establishes the architecture's ceiling."""
    return truth


def verify_cardinality(new, stored, domain, sim):
    """Single-valued attributes replace; multi-valued ones coexist."""
    return single_valued(domain)


def verify_cardinality_or_marker(new, stored, domain, sim):
    """Single-valued replaces; otherwise require explicit replacement language."""
    if single_valued(domain):
        return True
    return has_change_marker(new)


VERIFIERS = [
    ("stage 1 only (no verify)", verify_always),
    ("cardinality", verify_cardinality),
    ("cardinality + change marker", verify_cardinality_or_marker),
]


def build_backends():
    out = [("keyword", None)]
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        out.append(("sentence-transformers",
                    EmbeddingSimilarity(backend="sentence-transformers")))
    except Exception:                                          # noqa: BLE001
        print("  [skip] sentence-transformers unavailable")
    return out


def keyword_similarity():
    """The shipped default scorer, without constructing a MemoryLayer."""
    from voltmem.memory import MemoryLayer
    mem = MemoryLayer(":memory:")
    fn = mem._similarity
    return fn


def evaluate(sim_fn, recall_bar, verifier, pairs, use_oracle=False):
    """Returns (linked_correct, coexist_correct, n_pos, n_neg, recalled)."""
    must_link, must_not_link = pairs
    linked_ok = coexist_ok = recalled = 0
    for (sd, st, nd, nt, _note) in must_link:
        s = sim_fn(nt, st)
        if s < recall_bar:
            continue                                 # missed at stage 1
        recalled += 1
        ok = (verifier(nt, st, nd, s, truth=True) if use_oracle
              else verifier(nt, st, nd, s))
        linked_ok += bool(ok)
    for (sd, st, nd, nt, _note) in must_not_link:
        s = sim_fn(nt, st)
        if s < recall_bar:
            coexist_ok += 1                          # never a candidate: correct
            continue
        ok = (verifier(nt, st, nd, s, truth=False) if use_oracle
              else verifier(nt, st, nd, s))
        coexist_ok += (not ok)
    return linked_ok, coexist_ok, len(must_link), len(must_not_link), recalled


def cell(new_domain: str, new_text: str) -> str:
    """The structural cell a pair falls in — the two axes the verifier reads."""
    return (f"{'slot' if single_valued(new_domain) else 'multi'}"
            f"/{'marker' if has_change_marker(new_text) else 'no-marker'}")


def cell_breakdown(sim_fn, recall_bar, verifier, pairs):
    """Per-cell correct/total, so a failure is attributable to a signal."""
    must_link, must_not_link = pairs
    out: dict[str, list[int]] = {}
    for side, plist, want_link in (("link", must_link, True),
                                   ("coexist", must_not_link, False)):
        for (sd, st, nd, nt, _note) in plist:
            key = f"{side:<8}{cell(nd, nt)}"
            row = out.setdefault(key, [0, 0])
            row[1] += 1
            s = sim_fn(nt, st)
            linked = s >= recall_bar and verifier(nt, st, nd, s)
            row[0] += (linked == want_link)
    return out


BAR = 0.20   # recall bar chosen on dev; re-checked against held-out below


def main():
    print("=" * 94)
    print("RECALL-THEN-VERIFY PROTOTYPE — measured against Battery G")
    print("=" * 94)
    backends = build_backends()
    for sname, (ml, mnl) in SPLITS.items():
        print(f"  {sname:<14} {len(ml):>3} must-link  {len(mnl):>3} must-not-link")
    print("\n  'dev' is the set the verifier's two signals were chosen from, so its")
    print("  score there is an upper bound. 'held-out' is the honest number.")

    for label, fn in backends:
        sim_fn = keyword_similarity() if fn is None else fn
        print("\n" + "-" * 94)
        print(f"  BACKEND: {label}")
        print("-" * 94)

        # ── question 1: ceiling ───────────────────────────────────────────────
        print(f"\n  stage-1 recall and the ceiling a perfect verifier would reach:")
        header = (f"    {'recall bar':>10}" +
                  "".join(f"{s + ' recall':>22}" for s in SPLITS))
        print(header)
        print("    " + "-" * (len(header) - 4))
        for bar in RECALL_BARS:
            row = f"    {bar:>10.2f}"
            for sname, pairs in SPLITS.items():
                lo, co, npos, nneg, rec = evaluate(sim_fn, bar, verify_oracle,
                                                   pairs, use_oracle=True)
                row += f"{rec:>13}/{npos:<3} ({lo + co:>2}/{npos + nneg})".rjust(22)
            print(row)
        print("    recall = must-link pairs surviving stage 1; (n/m) = oracle total")

        # ── question 2: floor, dev vs held-out ───────────────────────────────
        print(f"\n  At recall bar {BAR:.2f} — what each verifier achieves per split:")
        header = (f"    {'verifier':<30}" +
                  "".join(f"{s:>16}" for s in SPLITS))
        print(header)
        print("    " + "-" * (len(header) - 4))
        for vlabel, vfn in VERIFIERS + [("oracle (ceiling)", verify_oracle)]:
            row = f"    {vlabel:<30}"
            for sname, pairs in SPLITS.items():
                use_oracle = vfn is verify_oracle
                lo, co, npos, nneg, _ = evaluate(sim_fn, BAR, vfn, pairs,
                                                 use_oracle=use_oracle)
                row += f"{lo + co:>10}/{npos + nneg:<5}"
            print(row)

    # ── question 3: which signal breaks, and on what ──────────────────────────
    sim_fn = keyword_similarity() if backends[-1][1] is None else backends[-1][1]
    blabel = backends[-1][0]
    print("\n" + "=" * 94)
    print(f"CELL BREAKDOWN — cardinality + change marker, {blabel}, held-out only")
    print("=" * 94)
    print("  A cell that fails wholesale means the signal is wrong for that shape,")
    print("  not that a threshold is misplaced.\n")
    cells = cell_breakdown(sim_fn, BAR, verify_cardinality_or_marker,
                           SPLITS["held-out"])
    for key in sorted(cells):
        ok, total = cells[key]
        flag = "" if ok == total else "   <-- signal wrong for this shape"
        print(f"    {key:<26}{ok:>3}/{total:<3}{flag}")

    print("\n" + "=" * 94)
    print(f"WHERE THE FREE VERIFIER STILL FAILS  ({blabel}, held-out)")
    print("=" * 94)
    ml, mnl = SPLITS["held-out"]
    print("\n  must-link missed (duplicate would be created):")
    none = True
    for (sd, st, nd, nt, note) in ml:
        s = sim_fn(nt, st)
        if s < BAR or not verify_cardinality_or_marker(nt, st, nd, s):
            none = False
            why = "stage 1 missed it" if s < BAR else "verifier refused"
            print(f"    sim={s:.2f} [{why}] {nd:<21} {note}")
            print(f"      stored: {st}")
            print(f"      new:    {nt}")
    if none:
        print("    (none)")

    print("\n  must-not-link wrongly linked (a true memory would be lost):")
    none = True
    for (sd, st, nd, nt, note) in mnl:
        s = sim_fn(nt, st)
        if s >= BAR and verify_cardinality_or_marker(nt, st, nd, s):
            none = False
            print(f"    sim={s:.2f} {nd:<21} {note}")
            print(f"      stored: {st}")
            print(f"      new:    {nt}")
    if none:
        print("    (none)")

    print("\n" + "=" * 94)
    print("READING")
    print("=" * 94)
    print("  Two claims are on trial. (1) The ARCHITECTURE: does a low recall bar")
    print("  keep nearly every must-link pair alive for a verifier to judge? That")
    print("  is the recall column, and it was never fitted to anything. (2) The")
    print("  CHEAP VERIFIER: does cardinality + change marker beat the best single")
    print("  threshold? Only the held-out column can answer that; if held-out is")
    print("  far below dev, 19/24 was hindsight and the verifier has to get better")
    print("  signals rather than the design being wrong.")


if __name__ == "__main__":
    main()
