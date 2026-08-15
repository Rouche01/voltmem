"""
Conservative extract-then-join, then mocked sleeptime
=====================================================

Write-path join is code-only on the cached extracts. It prefers a duplicate
over a wrong overwrite:

  * named ending of an entity that is not the stored value → KEEP_BOTH
  * generic slots (current_task / mood / project / manager) join only with
    value overlap, anaphora, or a positive manager fill
  * drifted slugs are not aliased

Sleeptime column: for pairs the write path refused, reuse the Battery H
qwen2.5-coder:14b pair-verifier cache (no new model calls). That mocks the
nightly twin-reconciliation job. ``consolidate`` does not do this today.

Pass (write path): 0 false merges on held-out. Must-link may drop.
Pass (write + sleeptime): near 49/56 and still 0 false merges.

Run:
    .venv/bin/python experiments/structured_join_eval.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voltmem.structure import (                                # noqa: E402
    LLMStructuredExtractor,
    HeuristicStructuredExtractor,
    join_structured,
)
from llm_verify_eval import Cache, PROMPT_V2, parse                    # noqa: E402
from linking_pairs import SPLITS                                       # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_PATH = os.path.join(CACHE_DIR, "structured_extract_v1.json")
VERIFY_MODEL = "qwen2.5-coder:14b"

WRITE_PASS_MERGES = 0
NIGHT_PASS_TOTAL = 49
NIGHT_PASS_MERGES = 0

SPOTLIGHT = [
    ("career change (must-link)", True,
     "User works as a data analyst",
     "User explicitly said they changed careers and now work as a nurse"),
    ("Python vs Japanese (must-not)", False,
     "User is proficient in Python",
     "User is proficient in Japanese"),
    ("plain correction: birth year", True,
     "User was born in 1990",
     "User was born in 1991"),
    ("coexisting slots: dentist vs flight", False,
     "User has a dentist appointment on Tuesday",
     "User has a flight on Friday"),
    ("coexisting slots: parents' city", False,
     "User lives in Berlin",
     "User's parents live in Hamburg"),
    ("marker on wrong entity", False,
     "User reports to Dana",
     "User no longer reports to Miguel"),
]


def _load_extract_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as fh:
            return json.load(fh)
    return {}


def _fmt_facts(facts):
    if not facts:
        return "(empty extract)"
    return " | ".join(
        f"{f.subject}/{f.attribute}={f.value}[{f.cardinality}"
        f"{' replaces' if f.replaces else ''}]"
        for f in facts
    )


def _join(extractor, stored, new, conservative):
    return join_structured(
        extractor.extract(stored), extractor.extract(new),
        new, stored, conservative=conservative,
    )


class SleeptimeMock:
    """Battery H cache only. A miss stays KEEP_BOTH (the write-path decision)."""

    def __init__(self):
        self.cache = Cache("qwen2.5-coder-14b")
        self.hits = self.misses = 0

    def would_update(self, stored, new):
        raw = self.cache.get(VERIFY_MODEL, PROMPT_V2.format(stored=stored, new=new))
        if raw is None:
            self.misses += 1
            return False
        self.hits += 1
        parsed = parse(raw)
        return bool(parsed and parsed[0] == "UPDATE")


def _score(extractor, pos_pairs, neg_pairs, conservative, night=None):
    linked = merged = 0
    misses, kills = [], []
    for _sd, st, _nd, nt, note in pos_pairs:
        ok = _join(extractor, st, nt, conservative)
        if (not ok) and night is not None and night.would_update(st, nt):
            ok = True
        if ok:
            linked += 1
        else:
            misses.append(note)
    for _sd, st, _nd, nt, note in neg_pairs:
        ok = _join(extractor, st, nt, conservative)
        if (not ok) and night is not None and night.would_update(st, nt):
            ok = True
        if ok:
            merged += 1
            kills.append(note)
    return linked, merged, misses, kills


def _report(label, linked, merged, n_pos, n_neg, misses, kills, show_detail=True):
    total = linked + (n_neg - merged)
    print(f"  {label:<28} linked {linked:>2}/{n_pos}   "
          f"false merges {merged:>2}/{n_neg}   "
          f"total {total}/{n_pos + n_neg}")
    if show_detail and misses:
        print("    missed links:")
        for note in misses:
            print(f"      {note}")
    if show_detail and kills:
        print("    false merges:")
        for note in kills:
            print(f"      {note}")
    return total


def main():
    print("=" * 84)
    print("CONSERVATIVE JOIN + MOCKED SLEEPTIME — cached extracts, no new calls")
    print("=" * 84)

    cache = _load_extract_cache()
    extractor = LLMStructuredExtractor(disk_cache=cache)
    held_pos, held_neg = SPLITS["held-out"]
    missing = sum(
        1 for pairs in (held_pos, held_neg)
        for _sd, st, _nd, nt, _n in pairs
        for t in (st, nt)
        if t not in cache
    )
    print(f"  extract cache {len(cache)} entries  "
          f"held-out statements missing {missing}")
    if missing:
        print("  FAIL — extract cache incomplete. Re-run the first join eval.")
        return

    night = SleeptimeMock()
    for pairs in (held_pos, held_neg):
        for _sd, st, _nd, nt, _n in pairs:
            night.would_update(st, nt)
    print(f"  sleeptime cache {night.hits}/56 held-out hits, {night.misses} misses")
    night.hits = night.misses = 0

    print("\n  spotlight (conservative write path):")
    cells_ok = True
    for name, want, stored, new in SPOTLIGHT:
        got = _join(extractor, stored, new, True)
        mark = "ok" if got == want else "MISS"
        if got != want and want is False:
            cells_ok = False
        print(f"  [{mark}] {name}  want={'UPDATE' if want else 'KEEP_BOTH'}  "
              f"got={'UPDATE' if got else 'KEEP_BOTH'}")
        if got != want:
            print(f"        stored {_fmt_facts(extractor.extract(stored))}")
            print(f"        new    {_fmt_facts(extractor.extract(new))}")

    print("\n  held-out 56")
    rows = []
    for label, conservative, use_night, detail in (
        ("naive join (baseline)", False, False, False),
        ("conservative write path", True, False, True),
        ("write + mocked sleeptime", True, True, True),
    ):
        linked, merged, misses, kills = _score(
            extractor, held_pos, held_neg, conservative,
            night=night if use_night else None,
        )
        total = _report(label, linked, merged, 28, 28, misses, kills, detail)
        rows.append((label, linked, merged, total))

    print("\n  dev 24 (fitted extracts; not the honest number)")
    dev_pos, dev_neg = SPLITS["dev (fitted)"]
    for label, conservative, use_night in (
        ("naive join", False, False),
        ("conservative write path", True, False),
        ("write + mocked sleeptime", True, True),
    ):
        linked, merged, misses, kills = _score(
            extractor, dev_pos, dev_neg, conservative,
            night=night if use_night else None,
        )
        _report(label, linked, merged, 12, 12, misses, kills, show_detail=False)

    print("\n  heuristic extract + conservative join (millisecond write path)")
    hx = HeuristicStructuredExtractor()
    h_link, h_merge, h_miss, h_kill = _score(hx, held_pos, held_neg, True)
    _report("heuristic write path", h_link, h_merge, 28, 28, h_miss, h_kill, True)
    hn_link, hn_merge, hn_miss, hn_kill = _score(
        hx, held_pos, held_neg, True, night=night)
    _report("heuristic + mocked sleeptime", hn_link, hn_merge, 28, 28,
            hn_miss, hn_kill, True)

    write = next(r for r in rows if r[0].startswith("conservative"))
    plus = next(r for r in rows if r[0].startswith("write +"))
    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    print(f"  write path:     {write[1]}/28 links, {write[2]}/28 merges  "
          f"({write[3]}/56)")
    print(f"  write+sleeptime:{plus[1]}/28 links, {plus[2]}/28 merges  "
          f"({plus[3]}/56)")
    print(f"  heuristic:      {h_link}/28 links, {h_merge}/28 merges  "
          f"({h_link + (28 - h_merge)}/56)")
    print(f"  heuristic+night:{hn_link}/28 links, {hn_merge}/28 merges  "
          f"({hn_link + (28 - hn_merge)}/56)")
    print(f"  write-path bar: {WRITE_PASS_MERGES} false merges "
          "(must-link may drop)")
    print(f"  night bar:      ≥{NIGHT_PASS_TOTAL}/56 and "
          f"{NIGHT_PASS_MERGES} false merges")

    write_ok = write[2] <= WRITE_PASS_MERGES
    heur_ok = h_merge <= WRITE_PASS_MERGES
    night_ok = plus[3] >= NIGHT_PASS_TOTAL and plus[2] <= NIGHT_PASS_MERGES
    if write_ok:
        print("  WRITE PATH PASS — 0 irreversible errors. Specific cards join now;")
        print("         generic / named-ending cases wait for sleeptime.")
    else:
        print("  WRITE PATH FAIL — a generic-slot pair still merged.")
        print("         Tighten the overlap rule. Do not add a model.")
    if night_ok:
        print("  SLEEPTIME PASS — the cached 14B recovers the twins without")
        print("         re-introducing a merge. Twin-reconciliation is worth wiring.")
    else:
        print("  SLEEPTIME FAIL — night does not recover enough, or it merges.")
        print("         The 14B stays on the live write if you need 49/56 now.")
    if heur_ok:
        print("  HEURISTIC PASS — 0 irreversible errors in milliseconds.")
        print("         Unrecognised frames insert; sleeptime may still merge.")
    else:
        print("  HEURISTIC FAIL — a pattern overwrote a trap. Narrow the frames.")


if __name__ == "__main__":
    main()
