"""
Shipped surface — remember() as it actually defaults, and recall() with twins.
==============================================================================

Matching evals scored extract/join and a mocked 14B in isolation. This scores
the library a caller actually gets:

  MemoryLayer(similarity_fn=embeddings)   # composite, verify_on_write=False
  remember(stored); remember(new)
  recall(new)

Then the same store after ``reconcile_twins`` with the Battery H 14B cache
(no new model calls).

Unmeasured until now:

  * does a live twin hide the new fact at rank 1?
  * do must-not-link twins both stay retrievable?
  * does sleeptime clean must-link twins without eating coexist pairs?

Run:
    .venv/bin/python experiments/shipped_surface_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voltmem import MemoryLayer                                          # noqa: E402
from voltmem.embeddings import EmbeddingSimilarity                       # noqa: E402
from voltmem.maintenance import MaintenanceWindow, reconcile_twins       # noqa: E402
from llm_verify_eval import Cache, PROMPT_V2, parse, load_dotenv         # noqa: E402
from linking_pairs import SPLITS                                         # noqa: E402

VERIFY_MODEL = "qwen2.5-coder:14b"


class CachedVerifier:
    def __init__(self):
        self.cache = Cache("qwen2.5-coder-14b")
        self.misses = 0

    def verify(self, new_text, stored_text, domain):
        raw = self.cache.get(
            VERIFY_MODEL, PROMPT_V2.format(stored=stored_text, new=new_text))
        if raw is None:
            self.misses += 1
            return False
        parsed = parse(raw)
        return bool(parsed and parsed[0] == "UPDATE")


def _hit(haystack, needle):
    return any(needle == h or needle in h or h in needle for h in haystack)


def _score_pair(sim_fn, stored, new, must_link, sleeptime):
    kwargs = {}
    if sim_fn is not None:
        kwargs["similarity_fn"] = sim_fn
    with MemoryLayer(":memory:", **kwargs) as mem:
        mem.remember(stored)
        mem.remember(new)
        n_live = len(mem._active())
        if sleeptime is not None:
            mw = MaintenanceWindow(mem)
            mw.register(
                "twins",
                lambda ctx: reconcile_twins(ctx, verifier=sleeptime),
            )
            mw.run_once("twins")
        n = len(mem._active())
        hits_new = mem.recall(new, top_k=5)
        hits_old = mem.recall(stored, top_k=5)
        new_top1 = bool(hits_new) and (
            hits_new[0] == new or new in hits_new[0] or hits_new[0] in new)
        new_found = _hit(hits_new, new)
        old_found = _hit(hits_old, stored)
        return {
            "twins_live": n_live == 2,
            "active": n,
            "merged": n == 1,
            "new_top1": new_top1,
            "new_found": new_found,
            "old_found": old_found,
            "must_link": must_link,
        }


def _report(label, rows, pos, neg):
    linked = sum(1 for r, p in zip(rows, pos) if r["merged"])
    merged = sum(1 for r, _n in zip(rows[len(pos):], neg) if r["merged"])
    twins = sum(1 for r in rows if r["twins_live"])
    stale = [
        note for r, (_sd, _st, _nd, _nt, note) in zip(rows[:len(pos)], pos)
        if r["twins_live"] and r["new_found"] and not r["new_top1"]
    ]
    hidden = [
        note for r, (_sd, _st, _nd, _nt, note) in zip(rows[:len(pos)], pos)
        if not r["new_found"]
    ]
    lost_coexist = [
        note for r, (_sd, _st, _nd, _nt, note) in zip(rows[len(pos):], neg)
        if not r["old_found"] or not r["new_found"]
    ]
    print(f"  {label}")
    print(f"    remember()  linked {linked}/{len(pos)}   "
          f"false merges {merged}/{len(neg)}   "
          f"live twins {twins}/{len(rows)}")
    print(f"    recall()    new fact in top-5 {len(pos) - len(hidden)}/{len(pos)}"
          f"   stale rank-1 on twins {len(stale)}"
          f"   coexist both findable {len(neg) - len(lost_coexist)}/{len(neg)}")
    if hidden:
        print("    new fact missing from recall:")
        for note in hidden:
            print(f"      {note}")
    if stale:
        print("    twin: new query ranks the old fact first:")
        for note in stale:
            print(f"      {note}")
    if lost_coexist:
        print("    coexist pair not both retrievable:")
        for note in lost_coexist:
            print(f"      {note}")
    return linked, merged, twins, len(stale), len(hidden), len(lost_coexist)


def main():
    load_dotenv()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    print("=" * 84)
    print("SHIPPED SURFACE — composite + heuristic write + sleeptime twins")
    print("=" * 84)

    pos, neg = SPLITS["held-out"]
    sleeptime = CachedVerifier()
    print(f"  14B cache {len(sleeptime.cache.data)} entries")

    print("\n  keyword MemoryLayer() (no embedder)")
    kw = [_score_pair(None, st, nt, True, None)
          for _sd, st, _nd, nt, _n in pos]
    kw += [_score_pair(None, st, nt, False, None)
           for _sd, st, _nd, nt, _n in neg]
    _report("keyword live", kw, pos, neg)

    try:
        sim = EmbeddingSimilarity(backend="sentence-transformers")
    except Exception as exc:
        print(f"  [skip] embeddings: {exc}")
        return

    print("\n  embeddings, verify_on_write=False (shipped)")
    live = [_score_pair(sim, st, nt, True, None)
            for _sd, st, _nd, nt, _n in pos]
    live += [_score_pair(sim, st, nt, False, None)
             for _sd, st, _nd, nt, _n in neg]
    _report("embeddings live", live, pos, neg)

    print("\n  embeddings + reconcile_twins (cached 14B)")
    night = [_score_pair(sim, st, nt, True, sleeptime)
             for _sd, st, _nd, nt, _n in pos]
    night += [_score_pair(sim, st, nt, False, sleeptime)
              for _sd, st, _nd, nt, _n in neg]
    _report("after sleeptime", night, pos, neg)
    if sleeptime.misses:
        print(f"  cache misses {sleeptime.misses} (treated as KEEP_BOTH)")

    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    live_merge = sum(1 for r in live[len(pos):] if r["merged"])
    night_merge = sum(1 for r in night[len(pos):] if r["merged"])
    live_link = sum(1 for r in live[:len(pos)] if r["merged"])
    night_link = sum(1 for r in night[:len(pos)] if r["merged"])
    hidden = sum(1 for r in live[:len(pos)] if not r["new_found"])
    print(f"  live embeddings:  {live_link}/28 links, {live_merge}/28 merges")
    print(f"  after sleeptime:  {night_link}/28 links, {night_merge}/28 merges")
    print(f"  new fact missing from recall before sleeptime: {hidden}/28")
    if live_merge or night_merge:
        print("  FAIL — a coexist pair was destroyed.")
    elif hidden:
        print("  Twins are safe (0 merges) but some new facts do not retrieve")
        print("         until sleeptime. That is the latency cost of the default.")
    else:
        print("  PASS — 0 false merges, and a query for the new statement still")
        print("         finds it while the twin is live.")


if __name__ == "__main__":
    main()
