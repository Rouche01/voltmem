"""
Three-band write-path cost — is live 14B worth it in the grey zone?
===================================================================

Heuristic cards already decide known frames (city, birth year, skill, job) with
zero overwrites. The leftover writes are either:

  cheap   new statement has cards → join or insert. No model.
  grey    no cards, but embeddings recall a neighbour (bar 0.20) → 14B.
  insert  no cards, nothing nearby → insert. No model.

Cheap refusals never fall through to 14B (that re-imports false merges).

This script scores that routing on the held-out 56, using the Battery H
qwen2.5-coder:14b cache (no new verifier calls). Embeddings are used only to
place a pair in grey vs insert; pairwise scores are cached on disk.

Question: of the leftovers, how many extra correct updates does a live 14B
call buy, at what call count, and does it stay at zero overwrites?

If grey 14B buys almost nothing that sleeptime would not recover anyway, leave
the 14B off the live path.

Run:
    .venv/bin/python experiments/three_band_eval.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voltmem.structure import (                                # noqa: E402
    HeuristicStructuredExtractor,
    join_structured,
)
from llm_verify_eval import Cache, PROMPT_V2, parse, load_dotenv       # noqa: E402
from linking_pairs import SPLITS                                       # noqa: E402
from link_verify_prototype import BAR                                  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
SIM_CACHE_PATH = os.path.join(CACHE_DIR, "pair_sim_minilm.json")
VERIFY_MODEL = "qwen2.5-coder:14b"
WRITE_PASS_MERGES = 0


def _sim_key(stored, new):
    return f"{stored}\n---\n{new}"


def _load_sim_cache():
    if os.path.exists(SIM_CACHE_PATH):
        with open(SIM_CACHE_PATH) as fh:
            return json.load(fh)
    return {}


def _save_sim_cache(cache):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(SIM_CACHE_PATH, "w") as fh:
        json.dump(cache, fh, indent=1, sort_keys=True)


def _fill_sims(pairs, cache):
    missing = [
        (st, nt) for _sd, st, _nd, nt, _n in pairs
        if _sim_key(st, nt) not in cache
    ]
    if not missing:
        return cache, "disk"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from voltmem.embeddings import EmbeddingSimilarity
    try:
        sim_fn = EmbeddingSimilarity(backend="sentence-transformers")
        source = "sentence-transformers"
    except Exception:
        sim_fn = EmbeddingSimilarity()
        source = sim_fn.backend
        if source == "hashing":
            raise SystemExit(
                "Need a real embedder to place grey vs insert. "
                "Install sentence-transformers or pull nomic-embed-text."
            )
    for st, nt in missing:
        cache[_sim_key(st, nt)] = float(sim_fn(nt, st))
    _save_sim_cache(cache)
    return cache, source


class VerifierCache:
    """Battery H cache only. A miss is KEEP_BOTH (conservative)."""

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


def _band(hx, stored, new, sim):
    """Return (band, heuristic_update). Grey leaves the 14B decision to the caller."""
    new_facts = hx.extract(new)
    if new_facts:
        stored_facts = hx.extract(stored)
        joined = bool(
            stored_facts and join_structured(
                stored_facts, new_facts, new, stored)
        )
        return "cheap", joined
    if sim >= BAR:
        return "grey", False
    return "insert", False


def _score_rows(hx, pos, neg, sims, night, mode):
    """mode: heuristic | three_band | three_band_night | nearby_14b."""
    linked = merged = asks = night_asks = 0
    misses, kills = [], []
    bands = {name: [0, 0] for name in ("cheap", "grey", "insert")}
    grey_buys = []

    def decide(stored, new, note, is_pos):
        nonlocal linked, merged, asks, night_asks
        sim = float(sims[_sim_key(stored, new)])
        nearby = sim >= BAR
        band, cheap_hit = _band(hx, stored, new, sim)
        bands[band][0 if is_pos else 1] += 1

        update = False
        live_ask = False
        if mode == "heuristic":
            update = cheap_hit
        elif mode == "heuristic_night":
            update = cheap_hit
            if not update and nearby:
                night_asks += 1
                update = night.would_update(stored, new)
        elif mode == "nearby_14b":
            if nearby:
                live_ask = True
                update = night.would_update(stored, new)
        else:
            if band == "cheap":
                update = cheap_hit
            elif band == "grey":
                live_ask = True
                update = night.would_update(stored, new)
                if update:
                    grey_buys.append(note)
            if mode == "three_band_night" and not update and nearby:
                night_asks += 1
                if night.would_update(stored, new):
                    update = True

        if live_ask:
            asks += 1
        if is_pos:
            if update:
                linked += 1
            else:
                misses.append(f"[{band}] {note}")
        else:
            if update:
                merged += 1
                kills.append(f"[{band}] {note}")

    for _sd, st, _nd, nt, note in pos:
        decide(st, nt, note, True)
    for _sd, st, _nd, nt, note in neg:
        decide(st, nt, note, False)
    return {
        "linked": linked, "merged": merged, "asks": asks,
        "night_asks": night_asks,
        "misses": misses, "kills": kills, "bands": bands,
        "grey_buys": grey_buys,
        "total": linked + (len(neg) - merged),
    }


def _report(label, row, n_pos, n_neg, detail=True):
    print(f"  {label:<28} linked {row['linked']:>2}/{n_pos}   "
          f"false merges {row['merged']:>2}/{n_neg}   "
          f"total {row['total']}/{n_pos + n_neg}   "
          f"14B asks {row['asks']}")
    if detail and row["misses"]:
        print("    missed links:")
        for note in row["misses"]:
            print(f"      {note}")
    if detail and row["kills"]:
        print("    false merges:")
        for note in row["kills"]:
            print(f"      {note}")


def main():
    load_dotenv()
    print("=" * 84)
    print("THREE-BAND COST — heuristic / grey 14B / insert")
    print("=" * 84)

    held_pos, held_neg = SPLITS["held-out"]
    all_pairs = list(held_pos) + list(held_neg)
    sims, sim_src = _fill_sims(all_pairs, _load_sim_cache())
    nearby = sum(
        1 for _sd, st, _nd, nt, _n in all_pairs
        if float(sims[_sim_key(st, nt)]) >= BAR
    )
    print(f"  embeddings {sim_src}  recall bar {BAR:.2f}  "
          f"nearby {nearby}/{len(all_pairs)}")

    night = VerifierCache()
    for _sd, st, _nd, nt, _n in all_pairs:
        night.would_update(st, nt)
    print(f"  14B cache {night.hits}/{len(all_pairs)} hits, "
          f"{night.misses} misses")
    if night.misses:
        print("  FAIL — verifier cache incomplete. Re-run Battery H ollama.")
        return
    night.hits = night.misses = 0

    hx = HeuristicStructuredExtractor()
    n_pos, n_neg = len(held_pos), len(held_neg)

    rows = {}
    for mode, label, detail in (
        ("heuristic", "heuristic only", True),
        ("three_band", "three-band live", True),
        ("heuristic_night", "heuristic + sleeptime", False),
        ("three_band_night", "three-band + sleeptime", False),
        ("nearby_14b", "14B on every nearby pair", False),
    ):
        rows[mode] = _score_rows(hx, held_pos, held_neg, sims, night, mode)
        _report(label, rows[mode], n_pos, n_neg, detail)
        if mode in ("heuristic_night", "three_band_night"):
            print(f"    extra night 14B asks: {rows[mode]['night_asks']}")

    live = rows["three_band"]
    heur = rows["heuristic"]
    heur_night = rows["heuristic_night"]
    always = rows["nearby_14b"]
    print("\n  band occupancy (must-link / must-not):")
    for name in ("cheap", "grey", "insert"):
        pos_n, neg_n = live["bands"][name]
        print(f"    {name:<8} {pos_n:>2}/{n_pos} pos   {neg_n:>2}/{n_neg} neg")
    extra = live["linked"] - heur["linked"]
    saved = always["asks"] - live["asks"]
    print(f"\n  grey 14B extra links vs heuristic: {extra}")
    print(f"  14B calls avoided vs ask-every-nearby: {saved} "
          f"({always['asks']} → {live['asks']})")
    print(f"  sleeptime on heuristic misses: {heur_night['night_asks']} "
          f"night 14B asks, {heur_night['linked']}/{n_pos} links")
    if live["grey_buys"]:
        print("  grey 14B bought these must-links:")
        for note in live["grey_buys"]:
            print(f"      {note}")

    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    live_ok = live["merged"] <= WRITE_PASS_MERGES
    night_ok = heur_night["merged"] <= WRITE_PASS_MERGES
    print(f"  live three-band: {live['linked']}/{n_pos} links, "
          f"{live['merged']}/{n_neg} merges, {live['asks']} live 14B calls")
    print(f"  heuristic+night: {heur_night['linked']}/{n_pos} links, "
          f"{heur_night['merged']}/{n_neg} merges, "
          f"{heur_night['night_asks']} night 14B asks")
    if not live_ok:
        print("  FAIL — grey 14B overwrote a trap. Keep 14B off the live path.")
        return
    if not night_ok:
        print("  FAIL — sleeptime 14B overwrote a trap. Do not auto-reconcile.")
        return
    if extra <= 0:
        print("  DROP LIVE 14B — grey calls buy no extra updates.")
        print("         Heuristic decides known frames; sleeptime can finish.")
    elif heur_night["linked"] >= live["linked"]:
        print("  SAME ACCURACY EITHER WAY — 0 overwrites.")
        print(f"         Live grey 14B: user sees the {extra} extra updates now,")
        print(f"         for {live['asks']} calls while they wait.")
        print(f"         Sleeptime: same {heur_night['linked']}/{n_pos} by morning,")
        print("         live path stays millisecond. Latency choice, not safety.")
    else:
        print("  KEEP GREY 14B ON LIVE — night recovers fewer links")
        print(f"         than the {extra} extra live update(s).")


if __name__ == "__main__":
    main()
