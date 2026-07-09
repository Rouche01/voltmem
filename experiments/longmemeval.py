"""
LongMemEval — public benchmark for VoltMem memory retrieval
==========================================================

Evaluates VoltMem on the official LongMemEval benchmark (ICLR 2025). Data is
streamed from HuggingFace — no full local download required.

Splits
------
  * oracle  — evidence sessions only (~tiny haystack; all systems tend to tie)
  * s       — full haystack (~115k tokens, ~40 sessions; freshness should matter)
  * m       — very large haystack (~500 sessions; slow, use --limit / --per-type)

This is a **memory-layer** evaluation (not end-to-end LLM QA): after ingesting
chat history we ask whether `recall(question)` returns text that contains the
ground-truth answer or annotated evidence turns. That is the metric memory
systems can be fairly compared on without an API key.

Systems compared
----------------
  * voltmem_real  — chunk storage + volatility-aware retrieval (real priors)
  * voltmem_flat  — same chunks, equal volatility at retrieval (ablation)
  * voltmem_swap  — inverted volatility at retrieval (negative control)
  * similarity_only — same chunks, cosine similarity only (no freshness weighting)

Ingestion: every chat turn is stored as a chunk (standard LongMemEval / RAG
practice). VoltMem's contribution is tested at **retrieval ranking** — whether
volatility-weighted freshness surfaces the right evidence vs flat cosine search.
(Update policy is covered separately in llm_memory_bench.py.)

Metrics (at top-k)
------------------
  * answer_hit   — ground-truth answer (normalized) found in recalled text
  * evidence_hit — overlap with turns marked has_answer=True in the dataset

Run:
    .venv/bin/python experiments/longmemeval.py --quick          # oracle, 2 per type
    .venv/bin/python experiments/longmemeval.py --split s --quick   # S pilot
    .venv/bin/python experiments/longmemeval.py --split s --per-type 5
    .venv/bin/python experiments/longmemeval.py --split s --systems voltmem_real,similarity_only

Requires: datasets, and a working embedding backend (Ollama nomic-embed-text or
sentence-transformers). Falls back to hashing if neither is available.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import voltmem.domains as vdomains  # noqa: E402
from voltmem import MemoryLayer, EmbeddingSimilarity  # noqa: E402
from voltmem.extract import HeuristicExtractor  # noqa: E402

SYSTEMS = ("voltmem_real", "voltmem_flat", "voltmem_swap", "similarity_only")
SPLIT_MAP = {
    "oracle": "longmemeval_oracle",
    "s": "longmemeval_s_cleaned",
    "m": "longmemeval_m_cleaned",
}
DATE_RE = re.compile(r"(\d{4}/\d{2}/\d{2}).*?(\d{2}:\d{2})")


def parse_lme_datetime(s: str) -> float:
    """Parse LongMemEval timestamps like '2023/05/25 (Thu) 20:21'."""
    m = DATE_RE.search(s)
    if not m:
        return 0.0
    return datetime.strptime(f"{m.group(1)} {m.group(2)}",
                           "%Y/%m/%d %H:%M").timestamp()


def normalize(text: str) -> str:
    t = str(text).lower().strip().strip('"').strip("'")
    return re.sub(r"\s+", " ", t)


def answer_hit(recalled: list[str], answer: Any) -> bool:
    ans = normalize(answer)
    if not ans:
        return False
    blob = " ".join(normalize(r) for r in recalled)
    if ans in blob:
        return True
    # partial token overlap for paraphrased answers
    tokens = [w for w in re.split(r"[^\w]+", ans) if len(w) > 2]
    if not tokens:
        return False
    hits = sum(1 for w in tokens if w in blob)
    return hits / len(tokens) >= 0.6


def evidence_hit(recalled: list[str], evidence: list[str]) -> bool:
    if not evidence:
        return False
    for r in recalled:
        rn = normalize(r)
        for ev in evidence:
            en = normalize(ev)
            if len(en) < 20:
                if en in rn:
                    return True
            else:
                # substantial substring overlap either direction
                if en[:50] in rn or rn[:50] in en:
                    return True
                ev_tokens = set(en.split())
                r_tokens = set(rn.split())
                if ev_tokens and len(ev_tokens & r_tokens) / len(ev_tokens) >= 0.45:
                    return True
    return False


def parse_turn(turn: Any) -> dict:
    """Turns are dicts on oracle; JSON strings on S/M cleaned splits."""
    if isinstance(turn, dict):
        return turn
    if isinstance(turn, str):
        return json.loads(turn)
    raise TypeError(f"unexpected turn type: {type(turn)}")


def collect_evidence(instance: dict) -> list[str]:
    out = []
    for sess in instance["haystack_sessions"]:
        for turn in sess:
            t = parse_turn(turn)
            if t.get("has_answer"):
                out.append(t["content"])
    return out


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


def haystack_stats(instance: dict) -> tuple[int, int]:
    """Return (session_count, turn_count) for progress reporting."""
    sessions = instance.get("haystack_sessions") or []
    turns = sum(len(sess) for sess in sessions)
    return len(sessions), turns


def ingest_instance(mem: MemoryLayer, instance: dict) -> None:
    """Store each chat turn as a chunk with the session timestamp."""
    sessions = list(zip(
        instance["haystack_session_ids"],
        instance["haystack_dates"],
        instance["haystack_sessions"],
    ))
    sessions.sort(key=lambda x: parse_lme_datetime(x[1]))

    extractor = HeuristicExtractor()
    for _sid, date_str, sess in sessions:
        ts = parse_lme_datetime(date_str)
        for turn in sess:
            t = parse_turn(turn)
            role = t.get("role", "user")
            text = t["content"].strip()
            if not text:
                continue
            prefix = "User" if role == "user" else "Assistant"
            stmt = f"{prefix}: {text}"
            source = "explicit_statement" if role == "user" else "weak_inference"
            dom = extractor.classify_domain(text)
            mem.write(stmt, domain=dom, source=source, at_time=ts)


def eval_all_systems(
    instance: dict,
    sim: EmbeddingSimilarity,
    top_k: int,
    systems: tuple[str, ...],
) -> dict[str, tuple[bool, bool]]:
    """Ingest once, score each retrieval policy (important for large S haystacks)."""
    q_now = parse_lme_datetime(instance["question_date"])
    namespace = f"lme_{instance['question_id']}_{id(instance)}"
    evidence = collect_evidence(instance)

    out: dict[str, tuple[bool, bool]] = {}
    with MemoryLayer(":memory:", similarity_fn=sim) as mem:
        view = mem.for_user(namespace)
        ingest_instance(view, instance)
        for system in systems:
            profile = (
                system.replace("voltmem_", "")
                if system.startswith("voltmem_") else None
            )
            ctx = volatility_profile(profile) if profile else nullcontext()
            with ctx:
                use_staleness = system != "similarity_only"
                recalled = view.recall(
                    instance["question"], top_k=top_k, now=q_now,
                    use_staleness=use_staleness)
            out[system] = (
                answer_hit(recalled, instance["answer"]),
                evidence_hit(recalled, evidence),
            )
    return out


def load_instances(
    split: str,
    limit: int | None,
    seed: int,
    per_type: int | None = None,
) -> list[dict]:
    from datasets import load_dataset

    hf_split = SPLIT_MAP[split]
    ds = load_dataset(
        "xiaowu0162/longmemeval-cleaned",
        split=hf_split,
        streaming=True,
    )
    if per_type:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for ex in ds:
            row = dict(ex)
            qt = row["question_type"]
            if len(buckets[qt]) < per_type:
                buckets[qt].append(row)
            if all(len(v) >= per_type for v in buckets.values()) and \
               len(buckets) >= 6:
                break
        rows = [r for bucket in buckets.values() for r in bucket]
    else:
        rows = []
        for ex in ds:
            rows.append(dict(ex))
            if limit and len(rows) >= limit:
                break
    if seed and len(rows) > 1:
        import random
        random.Random(seed).shuffle(rows)
    if limit and not per_type:
        rows = rows[:limit]
    return rows


def run(
    split: str,
    limit: int | None,
    top_k: int,
    seed: int,
    per_type: int | None,
    systems: tuple[str, ...],
) -> None:
    label = {"oracle": "ORACLE", "s": "S (full haystack)", "m": "M (500 sessions)"}
    print("=" * 74)
    print(f"LongMemEval {label[split]} — VoltMem memory retrieval benchmark")
    print("=" * 74)
    print("  Streaming from HuggingFace (xiaowu0162/longmemeval-cleaned)")
    print(f"  split={split}  instances={limit or 'all'}  top_k={top_k}")
    print(f"  systems={', '.join(systems)}\n")

    sim = EmbeddingSimilarity(verbose=True)
    print(f"  embedding backend: {sim.backend} ({sim.model})\n")

    instances = load_instances(split, limit, seed, per_type=per_type)
    print(f"  loaded {len(instances)} instances\n")
    if split != "oracle" and instances:
        ns, nt = haystack_stats(instances[0])
        print(f"  first instance haystack: {ns} sessions, {nt} turns")
        if split == "s":
            print("  (S split ~115k tokens — expect minutes per instance on Ollama)\n")
        elif split == "m":
            print("  (M split is very large — use --limit or --per-type)\n")

    stats = {s: {"answer": 0, "evidence": 0, "n": 0} for s in systems}
    by_type: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: {s: {"answer": 0, "evidence": 0, "n": 0} for s in systems})

    for i, inst in enumerate(instances):
        qtype = inst["question_type"]
        ns, nt = haystack_stats(inst)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i + 1}/{len(instances)}] {inst['question_id']} "
                  f"({qtype}, {ns} sess / {nt} turns)")
        scores = eval_all_systems(inst, sim, top_k, systems)
        for system in systems:
            ah, eh = scores[system]
            stats[system]["n"] += 1
            stats[system]["answer"] += int(ah)
            stats[system]["evidence"] += int(eh)
            by_type[qtype][system]["n"] += 1
            by_type[qtype][system]["answer"] += int(ah)
            by_type[qtype][system]["evidence"] += int(eh)

    def rate(num, den):
        return num / den if den else 0.0

    print("\n" + "-" * 74)
    print(f"  {'system':<14}{'answer@'+str(top_k):>12}{'evidence@'+str(top_k):>14}")
    print("  " + "-" * 40)
    for s in systems:
        n = stats[s]["n"]
        print(f"  {s:<14}"
              f"{rate(stats[s]['answer'], n):>12.3f}"
              f"{rate(stats[s]['evidence'], n):>14.3f}")

    print("\n  by question_type (answer hit rate):")
    types = sorted(by_type.keys())
    header = f"  {'type':<28}" + "".join(f"{s:>14}" for s in systems)
    print(header)
    print("  " + "-" * (28 + 14 * len(systems)))
    for qt in types:
        row = f"  {qt:<28}"
        for s in systems:
            d = by_type[qt][s]
            row += f"{rate(d['answer'], d['n']):>14.3f}"
        print(row)

    print("\n" + "-" * 74)
    print("VERDICT:")
    if "voltmem_real" not in systems:
        print("  (skipped — voltmem_real not in --systems)")
    else:
        real_a = rate(stats["voltmem_real"]["answer"], stats["voltmem_real"]["n"])
        flat_a = (
            rate(stats["voltmem_flat"]["answer"], stats["voltmem_flat"]["n"])
            if "voltmem_flat" in systems else None
        )
        swap_a = (
            rate(stats["voltmem_swap"]["answer"], stats["voltmem_swap"]["n"])
            if "voltmem_swap" in systems else None
        )
        store_a = (
            rate(stats["similarity_only"]["answer"], stats["similarity_only"]["n"])
            if "similarity_only" in systems else None
        )
        beats_sim = store_a is not None and real_a >= store_a * 0.98
        has_causal = flat_a is not None and swap_a is not None
        if store_a is not None and real_a > store_a and (
            (has_causal and real_a >= flat_a >= swap_a) or
            (swap_a is not None and real_a > swap_a)
        ):
            causal = f"flat={flat_a:.3f}, swap={swap_a:.3f}" if has_causal else "partial controls"
            print(f"  PASS (retrieval). voltmem_real answer@{top_k}={real_a:.3f} "
                  f"beats similarity_only={store_a:.3f} with causal ordering "
                  f"({causal}).")
        elif beats_sim:
            causal = (
                f"flat={flat_a:.3f}, swap={swap_a:.3f}"
                if has_causal else "controls not run"
            )
            print(f"  PARTIAL. voltmem_real ({real_a:.3f}) matches/beats "
                  f"similarity_only ({store_a:.3f}); volatility ordering unclear "
                  f"({causal}).")
        else:
            parts = [f"real={real_a:.3f}"]
            if flat_a is not None:
                parts.append(f"flat={flat_a:.3f}")
            if swap_a is not None:
                parts.append(f"swap={swap_a:.3f}")
            if store_a is not None:
                parts.append(f"similarity_only={store_a:.3f}")
            print(f"  NEEDS WORK on this split/regime. " + " ".join(parts) + ".")
    print("\n  NOTE: This is retrieval-mode only (no LLM answer generation). "
          "Scores measure whether the right facts land in top-k memories — "
          "the fair metric for a memory *layer*, not a full chat assistant.")


def parse_systems(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return SYSTEMS
    names = tuple(s.strip() for s in raw.split(",") if s.strip())
    bad = [s for s in names if s not in SYSTEMS]
    if bad:
        raise SystemExit(f"Unknown --systems: {bad}. Choose from {SYSTEMS}")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split", choices=tuple(SPLIT_MAP), default="oracle",
        help="oracle (evidence only), s (~115k tokens), or m (~500 sessions)",
    )
    ap.add_argument("--limit", type=int, default=None,
                    help="Max instances (default: all streamed)")
    ap.add_argument("--quick", action="store_true",
                    help="2 instances per question type (~12 total)")
    ap.add_argument("--per-type", type=int, default=None,
                    help="Sample N instances per question_type from the stream")
    ap.add_argument(
        "--systems", default=None,
        help=f"Comma-separated subset of {','.join(SYSTEMS)}",
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    per_type = 2 if args.quick else args.per_type
    limit = None if per_type else (15 if args.quick else args.limit)
    run(
        args.split, limit, args.top_k, args.seed, per_type,
        parse_systems(args.systems),
    )


if __name__ == "__main__":
    main()
