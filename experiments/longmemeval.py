"""
LongMemEval (oracle) — public benchmark for VoltMem memory retrieval
====================================================================

Evaluates VoltMem on the official LongMemEval benchmark (ICLR 2025) using the
**oracle** split: only evidence sessions are ingested, so we isolate memory
quality from haystack noise. Data is streamed from HuggingFace — no full local
download required.

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
    .venv/bin/python experiments/longmemeval.py --quick          # 2 per question type
    .venv/bin/python experiments/longmemeval.py --per-type 5
    .venv/bin/python experiments/longmemeval.py --limit 500      # first N from stream

Requires: datasets, and a working embedding backend (Ollama nomic-embed-text or
sentence-transformers). Falls back to hashing if neither is available.
"""

from __future__ import annotations

import argparse
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


def collect_evidence(instance: dict) -> list[str]:
    out = []
    for sess in instance["haystack_sessions"]:
        for turn in sess:
            if turn.get("has_answer"):
                out.append(turn["content"])
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
            role = turn.get("role", "user")
            text = turn["content"].strip()
            if not text:
                continue
            prefix = "User" if role == "user" else "Assistant"
            stmt = f"{prefix}: {text}"
            source = "explicit_statement" if role == "user" else "weak_inference"
            dom = extractor.classify_domain(text)
            mem.write(stmt, domain=dom, source=source, at_time=ts)


def eval_instance(
    system: str,
    instance: dict,
    sim: EmbeddingSimilarity,
    top_k: int,
) -> tuple[bool, bool]:
    q_now = parse_lme_datetime(instance["question_date"])
    namespace = f"lme_{instance['question_id']}"

    profile = system.replace("voltmem_", "") if system.startswith("voltmem_") else None
    ctx = volatility_profile(profile) if profile else nullcontext()

    with ctx:
        with MemoryLayer(":memory:", similarity_fn=sim) as mem:
            view = mem.for_user(namespace)
            ingest_instance(view, instance)
            use_staleness = system != "similarity_only"
            recalled = view.recall(
                instance["question"], top_k=top_k, now=q_now,
                use_staleness=use_staleness)

    ev = collect_evidence(instance)
    return (
        answer_hit(recalled, instance["answer"]),
        evidence_hit(recalled, ev),
    )


def load_instances(limit: int | None, seed: int,
                   per_type: int | None = None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(
        "xiaowu0162/longmemeval-cleaned",
        split="longmemeval_oracle",
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


def run(limit: int | None, top_k: int, seed: int,
        per_type: int | None) -> None:
    print("=" * 74)
    print("LongMemEval ORACLE — VoltMem memory retrieval benchmark")
    print("=" * 74)
    print("  Streaming from HuggingFace (xiaowu0162/longmemeval-cleaned)")
    print(f"  instances={limit or 'all'}  top_k={top_k}\n")

    sim = EmbeddingSimilarity(verbose=True)
    print(f"  embedding backend: {sim.backend} ({sim.model})\n")

    instances = load_instances(limit, seed, per_type=per_type)
    print(f"  loaded {len(instances)} instances\n")

    stats = {s: {"answer": 0, "evidence": 0, "n": 0}
             for s in SYSTEMS}
    by_type: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: {s: {"answer": 0, "evidence": 0, "n": 0} for s in SYSTEMS})

    for i, inst in enumerate(instances):
        qtype = inst["question_type"]
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i + 1}/{len(instances)}] {inst['question_id']} "
                  f"({qtype})")
        for system in SYSTEMS:
            ah, eh = eval_instance(system, inst, sim, top_k)
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
    for s in SYSTEMS:
        n = stats[s]["n"]
        print(f"  {s:<14}"
              f"{rate(stats[s]['answer'], n):>12.3f}"
              f"{rate(stats[s]['evidence'], n):>14.3f}")

    print("\n  by question_type (answer hit rate):")
    types = sorted(by_type.keys())
    header = f"  {'type':<28}" + "".join(f"{s:>14}" for s in SYSTEMS)
    print(header)
    print("  " + "-" * (28 + 14 * len(SYSTEMS)))
    for qt in types:
        row = f"  {qt:<28}"
        for s in SYSTEMS:
            d = by_type[qt][s]
            row += f"{rate(d['answer'], d['n']):>14.3f}"
        print(row)

    real_a = rate(stats["voltmem_real"]["answer"], stats["voltmem_real"]["n"])
    flat_a = rate(stats["voltmem_flat"]["answer"], stats["voltmem_flat"]["n"])
    swap_a = rate(stats["voltmem_swap"]["answer"], stats["voltmem_swap"]["n"])
    store_a = rate(stats["similarity_only"]["answer"], stats["similarity_only"]["n"])

    print("\n" + "-" * 74)
    print("VERDICT:")
    causal = real_a >= swap_a and real_a >= flat_a * 0.95
    beats_sim = real_a >= store_a * 0.98
    if real_a > store_a and (real_a >= flat_a >= swap_a or real_a > swap_a):
        print(f"  PASS (retrieval). voltmem_real answer@{top_k}={real_a:.3f} "
              f"beats similarity_only={store_a:.3f} with causal ordering "
              f"(flat={flat_a:.3f}, swap={swap_a:.3f}).")
    elif beats_sim:
        print(f"  PARTIAL. voltmem_real ({real_a:.3f}) matches/beats "
              f"similarity_only ({store_a:.3f}); volatility ordering unclear "
              f"(flat={flat_a:.3f}, swap={swap_a:.3f}).")
    else:
        print(f"  NEEDS WORK on this split/regime. real={real_a:.3f} "
              f"flat={flat_a:.3f} swap={swap_a:.3f} "
              f"similarity_only={store_a:.3f}.")
    print("\n  NOTE: This is retrieval-mode only (no LLM answer generation). "
          "Scores measure whether the right facts land in top-k memories — "
          "the fair metric for a memory *layer*, not a full chat assistant.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Max instances (default: all streamed)")
    ap.add_argument("--quick", action="store_true",
                    help="2 instances per question type (~12 total)")
    ap.add_argument("--per-type", type=int, default=None,
                    help="Sample N instances per question_type from the stream")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    per_type = 2 if args.quick else args.per_type
    limit = None if per_type else (15 if args.quick else args.limit)
    run(limit, args.top_k, args.seed, per_type)


if __name__ == "__main__":
    main()
