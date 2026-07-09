"""
VoltMem vs Mem0 — real side-by-side on scripted scenarios.
==========================================================

Fair comparison rules:
  * Same user turns fed to both systems
  * Fresh Mem0 user_id per scenario (delete_all after)
  * Same search query and scoring as mem0_comparison.py
  * VoltMem uses create_memory(embeddings=True)

Mem0 setup (pick one)
---------------------
**A) OpenAI (fastest — Mem0 default)**
    pip install mem0ai
    export OPENAI_API_KEY=sk-...

**B) Fully local (Ollama + Qdrant)**
    pip install mem0ai qdrant-client
    docker run -p 6333:6333 qdrant/qdrant
    ollama pull llama3.1
    ollama pull nomic-embed-text
    export MEM0_BACKEND=ollama

Run:
    python experiments/mem0_side_by_side.py
"""

from __future__ import annotations

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voltmem import create_memory  # noqa: E402
from voltmem.embeddings import EmbeddingSimilarity  # noqa: E402

# Reuse one embedder per process so benchmarks are stable across scenarios.
_SHARED_SIM: EmbeddingSimilarity | None = None


def shared_similarity() -> EmbeddingSimilarity:
    global _SHARED_SIM
    if _SHARED_SIM is None:
        _SHARED_SIM = EmbeddingSimilarity(backend="sentence-transformers")
    return _SHARED_SIM


def create_benchmark_memory(user_id: str):
    return create_memory(
        ":memory:",
        user_id=user_id,
        embeddings=False,
        similarity_fn=shared_similarity(),
    )

# Reuse scenarios + scoring from the always-add comparison
from experiments.mem0_comparison import (  # noqa: E402
    SCENARIOS,
    score_answer,
    score_scenario,
)


def mem0_config() -> dict:
    backend = os.environ.get("MEM0_BACKEND", "openai").lower()
    if backend == "ollama":
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": os.environ.get("QDRANT_HOST", "localhost"),
                    "port": int(os.environ.get("QDRANT_PORT", "6333")),
                    "collection_name": "voltmem_mem0_compare",
                },
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": os.environ.get("MEM0_LLM", "llama3.1"),
                    "temperature": 0,
                    "ollama_base_url": os.environ.get(
                        "OLLAMA_BASE_URL", "http://localhost:11434"),
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": os.environ.get(
                        "MEM0_EMBED_MODEL", "nomic-embed-text"),
                    "ollama_base_url": os.environ.get(
                        "OLLAMA_BASE_URL", "http://localhost:11434"),
                },
            },
        }
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "Set OPENAI_API_KEY for Mem0 (default backend), or "
            "MEM0_BACKEND=ollama for local Ollama + Qdrant.")
    # Explicit config — Mem0() defaults (gpt-5-mini, temp=0.1) break on some models.
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": os.environ.get("MEM0_LLM", "gpt-4o-mini"),
                "temperature": 0,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": os.environ.get(
                    "MEM0_EMBED_MODEL", "text-embedding-3-small"),
            },
        },
    }


def make_mem0():
    from mem0 import Memory

    return Memory.from_config(mem0_config())


def mem0_add_turn(m, user_id: str, text: str) -> None:
    m.add([{"role": "user", "content": text}], user_id=user_id)


def mem0_search_top(m, user_id: str, query: str, limit: int = 1) -> str:
    out = m.search(query, filters={"user_id": user_id}, limit=limit)
    rows = out.get("results") if isinstance(out, dict) else out
    rows = rows or []
    if rows:
        row = rows[0]
        if isinstance(row, dict):
            return str(row.get("memory") or row.get("text") or "")
        return str(row)
    # Fallback if vector search returns nothing (index lag / empty query edge case)
    all_out = m.get_all(filters={"user_id": user_id})
    all_rows = all_out.get("results") if isinstance(all_out, dict) else all_out
    if not all_rows:
        return ""
    q = set(query.lower().split())
    scored = []
    for row in all_rows:
        text = str(row.get("memory", "")) if isinstance(row, dict) else str(row)
        words = set(text.lower().split())
        scored.append((len(q & words) / max(len(q), 1), text))
    scored.sort(reverse=True)
    return scored[0][1] if scored else ""


def mem0_count(m, user_id: str) -> int:
    out = m.get_all(filters={"user_id": user_id})
    if isinstance(out, dict):
        rows = out.get("results") or out.get("memories") or []
    else:
        rows = out or []
    return len(rows)


def run_scenario(scenario: dict, m) -> dict[str, str]:
    uid = f"voltmem-compare-{scenario['name']}-{uuid.uuid4().hex[:8]}"
    volt = create_benchmark_memory(uid)

    try:
        for turn in scenario["turns"]:
            mem0_add_turn(m, uid, turn)
            volt.add(turn)

        q = scenario["query"]
        mem_top = mem0_search_top(m, uid, q)
        volt_hits = volt.search(q, limit=1)
        volt_top = volt_hits[0]["memory"] if volt_hits else ""
        mc, vc = mem0_count(m, uid), len(volt.get_all())

        ms, vs = score_scenario(scenario, mem_top, volt_top, mc, vc)
        return {
            "mem0": ms,
            "voltmem": vs,
            "mem0_top": mem_top,
            "volt_top": volt_top,
            "mem0_count": str(mc),
            "volt_count": str(vc),
        }
    finally:
        try:
            m.delete_all(user_id=uid)
        except Exception:
            pass
        volt.close()


def main() -> None:
    try:
        m = make_mem0()
    except ImportError:
        raise SystemExit(
            "Mem0 not installed. Run: pip install mem0ai\n"
            "Then set OPENAI_API_KEY or MEM0_BACKEND=ollama (see script docstring)."
        ) from None

    backend = os.environ.get("MEM0_BACKEND", "openai")
    print("=" * 72)
    print(f"VoltMem vs Mem0 (real) — backend={backend}")
    print("=" * 72)
    print(f"  {'scenario':<22}{'mem0':>12}{'voltmem':>12}")
    print("  " + "-" * 46)

    volt_wins = mem0_wins = 0
    for sc in SCENARIOS:
        r = run_scenario(sc, m)
        print(f"  {sc['name']:<22}{r['mem0']:>12}{r['voltmem']:>12}")
        print(f"    mem0    ({r['mem0_count']} facts): {r['mem0_top']!r}")
        print(f"    voltmem ({r['volt_count']} facts): {r['volt_top']!r}")
        if r["voltmem"] == "WIN" and r["mem0"] != "WIN":
            volt_wins += 1
        if r["mem0"] == "WIN" and r["voltmem"] != "WIN":
            mem0_wins += 1

    print("\n" + "-" * 72)
    print(f"  VoltMem clearer wins: {volt_wins}/{len(SCENARIOS)} scenarios")
    print(f"  Mem0 clearer wins:    {mem0_wins}/{len(SCENARIOS)} scenarios")
    print("=" * 72)


if __name__ == "__main__":
    main()
