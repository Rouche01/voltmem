"""CLI formatting helpers."""

from __future__ import annotations

from typing import Any


def format_memories(memories: list[dict[str, Any]], *, show_score: bool = False) -> str:
    if not memories:
        return "  (no memories yet)"
    lines: list[str] = []
    for m in memories:
        domain = m.get("domain") or "?"
        short_id = str(m.get("id", ""))[:8]
        score = m.get("score")
        score_part = f"  score={score:.3f}" if show_score and score is not None else ""
        lines.append(f"  [{short_id}] ({domain}){score_part}  {m.get('memory', '')}")
    return "\n".join(lines)


def format_writes(writes: list[dict[str, Any]]) -> str:
    if not writes:
        return ""
    parts = []
    for w in writes:
        action = w.get("action", "?")
        domain = w.get("domain", "?")
        parts.append(f"{action} [{domain}]")
    return "  memory: " + ", ".join(parts)


def format_discovery(report: dict[str, Any]) -> str:
    """Format domain auto-discovery stats from ``ChatSession.discovery_report()``."""
    if not report.get("auto_discover"):
        return "  auto_discover is off (restart with --auto-discover)"

    domains: dict[str, dict[str, Any]] = report.get("domain_discovery") or {}
    if not domains:
        return "  (no domain observations yet — chat to build stats)"

    lines = ["  Domain volatility (prior → empirical → resolved):"]
    for name in sorted(domains):
        d = domains[name]
        lines.append(
            f"  {name:<22}  prior={d['prior']:.3f}  "
            f"empirical={d['empirical']:.3f}  resolved={d['resolved']:.3f}  "
            f"confirms={d['n_confirms']}  mismatches={d['n_mismatches']}  "
            f"supersedes={d['n_supersedes']}"
        )
    return "\n".join(lines)
