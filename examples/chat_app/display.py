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
