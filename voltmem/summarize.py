"""
Memory content summarizers for maintenance consolidate.
=======================================================

``consolidate`` turns accumulated mismatch evidence into a new tip memory.
Two backends share one protocol:

  * HeuristicSummarizer (default, dependency-free)
      - Uses the most recent non-empty evidence text as the updated tip.
      - Deterministic; safe for tests and scheduled runs without a model.

  * LLMSummarizer (optional, local Ollama)
      - Asks a local model to rewrite current + evidence into one sentence.
      - Falls back to the heuristic on any error.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Protocol


class MemorySummarizer(Protocol):
    """Produce updated memory content from a tip + mismatch evidence texts."""

    def summarize(
        self, current: str, evidence: list[str], *, domain: str
    ) -> str:
        ...


def _clean_evidence(evidence: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in evidence:
        text = (raw or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


class HeuristicSummarizer:
    """Deterministic consolidate rewrite: latest distinct evidence wins.

    Returns ``current`` unchanged when evidence is empty or only restates the
    tip (case-insensitive). Never prefixes ``[consolidated]``.
    """

    def summarize(
        self, current: str, evidence: list[str], *, domain: str
    ) -> str:
        tip = (current or "").strip()
        cleaned = _clean_evidence(evidence)
        if not cleaned:
            return tip
        latest = cleaned[-1]
        if tip and latest.casefold() == tip.casefold():
            return tip
        return latest


class LLMSummarizer:
    """Optional Ollama-backed consolidator; falls back to HeuristicSummarizer."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:14b",
        ollama_url: str = "http://localhost:11434",
        fallback: HeuristicSummarizer | None = None,
        *,
        timeout: float = 30.0,
    ):
        self.model = model
        self.url = ollama_url.rstrip("/") + "/api/generate"
        self.fallback = fallback or HeuristicSummarizer()
        self.timeout = timeout

    def _generate(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }).encode()
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode()).get("response", "")

    def summarize(
        self, current: str, evidence: list[str], *, domain: str
    ) -> str:
        cleaned = _clean_evidence(evidence)
        if not cleaned:
            return (current or "").strip()
        bullets = "\n".join(f"- {e}" for e in cleaned)
        prompt = (
            "You update a single long-term memory fact for an AI agent.\n"
            f"Domain: {domain}\n"
            f"Current memory: \"{(current or '').strip()}\"\n"
            "Recent conflicting observations (oldest first):\n"
            f"{bullets}\n"
            "Write ONE updated memory sentence that reflects the emergent "
            "truth after these observations. Do not invent facts that are not "
            "supported by the current memory or the observations. "
            "Answer with only the memory sentence."
        )
        try:
            out = self._generate(prompt).strip()
            # Drop common quote wrapping from chatty models.
            if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'":
                out = out[1:-1].strip()
            if out:
                return out
        except Exception:
            pass
        return self.fallback.summarize(current, cleaned, domain=domain)
