"""
Extraction layer for the batteries-included API (remember() / recall()).
=======================================================================

The core VoltMem engine needs three things the caller would otherwise supply by
hand: which DOMAIN a statement belongs to, and (given a related existing memory)
how strongly the new statement CONTRADICTS it (the mismatch magnitude). This
module turns raw text into those signals so users can just call
`mem.remember("I moved to Paris")`.

Two backends:

  * HeuristicExtractor (default, dependency-free)
      - domain via an ordered keyword map
      - mismatch via the embedding-similarity band already computed at match time
      Good enough to be useful; imperfect. Honest about being a heuristic.

  * LLMExtractor (optional, local Ollama)
      - asks a small local model to classify the domain and judge contradiction
      Higher quality; needs Ollama running with a chat model. Falls back to the
      heuristic on any error so remember() never hard-fails.

Both expose the same tiny interface:
    classify_domain(text) -> str                       (a DOMAIN_VOLATILITY key)
    mismatch(new_text, existing_text, similarity) -> float in [0, 1]
"""

from __future__ import annotations

import json
import re
import urllib.request

from .domains import DOMAIN_VOLATILITY

# ── keyword map (ordered: earlier = higher priority) ─────────────────────────
# Each entry: (domain, [trigger substrings]). First domain with a hit wins.
_KEYWORDS: list[tuple[str, list[str]]] = [
    ("emotional_context", ["feel", "feeling", "mood", "stressed", "anxious",
                            "happy", "sad", "excited", "tired", "angry",
                            "overwhelmed", "burned out", "burnt out"]),
    ("current_project", ["working on", "work on", "project", "building",
                          "shipping", "migrating", "developing", "feature",
                          "refactoring"]),
    ("current_task", ["today", "right now", "currently", "this morning",
                      "this afternoon", "to-do", "todo", "task", "reviewing",
                      "fixing"]),
    ("professional_context", ["my job", "work at", "company", "job title",
                              "my role", "promoted", "got hired", "employer",
                              "i work as", "position at"]),
    ("relationship", ["manager", "colleague", "coworker", "co-worker",
                      "teammate", "my friend", "partner", "spouse", "wife",
                      "husband", "collaborator", "my boss", "mentor"]),
    ("skill", ["learning", "know how to", "skilled", "good at", "expert in",
               "proficient", "i can code", "fluent in"]),
    ("biographical", ["born", "grew up", "hometown", "native", "originally from",
                      "nationality", "raised in"]),
    ("location", ["i live", "living in", "reside", "based in", "moved to",
                  "relocat", "i'm in", "i am in"]),
    ("long_term_goal", ["my goal", "aspire", "want to become", "dream of",
                        "long-term", "someday", "five years"]),
    ("core_preference", ["prefer", "favorite", "favourite", "i love", "i hate",
                         "i enjoy", "i like", "i dislike", "can't stand",
                         "always want", "never want"]),
    ("personality_trait", ["introvert", "extrovert", "introverted",
                           "extroverted", "i am patient", "organized",
                           "disciplined", "my personality", "i am a",
                           "i'm a", "i tend to"]),
    ("opinion", ["i think", "i believe", "in my opinion", "in my view",
                 "i feel that"]),
]

_DEFAULT_DOMAIN = "stated_preference"   # neutral mid-volatility fallback


class HeuristicExtractor:
    """Dependency-free domain + mismatch estimator."""

    def __init__(self, confirm_similarity: float = 0.82,
                 relate_similarity: float = 0.55):
        self.confirm_similarity = confirm_similarity
        self.relate_similarity = relate_similarity

    def classify_domain(self, text: str) -> str:
        t = text.lower()
        for domain, triggers in _KEYWORDS:
            if any(kw in t for kw in triggers):
                if domain in DOMAIN_VOLATILITY:
                    return domain
        return _DEFAULT_DOMAIN

    def mismatch(self, new_text: str, existing_text: str,
                 similarity: float) -> float:
        """Map "how similar is the new statement to the one it matched" into a
        contradiction estimate. Near-identical -> confirmation (low mismatch);
        same topic but diverging -> high mismatch."""
        if similarity >= self.confirm_similarity:
            return 0.05
        span = max(self.confirm_similarity - self.relate_similarity, 1e-6)
        frac = (self.confirm_similarity - similarity) / span     # 0..1
        return float(min(0.9, max(0.5, 0.5 + 0.4 * frac)))


class LLMExtractor:
    """Optional higher-quality extractor backed by a local Ollama chat model.

    Falls back to the heuristic on any error, so remember() never hard-fails.
    """

    def __init__(self, model: str = "qwen2.5-coder:14b",
                 ollama_url: str = "http://localhost:11434",
                 fallback: HeuristicExtractor | None = None):
        self.model = model
        self.url = ollama_url.rstrip("/") + "/api/generate"
        self.fallback = fallback or HeuristicExtractor()
        self._domains = list(DOMAIN_VOLATILITY.keys())

    def _generate(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.0},
        }).encode()
        req = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode()).get("response", "")

    def classify_domain(self, text: str) -> str:
        try:
            opts = ", ".join(self._domains)
            out = self._generate(
                f"Classify the user statement into exactly one memory domain.\n"
                f"Domains: {opts}\n"
                f"Statement: \"{text}\"\n"
                f"Answer with only the domain name.")
            out = out.strip().lower()
            for d in self._domains:
                if d in out:
                    return d
        except Exception:
            pass
        return self.fallback.classify_domain(text)

    def mismatch(self, new_text: str, existing_text: str,
                 similarity: float) -> float:
        try:
            out = self._generate(
                f"Existing memory: \"{existing_text}\"\n"
                f"New statement: \"{new_text}\"\n"
                f"Does the new statement CONTRADICT or CHANGE the existing "
                f"memory? Answer with a number from 0.0 (fully consistent / a "
                f"restatement) to 1.0 (directly contradicts). Answer only the "
                f"number.")
            m = re.search(r"[01](?:\.\d+)?", out)
            if m:
                return float(min(1.0, max(0.0, float(m.group()))))
        except Exception:
            pass
        return self.fallback.mismatch(new_text, existing_text, similarity)
