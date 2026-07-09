"""
Pluggable embedding-based similarity for VoltMem retrieval.
==========================================================

VoltMem's contribution is the volatility/freshness weighting applied *on top of*
a semantic-similarity signal. The core library ships with a dependency-free
keyword scorer; this module provides a drop-in embedding scorer for real semantic
retrieval:

    from voltmem import MemoryLayer
    from voltmem.embeddings import EmbeddingSimilarity

    sim = EmbeddingSimilarity()            # auto-detects the best available backend
    mem = MemoryLayer(":memory:", similarity_fn=sim)

Backend auto-detection (first that works wins), overridable via backend=...:

  1. "sentence-transformers"  — local, high quality (all-MiniLM-L6-v2 by default)
  2. "ollama"                 — local daemon at http://localhost:11434
                               (nomic-embed-text by default)
  3. "hashing"               — deterministic, dependency-free fallback so the
                               library and benchmarks ALWAYS run offline. Lower
                               quality (bag-of-hashed-tokens), but reproducible.

The returned callable maps a (query, content) pair to a similarity in [0, 1]
(cosine similarity rescaled from [-1, 1]). Embeddings are cached per text.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Callable, Optional


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class EmbeddingSimilarity:
    """Callable (query, content) -> similarity in [0, 1] backed by embeddings."""

    def __init__(
        self,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        dim: int = 256,
        verbose: bool = False,
    ):
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.dim = dim
        self.verbose = verbose
        self._cache: dict[str, list[float]] = {}
        self._embed_fn: Callable[[str], list[float]]

        # Try backends in order; a backend counts as usable only if it can
        # actually produce an embedding (a running Ollama daemon with no embed
        # model pulled, for example, must NOT win — it 404s on first use).
        candidates = [backend] if backend else [
            "sentence-transformers", "ollama", "hashing"]
        errors = []
        user_model = model
        for cand in candidates:
            try:
                self.model = user_model   # reset per attempt; don't leak labels
                fn = self._make_backend(cand)
                _ = fn("voltmem backend probe")   # trial embed; raises if broken
                self.backend = cand
                self._embed_fn = fn
                if self.verbose:
                    print(f"[EmbeddingSimilarity] using backend={cand} "
                          f"model={self.model}")
                return
            except Exception as e:                 # noqa: BLE001
                errors.append(f"{cand}: {type(e).__name__}: {e}")
                if self.verbose:
                    print(f"[EmbeddingSimilarity] backend {cand} unavailable "
                          f"({type(e).__name__})")
        # hashing never fails, so we only get here if an explicit backend was bad
        raise RuntimeError("No usable embedding backend. Tried:\n  " +
                           "\n  ".join(errors))

    def _make_backend(self, backend: str) -> Callable[[str], list[float]]:
        if backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            model_name = self.model or "all-MiniLM-L6-v2"
            st = SentenceTransformer(model_name)
            self.model = model_name

            def embed(text: str) -> list[float]:
                return [float(x) for x in st.encode(text, normalize_embeddings=False)]

            return embed

        if backend == "ollama":
            model_name = self.model or "nomic-embed-text"
            self.model = model_name
            url = f"{self.ollama_url}/api/embeddings"

            def embed(text: str) -> list[float]:
                payload = json.dumps({"model": model_name, "prompt": text}).encode()
                req = urllib.request.Request(
                    url, data=payload, headers={"Content-Type": "application/json"})
                last_err: Exception | None = None
                for attempt in range(5):
                    try:
                        with urllib.request.urlopen(req, timeout=60) as resp:
                            data = json.loads(resp.read().decode())
                        return [float(x) for x in data["embedding"]]
                    except urllib.error.HTTPError as e:
                        last_err = e
                        if e.code in (429, 500, 502, 503) and attempt < 4:
                            time.sleep(2 ** attempt)
                            continue
                        raise
                raise last_err  # type: ignore[misc]

            return embed

        # ── hashing fallback ──────────────────────────────────────────────────
        self.model = self.model or f"hashing-{self.dim}"

        def embed(text: str) -> list[float]:
            vec = [0.0] * self.dim
            tokens = text.lower().split()
            for tok in tokens:
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 8) & 1 else -1.0
                vec[idx] += sign
            return vec

        return embed

    # ── public API ────────────────────────────────────────────────────────────
    def embed(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = self._embed_fn(text)
        return self._cache[text]

    def __call__(self, query: str, content: str) -> float:
        if not query or not content:
            return 0.0
        cos = _cosine(self.embed(query), self.embed(content))
        # Return clamped raw cosine. We deliberately do NOT rescale [-1,1]->[0,1]:
        # that compresses the dynamic range and collapses the gap between
        # "related" and "unrelated" (which breaks similarity thresholds used by
        # remember()). Clamping negatives to 0 is fine for retrieval and matching.
        return max(0.0, min(1.0, cos))
