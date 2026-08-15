"""
Link verification — the precision half of two-stage linking.

Why this exists
---------------
``remember()`` has to answer two different questions with one similarity score:
which stored memory might this statement be about (recall), and is it really the
same fact (precision). Battery G measured that no single threshold can do both,
because the ranking is inverted — "proficient in Python" vs "proficient in
Japanese" scores 0.80 while a genuine career change scores 0.25. Lowering the
bar to catch the career change merges the two skills and destroys a true memory.

Splitting the questions fixes it. Similarity recalls candidates at a
deliberately low bar, then a verifier decides. Measured on 56 held-out pairs:

    27/56   today's shipped ladder, keyword similarity      13 false merges
    41/56   shipped ladder, embeddings                      12 false merges
    49/56   best score ANY single threshold can reach        —
    49/56   embeddings + qwen2.5-coder:14b verifier          0 false merges
    52/56   embeddings + gpt-4o-mini verifier                2 false merges
    55/56   embeddings + a perfect verifier (recall limit)   —

The false-merge column is the reason to bother. A duplicate is recoverable —
both facts stay retrievable and consolidation can reconcile them — while a false
merge supersedes and destroys a stored memory. Verification takes that from
roughly half of all distinct-fact pairs to zero.

Choosing a verifier
-------------------
Model choice matters far more than prompt wording and does not track model
quality. Measured on the same held-out pairs and the same prompt:

    qwen2.5:3b          31/56, 24 false merges   answers UPDATE to everything
    qwen2.5:14b         33/56,  0 false merges   answers KEEP_BOTH to everything
    qwen2.5-coder:14b   49/56,  0 false merges   the default below
    gpt-4o-mini         52/56,  2 false merges

Small models are unsafe here: they rubber-stamp, and a verifier that always
agrees is not a verifier. The code-tuned model beats the general instruct model
of identical size by 16 pairs, most likely because this is strict schema-filling
rather than open-ended reasoning. Verify any substitute before trusting it.

Cost and failure
----------------
One model call per candidate pair, roughly 5s locally on a 14B. Too slow to sit
inside an interactive write on a large store, which is why callers should keep
``link_recall_top_k`` small or run verification on a deferred pass.

Every failure path returns False — unreachable model, timeout, malformed JSON.
False means "keep both", which produces a duplicate: the recoverable error. A
broken verifier degrades to today's behaviour rather than to data loss.
"""

import json
import re
import urllib.error
import urllib.request
from typing import Optional, Protocol

VERIFY_SYSTEM = (
    "You maintain a long-term memory store about one user. Each memory holds a "
    "single fact. You decide whether a new statement refers to the same "
    "underlying fact as a stored memory."
)

# Validated in experiments/llm_verify_eval.py. The attribute-vs-value paragraph
# is load-bearing: without it gpt-4o-mini scores 29/56 instead of 52/56, because
# it reads "same attribute" as "same value" and refuses to update anything.
VERIFY_PROMPT = """Stored memory: "{stored}"
New statement: "{new}"

An ATTRIBUTE is the QUESTION a fact answers, never the answer itself.

  "I live in Berlin" and "I live in Paris" answer the same question — which city
  does the user live in — so they share one attribute and hold different values.
  The newer value supersedes the older one.

  "User is proficient in Python" and "User is proficient in Japanese" answer
  different questions. The user holds both at once, so these are two attributes.

Answer three things.

1. same_subject — do both statements describe the same person or thing? The
   user, the user's parents, the user's spouse and the user's colleagues are
   different subjects.
2. same_attribute — do both answer the SAME question about that subject, even
   when the answers differ, contradict, or cancel each other? Judge the
   question. Never judge the answer.
3. decision:
     UPDATE     — same_subject and same_attribute. The new statement corrects,
                  replaces or cancels the stored value; only one should remain.
     KEEP_BOTH  — anything else. Both are true at the same time.

Different, opposite, or cancelling values are the NORMAL case for UPDATE. A
statement that ends or negates the stored one ("no longer ...", "gave up on
...", "left ...") still answers the same question, provided the subject and the
question match.

Reply with JSON only, no prose:
{{"same_subject": true|false, "same_attribute": true|false, "decision": "UPDATE"|"KEEP_BOTH"}}"""


class LinkVerifier(Protocol):
    """Decides whether a recalled candidate is really the same fact."""

    def verify(self, new_text: str, stored_text: str, domain: str) -> bool:
        ...


def parse_verdict(raw: str) -> Optional[bool]:
    """True = UPDATE, False = KEEP_BOTH, None = could not be read."""
    if not raw:
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            decision = str(obj.get("decision", "")).strip().upper()
            if decision in ("UPDATE", "KEEP_BOTH"):
                return decision == "UPDATE"
    # Models that ignore the JSON instruction but still state a verdict.
    match = re.search(r"\b(UPDATE|KEEP_BOTH)\b", raw.upper())
    return match.group(1) == "UPDATE" if match else None


class LLMLinkVerifier:
    """Ollama-backed verifier. Same transport and fallback style as LLMExtractor.

    Results are memoised per (stored, new) pair for the life of the object, so
    re-verifying the same candidate inside one session is free.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:14b",
        ollama_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.url = ollama_url.rstrip("/") + "/api/generate"
        self.timeout = timeout
        self._cache: dict[tuple[str, str], bool] = {}
        self.calls = 0
        self.failures = 0

    def _generate(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "system": VERIFY_SYSTEM,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 120},
        }).encode()
        req = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode()).get("response", "")

    def verify(self, new_text: str, stored_text: str, domain: str) -> bool:
        key = (stored_text, new_text)
        if key in self._cache:
            return self._cache[key]
        verdict: Optional[bool] = None
        try:
            self.calls += 1
            verdict = parse_verdict(self._generate(
                VERIFY_PROMPT.format(stored=stored_text, new=new_text)))
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            verdict = None
        if verdict is None:
            self.failures += 1
            verdict = False        # keep both: a duplicate, not a lost memory
        self._cache[key] = verdict
        return verdict


class CallableVerifier:
    """Adapts a plain ``fn(new, stored, domain) -> bool`` to the protocol."""

    def __init__(self, fn) -> None:
        self._fn = fn

    def verify(self, new_text: str, stored_text: str, domain: str) -> bool:
        return bool(self._fn(new_text, stored_text, domain))


def as_verifier(obj) -> Optional[LinkVerifier]:
    """Accept a verifier, a bare callable, or None."""
    if obj is None or hasattr(obj, "verify"):
        return obj
    if callable(obj):
        return CallableVerifier(obj)
    raise TypeError(f"link_verifier must be a LinkVerifier or callable, got {type(obj)}")


def resolve_link_verifier(
    link_verifier,
    *,
    has_embedder: bool,
    ollama_url: str = "http://localhost:11434",
    llm_model: str = "qwen2.5-coder:14b",
) -> Optional[LinkVerifier]:
    """``"auto"`` attaches the local verifier when an embedder is present.

    Keyword-only layers stay on the threshold ladder. Embeddings without a
    verifier raise false-merge severity, which is why the two are coupled.
    Pass ``None`` or ``False`` to force the ladder on.
    """
    if link_verifier is False or link_verifier is None:
        return None
    if link_verifier == "auto":
        if not has_embedder:
            return None
        return LLMLinkVerifier(model=llm_model, ollama_url=ollama_url)
    return as_verifier(link_verifier)


def fit_score_threshold(pos_scores: list[float], neg_scores: list[float]):
    """Threshold for ``score >= t → UPDATE``, fitted on DEV only.

    Minimises total errors, then false merges, then prefers a higher (more
    conservative) cut. Returns ``(threshold, missed_links, false_merges)``.
    """
    if not pos_scores or not neg_scores:
        raise ValueError("need both must-link and must-not-link scores")
    cands = sorted(set(pos_scores) | set(neg_scores))
    # Also consider just below the lowest pos and just above the highest neg.
    cands = [cands[0] - 1.0] + cands + [cands[-1] + 1e-6]
    best = None
    for t in cands:
        missed = sum(1 for s in pos_scores if s < t)
        merged = sum(1 for s in neg_scores if s >= t)
        key = (missed + merged, merged, -t)
        if best is None or key < best[0]:
            best = (key, t, missed, merged)
    _, t, missed, merged = best
    return float(t), int(missed), int(merged)


class CrossEncoderVerifier:
    """Stage-2 pair scorer. Embeddings still recall; this decides same-fact.

    Default model is the MS MARCO MiniLM cross-encoder from the matching
    research note: cheap (~10ms), standard IR middle stage. Threshold must be
    fitted on the DEV split of ``linking_pairs`` — never on held-out.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model: str | None = None,
        threshold: float | None = None,
        encoder=None,
    ) -> None:
        self.model_name = model or self.DEFAULT_MODEL
        self.threshold = threshold
        if encoder is not None:
            self._ce = encoder
        else:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "CrossEncoderVerifier needs sentence-transformers "
                    "(pip install voltmem[embeddings])"
                ) from exc
            self._ce = CrossEncoder(self.model_name)

    def score(self, new_text: str, stored_text: str) -> float:
        raw = self._ce.predict([(stored_text, new_text)])
        return float(raw[0] if hasattr(raw, "__len__") else raw)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """``pairs`` are (stored, new)."""
        if not pairs:
            return []
        raw = self._ce.predict(pairs)
        return [float(x) for x in raw]

    def verify(self, new_text: str, stored_text: str, domain: str) -> bool:
        if self.threshold is None:
            raise ValueError(
                "CrossEncoderVerifier.threshold must be set after fitting on DEV")
        return self.score(new_text, stored_text) >= self.threshold
