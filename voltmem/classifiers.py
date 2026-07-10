"""
Pluggable classifiers for VoltMem's batteries-included ``remember()`` path.

A classifier infers which domain a statement belongs to and how strongly it
contradicts an existing memory. Pass one to ``create_memory(classifier=...)``.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, Union, runtime_checkable

from .domains import DOMAIN_VOLATILITY
from .extract import HeuristicExtractor, LLMExtractor

ClassifierName = str
ClassifierInput = Union[ClassifierName, "Classifier", dict[str, Any]]


@runtime_checkable
class Classifier(Protocol):
    """Domain + contradiction signals for ``MemoryLayer.remember()``."""

    def classify_domain(self, text: str) -> str:
        """Return a domain key (built-in or registered via ``DomainRegistry``)."""

    def mismatch(
        self, new_text: str, existing_text: str, similarity: float
    ) -> float:
        """Return contradiction magnitude in [0, 1]."""


# Public aliases — same implementations, clearer product naming.
HeuristicClassifier = HeuristicExtractor
LLMClassifier = LLMExtractor


class KeywordClassifier:
    """Ordered keyword → domain map. First match wins.

    Use alone or as the first stage in ``ChainedClassifier`` (call
    ``match_domain()`` for optional matching).
    """

    def __init__(
        self,
        keywords: dict[str, list[str]],
        *,
        default_domain: str = "stated_preference",
        fallback: Classifier | None = None,
    ) -> None:
        self._keywords = list(keywords.items())
        self._default_domain = default_domain
        self._fallback = fallback
        self._mismatch = (fallback or HeuristicExtractor()).mismatch

    def match_domain(self, text: str) -> str | None:
        """Return a domain when a keyword hits, else ``None``."""
        lowered = text.lower()
        for domain, triggers in self._keywords:
            if any(kw in lowered for kw in triggers):
                if domain in DOMAIN_VOLATILITY:
                    return domain
        return None

    def classify_domain(self, text: str) -> str:
        hit = self.match_domain(text)
        if hit is not None:
            return hit
        if self._fallback is not None:
            return self._fallback.classify_domain(text)
        return self._default_domain

    def mismatch(
        self, new_text: str, existing_text: str, similarity: float
    ) -> float:
        return self._mismatch(new_text, existing_text, similarity)


class ChainedClassifier:
    """Try classifiers in order; first non-``None`` ``KeywordClassifier`` match wins."""

    def __init__(
        self,
        classifiers: list[Classifier],
        *,
        default_domain: str = "stated_preference",
        fallback: Classifier | None = None,
    ) -> None:
        if not classifiers:
            raise ValueError("ChainedClassifier requires at least one classifier")
        self._chain = classifiers
        self._default_domain = default_domain
        self._fallback = fallback or HeuristicExtractor()
        self._mismatch = self._fallback.mismatch

    def classify_domain(self, text: str) -> str:
        for clf in self._chain:
            if isinstance(clf, KeywordClassifier):
                hit = clf.match_domain(text)
                if hit is not None:
                    return hit
            else:
                return clf.classify_domain(text)
        if self._fallback is not None:
            return self._fallback.classify_domain(text)
        return self._default_domain

    def mismatch(
        self, new_text: str, existing_text: str, similarity: float
    ) -> float:
        return self._mismatch(new_text, existing_text, similarity)


class CallableClassifier:
    """Wrap callables or a ``{"classify": ..., "mismatch": ...}`` dict."""

    def __init__(
        self,
        spec: dict[str, Callable[..., Any]],
        *,
        fallback: Classifier | None = None,
    ) -> None:
        fb = fallback or HeuristicExtractor()
        classify = spec.get("classify") or spec.get("classify_domain")
        mismatch = spec.get("mismatch")
        if classify is None:
            raise ValueError("classifier dict needs 'classify' or 'classify_domain'")
        self._classify = classify
        self._mismatch = mismatch or fb.mismatch

    def classify_domain(self, text: str) -> str:
        return str(self._classify(text))

    def mismatch(
        self, new_text: str, existing_text: str, similarity: float
    ) -> float:
        return float(self._mismatch(new_text, existing_text, similarity))


def resolve_classifier(
    classifier: ClassifierInput | None,
    *,
    llm_domain: bool = False,
    relate_threshold: float = 0.55,
    ollama_url: str = "http://localhost:11434",
    llm_model: str = "qwen2.5-coder:14b",
) -> Classifier:
    """Build a classifier from a string preset, instance, or callable dict."""
    if classifier is not None and not isinstance(classifier, str):
        if isinstance(classifier, dict):
            return CallableClassifier(classifier)
        return classifier  # type: ignore[return-value]

    name = classifier
    if name is None and llm_domain:
        name = "llm"
    if name is None or name == "heuristic":
        return HeuristicClassifier(relate_similarity=relate_threshold)
    if name in ("llm", "ollama"):
        return LLMClassifier(
            model=llm_model,
            ollama_url=ollama_url,
            fallback=HeuristicClassifier(relate_similarity=relate_threshold),
        )
    raise ValueError(
        f"unknown classifier {name!r}; use 'heuristic', 'llm', a Classifier instance, "
        "or a dict with 'classify' / 'mismatch' callables"
    )
