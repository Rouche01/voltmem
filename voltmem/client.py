"""
Product-facing API — Mem0-shaped surface over MemoryLayer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Union

from .classifiers import (
    Classifier,
    ClassifierInput,
    resolve_classifier,
)
from .domains import DomainRegistry
from .embeddings import EmbeddingSimilarity
from .extract import (
    FactExtractor,
    HeuristicFactExtractor,
    LLMFactExtractor,
)
from .memory import MemoryLayer, RetrieveResult, WriteResult

Message = dict[str, str]
AddInput = Union[str, Message, list[Message]]
FactExtractorInput = str | FactExtractor


def resolve_fact_extractor(
    fact_extractor: FactExtractorInput | None,
    *,
    llm_extract: bool = False,
    classifier: Classifier | None = None,
    ollama_url: str = "http://localhost:11434",
    llm_model: str = "qwen2.5-coder:14b",
) -> FactExtractor:
    if fact_extractor is not None and not isinstance(fact_extractor, str):
        return fact_extractor

    name = fact_extractor
    if name is None and llm_extract:
        name = "llm"
    if name is None or name == "heuristic":
        return HeuristicFactExtractor(classifier)  # type: ignore[arg-type]
    if name in ("llm", "ollama"):
        return LLMFactExtractor(model=llm_model, ollama_url=ollama_url)
    raise ValueError(
        f"unknown fact_extractor {name!r}; use 'heuristic', 'llm', or a FactExtractor instance"
    )


def create_memory(
    db_path: str | Path = "voltmem.db",
    user_id: str = "default",
    *,
    embeddings: bool = True,
    verbose: bool = False,
    classifier: ClassifierInput | None = "heuristic",
    fact_extractor: FactExtractorInput | None = "heuristic",
    domains: DomainRegistry | None = None,
    llm_extract: bool = False,
    llm_domain: bool = False,
    vector_index: str = "auto",
    auto_discover: bool = False,
    ollama_url: str = "http://localhost:11434",
    llm_model: str = "qwen2.5-coder:14b",
    **kwargs: Any,
) -> "Memory":
    """Create a production-ready memory instance with sensible defaults.

    Auto-detects an embedding backend when ``embeddings=True``
    (sentence-transformers → Ollama → offline hashing fallback).

    Parameters
    ----------
    classifier : str, Classifier, or dict
        ``"heuristic"`` (default), ``"llm"``, a :class:`~voltmem.classifiers.Classifier`
        instance, or a dict with ``classify`` / ``mismatch`` callables.
    fact_extractor : str or FactExtractor
        ``"heuristic"`` (default) or ``"llm"`` for message-list fact splitting.
    domains : DomainRegistry, optional
        Custom domain volatility priors (restored when the memory is closed).
    vector_index : str
        ``auto`` (sqlite index when embedder present), ``sqlite``, ``brute``,
        or ``off`` for full-scan retrieval.
    auto_discover : bool
        When True, learn per-domain volatility from confirm/mismatch/supersede
        patterns and blend with hand-tuned priors (prior-anchored EMA).
    llm_extract : bool
        Deprecated — use ``fact_extractor="llm"``.
    llm_domain : bool
        Deprecated — use ``classifier="llm"``.
    """
    similarity_fn = kwargs.pop("similarity_fn", None)
    if embeddings and similarity_fn is None:
        similarity_fn = EmbeddingSimilarity(verbose=verbose)

    embed_fn = getattr(similarity_fn, "embed", None)
    relate_threshold = float(kwargs.pop("relate_threshold", 0.55))

    # ``extractor=`` remains supported for backward compatibility.
    legacy_extractor = kwargs.pop("extractor", None)
    if legacy_extractor is not None and classifier == "heuristic":
        resolved_classifier = legacy_extractor
    else:
        resolved_classifier = resolve_classifier(
            classifier,
            llm_domain=llm_domain,
            relate_threshold=relate_threshold,
            ollama_url=ollama_url,
            llm_model=llm_model,
        )

    resolved_fact_extractor = resolve_fact_extractor(
        fact_extractor,
        llm_extract=llm_extract,
        classifier=resolved_classifier,
        ollama_url=ollama_url,
        llm_model=llm_model,
    )

    domain_restore: Callable[[], None] | None = None
    if domains is not None:
        domain_restore = domains.install()

    layer_kwargs = dict(kwargs)
    layer_kwargs["extractor"] = resolved_classifier
    layer_kwargs.setdefault("vector_index", vector_index)
    layer_kwargs.setdefault("embed_fn", embed_fn)
    layer_kwargs.setdefault("relate_threshold", relate_threshold)
    layer_kwargs.setdefault("auto_discover", auto_discover)

    return Memory(
        user_id=user_id,
        db_path=db_path,
        similarity_fn=similarity_fn,
        fact_extractor=resolved_fact_extractor,
        domain_restore=domain_restore,
        **layer_kwargs,
    )


class Memory:
    """Current-truth memory for LLM agents.

    Familiar ``add`` / ``search`` surface; VoltMem volatility engine underneath.
    Volatile facts update; stable facts resist corruption; stale volatile memories
    rank lower at retrieval.
    """

    def __init__(
        self,
        user_id: str = "default",
        db_path: str | Path = ":memory:",
        *,
        layer: Optional[MemoryLayer] = None,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        fact_extractor: Optional[HeuristicFactExtractor] = None,
        domain_restore: Callable[[], None] | None = None,
        **kwargs: Any,
    ):
        self.user_id = user_id
        self._fact_extractor = fact_extractor or HeuristicFactExtractor()
        self._domain_restore = domain_restore
        if layer is not None:
            self._layer = (
                layer if layer.namespace == user_id else layer.for_user(user_id)
            )
            self._owns_layer = False
        else:
            self._layer = MemoryLayer(
                db_path,
                similarity_fn=similarity_fn,
                namespace=user_id,
                **kwargs,
            )
            self._owns_layer = True

    @property
    def layer(self) -> MemoryLayer:
        """Low-level ``MemoryLayer`` for advanced control."""
        return self._layer

    def add(
        self,
        data: AddInput,
        *,
        source: str = "explicit_statement",
        extract: bool | None = None,
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        """Store a fact or conversation turn(s).

        Accepts a plain string, one message dict ``{"role": ..., "content": ...}``,
        or a list of message dicts. For message lists, ``extract=True`` (default)
        pulls atomic user facts before storing.
        """
        if isinstance(data, str):
            return self._format_write(self._layer.remember(data, source=source))
        if isinstance(data, dict):
            return self._add_message(data, source=source)
        if isinstance(data, list):
            if not data:
                return []
            do_extract = extract if extract is not None else True
            if do_extract:
                return self._add_extracted_facts(data, source=source)
            out = []
            for msg in data:
                if not isinstance(msg, dict):
                    raise TypeError("each message must be a dict with role/content")
                result = self._add_message(msg, source=source)
                if result is not None:
                    out.append(result)
            return out
        raise TypeError("add() expects str, message dict, or list of message dicts")

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Retrieve current-truth memories ranked by relevance and freshness."""
        result = self._layer.retrieve(
            query, top_k=limit, min_score=min_score)
        return self._format_retrieve(result)

    def get_all(self) -> list[dict[str, Any]]:
        """Return all active memories for this user."""
        return [
            self._format_item(item)
            for item in self._layer._active()
        ]

    def get(self, memory_id: str) -> Optional[dict[str, Any]]:
        """Return one memory by id, or None."""
        info = self._layer.inspect(memory_id)
        if info.get("error"):
            return None
        return info

    def delete(self, memory_id: str) -> bool:
        """Remove a memory by id. Returns True if deleted."""
        item = self._layer._store.get(memory_id)
        if not item or item.namespace != self.user_id:
            return False
        return self._layer.remove(memory_id)

    def clear(self) -> None:
        """Remove all memories for this user."""
        self._layer.clear()

    def summary(self) -> dict[str, Any]:
        return self._layer.summary()

    def close(self) -> None:
        if self._owns_layer:
            self._layer.close()
        if self._domain_restore is not None:
            self._domain_restore()
            self._domain_restore = None

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _add_extracted_facts(
        self, messages: list[Message], *, source: str
    ) -> list[dict[str, Any]]:
        facts = self._fact_extractor.extract(messages)
        out: list[dict[str, Any]] = []
        for fact in facts:
            src = fact.source or source
            if fact.domain:
                result = self._layer.remember(
                    fact.content, domain=fact.domain, source=src)
            else:
                result = self._layer.remember(fact.content, source=src)
            out.append(self._format_write(result))
        return out

    def _add_message(
        self, msg: Message, *, source: str
    ) -> Optional[dict[str, Any]]:
        role = msg.get("role", "user")
        content = str(msg.get("content", "")).strip()
        if not content:
            return None
        src = source if role == "user" else "assistant_response"
        return self._format_write(self._layer.remember(content, source=src))

    @staticmethod
    def _format_write(result: WriteResult) -> dict[str, Any]:
        item = result.item
        return {
            "id": item.id,
            "memory": item.content,
            "action": result.action,
            "domain": item.domain,
            "detail": result.detail,
        }

    @staticmethod
    def _format_item(item: Any) -> dict[str, Any]:
        return {
            "id": item.id,
            "memory": item.content,
            "domain": item.domain,
            "source": item.source,
            "created_at": item.created_at,
            "last_confirmed_at": item.last_confirmed_at,
        }

    def _format_retrieve(self, result: RetrieveResult) -> list[dict[str, Any]]:
        out = []
        for item, score in zip(result.items, result.scores):
            row = self._format_item(item)
            row["score"] = round(score, 4)
            out.append(row)
        return out
