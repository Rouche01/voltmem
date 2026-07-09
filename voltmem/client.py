"""
Product-facing API — Mem0-shaped surface over MemoryLayer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Union

from .embeddings import EmbeddingSimilarity
from .extract import HeuristicFactExtractor, LLMFactExtractor
from .memory import MemoryLayer, RetrieveResult, WriteResult

Message = dict[str, str]
AddInput = Union[str, Message, list[Message]]


def create_memory(
    db_path: str | Path = "voltmem.db",
    user_id: str = "default",
    *,
    embeddings: bool = True,
    verbose: bool = False,
    llm_extract: bool = False,
    llm_domain: bool = False,
    vector_index: str = "auto",
    **kwargs: Any,
) -> "Memory":
    """Create a production-ready memory instance with sensible defaults.

    Auto-detects an embedding backend when ``embeddings=True``
    (sentence-transformers → Ollama → offline hashing fallback).

    Parameters
    ----------
    vector_index : str
        ``auto`` (sqlite index when embedder present), ``sqlite``, ``brute``,
        or ``off`` for full-scan retrieval.
    llm_extract : bool
        Use Ollama to extract atomic facts from conversation message lists.
    llm_domain : bool
        Use Ollama for domain classification and contradiction detection
        inside ``remember()`` (passed to ``MemoryLayer`` as ``extractor``).
    """
    similarity_fn = kwargs.pop("similarity_fn", None)
    if embeddings and similarity_fn is None:
        similarity_fn = EmbeddingSimilarity(verbose=verbose)

    embed_fn = getattr(similarity_fn, "embed", None)

    fact_extractor = LLMFactExtractor() if llm_extract else HeuristicFactExtractor()

    layer_kwargs = dict(kwargs)
    if llm_domain:
        from .extract import LLMExtractor
        layer_kwargs["extractor"] = LLMExtractor()

    layer_kwargs.setdefault("vector_index", vector_index)
    layer_kwargs.setdefault("embed_fn", embed_fn)

    return Memory(
        user_id=user_id,
        db_path=db_path,
        similarity_fn=similarity_fn,
        fact_extractor=fact_extractor,
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
        **kwargs: Any,
    ):
        self.user_id = user_id
        self._fact_extractor = fact_extractor or HeuristicFactExtractor()
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
