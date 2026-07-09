"""
Product-facing API — Mem0-shaped surface over MemoryLayer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Union

from .embeddings import EmbeddingSimilarity
from .memory import MemoryLayer, RetrieveResult, WriteResult

Message = dict[str, str]
AddInput = Union[str, Message, list[Message]]


def create_memory(
    db_path: str | Path = "voltmem.db",
    user_id: str = "default",
    *,
    embeddings: bool = True,
    verbose: bool = False,
    **kwargs: Any,
) -> "Memory":
    """Create a production-ready memory instance with sensible defaults.

    Auto-detects an embedding backend when ``embeddings=True``
    (sentence-transformers → Ollama → offline hashing fallback).
    """
    similarity_fn = None
    if embeddings:
        similarity_fn = EmbeddingSimilarity(verbose=verbose)
    return Memory(
        user_id=user_id,
        db_path=db_path,
        similarity_fn=similarity_fn,
        **kwargs,
    )


class Memory:
    """Current-truth memory for LLM agents.

    Familiar ``add`` / ``search`` surface; VoltMem volatility engine underneath.
  Volatile facts update; stable facts resist corruption; stale volatile memories
    rank lower at retrieval.

    Examples
    --------
    >>> mem = create_memory("app.db", user_id="alice")
    >>> mem.add("I live in Berlin")
    >>> mem.add("Actually I moved to Paris")
    >>> mem.search("where does the user live?")
    """

    def __init__(
        self,
        user_id: str = "default",
        db_path: str | Path = ":memory:",
        *,
        layer: Optional[MemoryLayer] = None,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        **kwargs: Any,
    ):
        self.user_id = user_id
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
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        """Store a fact or conversation turn(s).

        Accepts a plain string, one message dict ``{"role": ..., "content": ...}``,
        or a list of message dicts (user turns are remembered; assistant optional).
        """
        if isinstance(data, str):
            return self._format_write(self._layer.remember(data, source=source))
        if isinstance(data, dict):
            return self._add_message(data, source=source)
        if isinstance(data, list):
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
        self._layer._store.delete(memory_id, namespace=self.user_id)
        return True

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
