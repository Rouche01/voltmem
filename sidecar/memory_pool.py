"""Shared MemoryLayer pool — one SQLite connection, many user namespaces."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Union

from voltmem import create_memory
from voltmem.classifiers import Classifier
from voltmem.client import Memory
from voltmem.extract import HeuristicFactExtractor

DbPath = Union[str, Path]


class MemoryPool:
    """Lazy per-user ``Memory`` views over a single shared layer/DB."""

    def __init__(
        self,
        db_path: DbPath,
        *,
        embeddings: bool = True,
        classifier: Classifier | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, Memory] = {}
        clf: Classifier | str = (
            classifier if classifier is not None else "heuristic"
        )
        self._root = create_memory(
            db_path,
            user_id="__sidecar__",
            embeddings=embeddings,
            classifier=clf,
        )
        # Same classifier the layer uses — keeps message-list extract aligned.
        layer_clf = self._root.layer._extractor
        self._fact_extractor = HeuristicFactExtractor(layer_clf)  # type: ignore[arg-type]

    def for_user(self, user_id: str) -> Memory:
        if not user_id or user_id == "__sidecar__":
            raise ValueError("user_id must be a non-empty tenant id")
        with self._lock:
            mem = self._cache.get(user_id)
            if mem is None:
                mem = Memory(
                    user_id=user_id,
                    layer=self._root.layer,
                    fact_extractor=self._fact_extractor,
                )
                self._cache[user_id] = mem
            return mem

    def close(self) -> None:
        with self._lock:
            self._cache.clear()
            self._root.close()
