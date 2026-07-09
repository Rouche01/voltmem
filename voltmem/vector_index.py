"""
Vector index — fast embedding retrieval for MemoryLayer.
========================================================

SQLite remains the source of truth for memory metadata. The vector index
accelerates candidate retrieval; MemoryLayer still applies volatility-aware
re-ranking on top of ANN results.

Backends:
  * BruteForceVectorIndex — in-memory, zero deps (tests / small stores)
  * SqliteVectorIndex     — vectors in a ``memory_vectors`` table (default)
"""

from __future__ import annotations

import math
import sqlite3
import struct
from pathlib import Path
from typing import Optional, Protocol


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def unpack_vector(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class VectorIndex(Protocol):
    """ANN-style index keyed by memory item id."""

    def upsert(
        self,
        item_id: str,
        namespace: str,
        domain: str,
        vector: list[float],
    ) -> None: ...

    def delete(self, item_id: str, namespace: str) -> None: ...

    def delete_namespace(self, namespace: str) -> None: ...

    def search(
        self,
        query_vector: list[float],
        namespace: str,
        top_k: int,
        *,
        domain: str | None = None,
    ) -> list[tuple[str, float]]: ...

    def close(self) -> None: ...


class BruteForceVectorIndex:
    """In-memory brute-force cosine search."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], tuple[str, list[float]]] = {}

    def upsert(
        self,
        item_id: str,
        namespace: str,
        domain: str,
        vector: list[float],
    ) -> None:
        self._rows[(namespace, item_id)] = (domain, list(vector))

    def delete(self, item_id: str, namespace: str) -> None:
        self._rows.pop((namespace, item_id), None)

    def delete_namespace(self, namespace: str) -> None:
        keys = [k for k in self._rows if k[0] == namespace]
        for key in keys:
            del self._rows[key]

    def search(
        self,
        query_vector: list[float],
        namespace: str,
        top_k: int,
        *,
        domain: str | None = None,
    ) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        for (ns, item_id), (dom, vec) in self._rows.items():
            if ns != namespace:
                continue
            if domain is not None and dom != domain:
                continue
            sim = cosine_similarity(query_vector, vec)
            scored.append((item_id, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def close(self) -> None:
        return None


_CREATE_VECTORS = """
CREATE TABLE IF NOT EXISTS memory_vectors (
    item_id    TEXT PRIMARY KEY,
    namespace  TEXT NOT NULL,
    domain     TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_vectors_ns
    ON memory_vectors(namespace);
CREATE INDEX IF NOT EXISTS idx_memory_vectors_ns_domain
    ON memory_vectors(namespace, domain);
"""


class SqliteVectorIndex:
    """SQLite-backed vector table sharing the memory database file."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.executescript(_CREATE_VECTORS)
        self._conn.commit()

    def upsert(
        self,
        item_id: str,
        namespace: str,
        domain: str,
        vector: list[float],
    ) -> None:
        blob = pack_vector(vector)
        self._conn.execute(
            """
            INSERT INTO memory_vectors (item_id, namespace, domain, dim, vector)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                namespace=excluded.namespace,
                domain=excluded.domain,
                dim=excluded.dim,
                vector=excluded.vector
            """,
            (item_id, namespace, domain, len(vector), blob),
        )
        self._conn.commit()

    def delete(self, item_id: str, namespace: str) -> None:
        self._conn.execute(
            "DELETE FROM memory_vectors WHERE item_id=? AND namespace=?",
            (item_id, namespace),
        )
        self._conn.commit()

    def delete_namespace(self, namespace: str) -> None:
        self._conn.execute(
            "DELETE FROM memory_vectors WHERE namespace=?", (namespace,))
        self._conn.commit()

    def search(
        self,
        query_vector: list[float],
        namespace: str,
        top_k: int,
        *,
        domain: str | None = None,
    ) -> list[tuple[str, float]]:
        if domain is None:
            rows = self._conn.execute(
                "SELECT item_id, vector FROM memory_vectors WHERE namespace=?",
                (namespace,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT item_id, vector FROM memory_vectors "
                "WHERE namespace=? AND domain=?",
                (namespace, domain),
            ).fetchall()
        scored = [
            (item_id, cosine_similarity(query_vector, unpack_vector(blob)))
            for item_id, blob in rows
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def close(self) -> None:
        self._conn.close()


def create_vector_index(
    mode: str,
    db_path: str | Path = ":memory:",
    *,
    has_embedder: bool = False,
) -> Optional[VectorIndex]:
    """Factory: ``off`` | ``brute`` | ``sqlite`` | ``auto``."""
    mode = (mode or "auto").lower()
    if mode == "off":
        return None
    if mode == "brute":
        return BruteForceVectorIndex()
    if mode == "sqlite":
        return SqliteVectorIndex(db_path)
    if mode == "auto":
        return SqliteVectorIndex(db_path) if has_embedder else None
    raise ValueError(
        f"Unknown vector_index mode {mode!r}; use off, brute, sqlite, or auto")
