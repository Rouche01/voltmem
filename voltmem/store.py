"""
SQLite-backed persistent store for MemoryItems.
Handles all serialisation/deserialisation; the rest of the library
works purely with MemoryItem objects.
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional
from .domains import MemoryItem
from .discovery import DomainStats


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id                TEXT PRIMARY KEY,
    content           TEXT NOT NULL,
    domain            TEXT NOT NULL,
    source            TEXT NOT NULL,
    repetition_count  INTEGER DEFAULT 1,
    volatility_ema    REAL    DEFAULT -1.0,
    mismatch_count    INTEGER DEFAULT 0,
    goal_delta        REAL    DEFAULT 0.0,
    created_at        REAL    NOT NULL,
    last_confirmed_at REAL    NOT NULL,
    last_audited_at   REAL    DEFAULT 0.0,
    tags              TEXT    DEFAULT '[]',
    superseded_by     TEXT    DEFAULT NULL,
    namespace         TEXT    NOT NULL DEFAULT 'default'
);
CREATE INDEX IF NOT EXISTS idx_domain ON memories(namespace, domain);
CREATE INDEX IF NOT EXISTS idx_active ON memories(namespace, superseded_by);
CREATE TABLE IF NOT EXISTS domain_stats (
    namespace     TEXT NOT NULL,
    domain        TEXT NOT NULL,
    n_confirms    INTEGER DEFAULT 0,
    n_mismatches  INTEGER DEFAULT 0,
    n_supersedes  INTEGER DEFAULT 0,
    mismatch_sum  REAL    DEFAULT 0.0,
    PRIMARY KEY (namespace, domain)
);
"""


def _row_to_item(row: sqlite3.Row) -> MemoryItem:
    keys = row.keys()
    return MemoryItem(
        id=row["id"],
        content=row["content"],
        domain=row["domain"],
        source=row["source"],
        namespace=row["namespace"] if "namespace" in keys else "default",
        repetition_count=row["repetition_count"],
        volatility_ema=row["volatility_ema"],
        mismatch_count=row["mismatch_count"],
        goal_delta=row["goal_delta"],
        created_at=row["created_at"],
        last_confirmed_at=row["last_confirmed_at"],
        last_audited_at=row["last_audited_at"],
        tags=json.loads(row["tags"]),
        superseded_by=row["superseded_by"],
    )


class MemoryStore:
    """
    Low-level SQLite store. Use MemoryLayer (memory.py) for the
    full volatility-aware read/write policy.
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_TABLE)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Additive migrations for databases created by older versions."""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(memories)")}
        if "namespace" not in cols:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN namespace "
                "TEXT NOT NULL DEFAULT 'default'")

    # ── write ─────────────────────────────────────────────────────────────────

    def insert(self, item: MemoryItem) -> MemoryItem:
        if not item.id:
            item.id = str(uuid.uuid4())
        now = time.time()
        if not item.created_at:
            item.created_at = now
        if not item.last_confirmed_at:
            item.last_confirmed_at = now
        self._conn.execute("""
            INSERT INTO memories (
                id, content, domain, source,
                repetition_count, volatility_ema, mismatch_count, goal_delta,
                created_at, last_confirmed_at, last_audited_at,
                tags, superseded_by, namespace
            ) VALUES (
                :id, :content, :domain, :source,
                :repetition_count, :volatility_ema, :mismatch_count, :goal_delta,
                :created_at, :last_confirmed_at, :last_audited_at,
                :tags, :superseded_by, :namespace
            )""", {
            "id": item.id,
            "content": item.content,
            "domain": item.domain,
            "source": item.source,
            "repetition_count": item.repetition_count,
            "volatility_ema": item.volatility_ema,
            "mismatch_count": item.mismatch_count,
            "goal_delta": item.goal_delta,
            "created_at": item.created_at,
            "last_confirmed_at": item.last_confirmed_at,
            "last_audited_at": item.last_audited_at,
            "tags": json.dumps(item.tags),
            "superseded_by": item.superseded_by,
            "namespace": item.namespace,
        })
        self._conn.commit()
        return item

    def update(self, item: MemoryItem) -> None:
        self._conn.execute("""
            UPDATE memories SET
                content=:content, domain=:domain, source=:source,
                repetition_count=:repetition_count,
                volatility_ema=:volatility_ema,
                mismatch_count=:mismatch_count,
                goal_delta=:goal_delta,
                last_confirmed_at=:last_confirmed_at,
                last_audited_at=:last_audited_at,
                tags=:tags,
                superseded_by=:superseded_by
            WHERE id=:id
        """, {
            "id": item.id,
            "content": item.content,
            "domain": item.domain,
            "source": item.source,
            "repetition_count": item.repetition_count,
            "volatility_ema": item.volatility_ema,
            "mismatch_count": item.mismatch_count,
            "goal_delta": item.goal_delta,
            "last_confirmed_at": item.last_confirmed_at,
            "last_audited_at": item.last_audited_at,
            "tags": json.dumps(item.tags),
            "superseded_by": item.superseded_by,
        })
        self._conn.commit()

    # ── read ──────────────────────────────────────────────────────────────────

    def get(self, item_id: str) -> Optional[MemoryItem]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id=?", (item_id,)
        ).fetchone()
        return _row_to_item(row) if row else None

    def all_active(
        self, namespace: str | None = None, domain: str | None = None
    ) -> list[MemoryItem]:
        """Active (non-superseded) items. Scope to a tenant via `namespace`;
        pass namespace=None only for cross-tenant/admin queries."""
        clauses = ["superseded_by IS NULL"]
        params: list[object] = []
        if namespace is not None:
            clauses.append("namespace=?")
            params.append(namespace)
        if domain:
            clauses.append("domain=?")
            params.append(domain)
        sql = "SELECT * FROM memories WHERE " + " AND ".join(clauses)
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_item(r) for r in rows]

    def search_by_content(
        self, query: str, namespace: str | None = None, limit: int = 20
    ) -> list[MemoryItem]:
        """Simple keyword search — replace with embedding search for production."""
        clauses = ["superseded_by IS NULL", "LOWER(content) LIKE LOWER(:q)"]
        params: dict[str, object] = {"q": f"%{query}%", "lim": limit}
        if namespace is not None:
            clauses.append("namespace=:ns")
            params["ns"] = namespace
        sql = ("SELECT * FROM memories WHERE " + " AND ".join(clauses)
               + " LIMIT :lim")
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_item(r) for r in rows]

    def delete(self, item_id: str, namespace: str) -> bool:
        """Delete one memory row. Returns True if a row was removed."""
        cur = self._conn.execute(
            "DELETE FROM memories WHERE id=? AND namespace=?",
            (item_id, namespace),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_namespace(self, namespace: str) -> None:
        """Remove every row for a tenant (including superseded history)."""
        self._conn.execute("DELETE FROM memories WHERE namespace=?", (namespace,))
        self._conn.execute(
            "DELETE FROM domain_stats WHERE namespace=?", (namespace,))
        self._conn.commit()

    # ── domain stats (auto-discovery) ─────────────────────────────────────────

    def get_domain_stats(
        self, namespace: str, domain: str
    ) -> Optional[DomainStats]:
        row = self._conn.execute(
            "SELECT * FROM domain_stats WHERE namespace=? AND domain=?",
            (namespace, domain),
        ).fetchone()
        if not row:
            return None
        return DomainStats.from_row(dict(row))

    def all_domain_stats(self, namespace: str) -> dict[str, DomainStats]:
        rows = self._conn.execute(
            "SELECT * FROM domain_stats WHERE namespace=?",
            (namespace,),
        ).fetchall()
        return {dict(r)["domain"]: DomainStats.from_row(dict(r)) for r in rows}

    def upsert_domain_stats(self, namespace: str, stats: DomainStats) -> None:
        row = stats.to_row(namespace)
        self._conn.execute("""
            INSERT INTO domain_stats (
                namespace, domain, n_confirms, n_mismatches,
                n_supersedes, mismatch_sum
            ) VALUES (
                :namespace, :domain, :n_confirms, :n_mismatches,
                :n_supersedes, :mismatch_sum
            )
            ON CONFLICT(namespace, domain) DO UPDATE SET
                n_confirms=:n_confirms,
                n_mismatches=:n_mismatches,
                n_supersedes=:n_supersedes,
                mismatch_sum=:mismatch_sum
        """, row)
        self._conn.commit()

    def delete_domain_stats_namespace(self, namespace: str) -> None:
        self._conn.execute(
            "DELETE FROM domain_stats WHERE namespace=?", (namespace,))
        self._conn.commit()

    def close(self):
        self._conn.close()
