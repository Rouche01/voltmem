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
    superseded_by     TEXT    DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_domain ON memories(domain);
CREATE INDEX IF NOT EXISTS idx_active ON memories(superseded_by);
"""


def _row_to_item(row: sqlite3.Row) -> MemoryItem:
    return MemoryItem(
        id=row["id"],
        content=row["content"],
        domain=row["domain"],
        source=row["source"],
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
        self._conn.commit()

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
            INSERT INTO memories VALUES (
                :id, :content, :domain, :source,
                :repetition_count, :volatility_ema, :mismatch_count, :goal_delta,
                :created_at, :last_confirmed_at, :last_audited_at,
                :tags, :superseded_by
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

    def all_active(self, domain: str | None = None) -> list[MemoryItem]:
        if domain:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE superseded_by IS NULL AND domain=?",
                (domain,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE superseded_by IS NULL"
            ).fetchall()
        return [_row_to_item(r) for r in rows]

    def search_by_content(self, query: str, limit: int = 20) -> list[MemoryItem]:
        """Simple keyword search — replace with embedding search for production."""
        rows = self._conn.execute(
            """SELECT * FROM memories
               WHERE superseded_by IS NULL
               AND LOWER(content) LIKE LOWER(:q)
               LIMIT :lim""",
            {"q": f"%{query}%", "lim": limit}
        ).fetchall()
        return [_row_to_item(r) for r in rows]

    def close(self):
        self._conn.close()
