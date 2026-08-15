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


def _configure_sqlite(conn: sqlite3.Connection, db_path: str | Path) -> None:
    """Tune connection for sidecar + maintenance concurrency.

    File-backed DBs use WAL so readers (HTTP) and writers (maintenance daemon)
    contend less. ``:memory:`` is left alone (WAL is meaningless / unsupported
    the same way for ephemeral DBs).
    """
    path = str(db_path)
    # busy_timeout helps either mode when the other connection holds a lock briefly
    conn.execute("PRAGMA busy_timeout=5000")
    if path == ":memory:" or path.startswith("file::") and "mode=memory" in path:
        return
    # journal_mode=WAL persists on the database file; safe to re-run every open
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id                TEXT PRIMARY KEY,
    content           TEXT NOT NULL,
    domain            TEXT NOT NULL,
    source            TEXT NOT NULL,
    repetition_count  INTEGER DEFAULT 1,
    volatility_ema    REAL    DEFAULT -1.0,
    surprise_ema      REAL    DEFAULT 0.0,
    surprise_at       REAL    DEFAULT 0.0,
    mismatch_ema      REAL    DEFAULT -1.0,
    mismatch_var      REAL    DEFAULT -1.0,
    mismatch_count    INTEGER DEFAULT 0,
    goal_delta        REAL    DEFAULT 0.0,
    created_at        REAL    NOT NULL,
    last_confirmed_at REAL    NOT NULL,
    last_audited_at   REAL    DEFAULT 0.0,
    tags              TEXT    DEFAULT '[]',
    facts             TEXT    DEFAULT '[]',
    superseded_by     TEXT    DEFAULT NULL,
    namespace         TEXT    NOT NULL DEFAULT 'default',
    event_id          TEXT    DEFAULT NULL,
    modality          TEXT    DEFAULT NULL,
    expires_at        REAL    DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_domain ON memories(namespace, domain);
CREATE INDEX IF NOT EXISTS idx_active ON memories(namespace, superseded_by);
CREATE INDEX IF NOT EXISTS idx_event ON memories(namespace, event_id);
CREATE TABLE IF NOT EXISTS domain_stats (
    namespace     TEXT NOT NULL,
    domain        TEXT NOT NULL,
    n_confirms    INTEGER DEFAULT 0,
    n_mismatches  INTEGER DEFAULT 0,
    n_supersedes  INTEGER DEFAULT 0,
    n_inserts     INTEGER DEFAULT 0,
    mismatch_sum  REAL    DEFAULT 0.0,
    PRIMARY KEY (namespace, domain)
);
CREATE TABLE IF NOT EXISTS maintenance_runs (
    run_id          TEXT PRIMARY KEY,
    namespace       TEXT NOT NULL,
    dry_run         INTEGER NOT NULL DEFAULT 0,
    started_at      REAL NOT NULL,
    finished_at     REAL,
    rolled_back_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_maint_runs_ns ON maintenance_runs(namespace);
CREATE TABLE IF NOT EXISTS maintenance_actions (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    task            TEXT NOT NULL,
    action_type     TEXT NOT NULL,
    old_id          TEXT,
    new_id          TEXT,
    payload_json    TEXT DEFAULT '{}',
    created_at      REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES maintenance_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_maint_actions_run ON maintenance_actions(run_id);
CREATE TABLE IF NOT EXISTS mismatch_evidence (
    id                  TEXT PRIMARY KEY,
    item_id             TEXT NOT NULL,
    namespace           TEXT NOT NULL,
    content             TEXT NOT NULL,
    mismatch_magnitude  REAL NOT NULL,
    source              TEXT NOT NULL,
    created_at          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mismatch_evidence_item
    ON mismatch_evidence(item_id, created_at);
CREATE INDEX IF NOT EXISTS idx_mismatch_evidence_ns
    ON mismatch_evidence(namespace);
"""


def _parse_facts(raw) -> list[dict]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def _row_to_item(row: sqlite3.Row) -> MemoryItem:
    keys = row.keys()
    return MemoryItem(
        id=row["id"],
        content=row["content"],
        domain=row["domain"],
        source=row["source"],
        namespace=row["namespace"] if "namespace" in keys else "default",
        event_id=row["event_id"] if "event_id" in keys else None,
        modality=row["modality"] if "modality" in keys else None,
        expires_at=row["expires_at"] if "expires_at" in keys else None,
        repetition_count=row["repetition_count"],
        volatility_ema=row["volatility_ema"],
        surprise_ema=row["surprise_ema"] if "surprise_ema" in keys else 0.0,
        surprise_at=row["surprise_at"] if "surprise_at" in keys else 0.0,
        mismatch_ema=row["mismatch_ema"] if "mismatch_ema" in keys else -1.0,
        mismatch_var=row["mismatch_var"] if "mismatch_var" in keys else -1.0,
        mismatch_count=row["mismatch_count"],
        goal_delta=row["goal_delta"],
        created_at=row["created_at"],
        last_confirmed_at=row["last_confirmed_at"],
        last_audited_at=row["last_audited_at"],
        tags=json.loads(row["tags"]),
        facts=_parse_facts(row["facts"] if "facts" in keys else []),
        superseded_by=row["superseded_by"],
    )


class MemoryStore:
    """
    Low-level SQLite store. Use MemoryLayer (memory.py) for the
    full volatility-aware read/write policy.
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        _configure_sqlite(self._conn, self._db_path)
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
        if "event_id" not in cols:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN event_id "
                "TEXT DEFAULT NULL")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event ON memories(namespace, event_id)")
        if "modality" not in cols:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN modality "
                "TEXT DEFAULT NULL")
        if "expires_at" not in cols:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN expires_at "
                "REAL DEFAULT NULL")
        if "surprise_ema" not in cols:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN surprise_ema "
                "REAL NOT NULL DEFAULT 0.0")
        if "surprise_at" not in cols:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN surprise_at "
                "REAL NOT NULL DEFAULT 0.0")
        if "mismatch_ema" not in cols:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN mismatch_ema "
                "REAL NOT NULL DEFAULT -1.0")
        if "mismatch_var" not in cols:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN mismatch_var "
                "REAL NOT NULL DEFAULT -1.0")
        if "facts" not in cols:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN facts "
                "TEXT NOT NULL DEFAULT '[]'")
        ds_cols = {
            r[1] for r in self._conn.execute("PRAGMA table_info(domain_stats)")
        }
        if ds_cols and "n_inserts" not in ds_cols:
            self._conn.execute(
                "ALTER TABLE domain_stats ADD COLUMN n_inserts "
                "INTEGER DEFAULT 0")
        # Maintenance ledger (idempotent CREATE IF NOT EXISTS)
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS maintenance_runs (
                run_id          TEXT PRIMARY KEY,
                namespace       TEXT NOT NULL,
                dry_run         INTEGER NOT NULL DEFAULT 0,
                started_at      REAL NOT NULL,
                finished_at     REAL,
                rolled_back_at  REAL
            );
            CREATE INDEX IF NOT EXISTS idx_maint_runs_ns
                ON maintenance_runs(namespace);
            CREATE TABLE IF NOT EXISTS maintenance_actions (
                id              TEXT PRIMARY KEY,
                run_id          TEXT NOT NULL,
                task            TEXT NOT NULL,
                action_type     TEXT NOT NULL,
                old_id          TEXT,
                new_id          TEXT,
                payload_json    TEXT DEFAULT '{}',
                created_at      REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_maint_actions_run
                ON maintenance_actions(run_id);
            CREATE TABLE IF NOT EXISTS mismatch_evidence (
                id                  TEXT PRIMARY KEY,
                item_id             TEXT NOT NULL,
                namespace           TEXT NOT NULL,
                content             TEXT NOT NULL,
                mismatch_magnitude  REAL NOT NULL,
                source              TEXT NOT NULL,
                created_at          REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_mismatch_evidence_item
                ON mismatch_evidence(item_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_mismatch_evidence_ns
                ON mismatch_evidence(namespace);
        """)

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
                repetition_count, volatility_ema, surprise_ema, surprise_at,
                mismatch_ema, mismatch_var,
                mismatch_count, goal_delta,
                created_at, last_confirmed_at, last_audited_at,
                tags, facts, superseded_by, namespace,
                event_id, modality, expires_at
            ) VALUES (
                :id, :content, :domain, :source,
                :repetition_count, :volatility_ema, :surprise_ema, :surprise_at,
                :mismatch_ema, :mismatch_var,
                :mismatch_count, :goal_delta,
                :created_at, :last_confirmed_at, :last_audited_at,
                :tags, :facts, :superseded_by, :namespace,
                :event_id, :modality, :expires_at
            )""", {
            "id": item.id,
            "content": item.content,
            "domain": item.domain,
            "source": item.source,
            "repetition_count": item.repetition_count,
            "volatility_ema": item.volatility_ema,
            "surprise_ema": item.surprise_ema,
            "surprise_at": item.surprise_at,
            "mismatch_ema": item.mismatch_ema,
            "mismatch_var": item.mismatch_var,
            "mismatch_count": item.mismatch_count,
            "goal_delta": item.goal_delta,
            "created_at": item.created_at,
            "last_confirmed_at": item.last_confirmed_at,
            "last_audited_at": item.last_audited_at,
            "tags": json.dumps(item.tags),
            "facts": json.dumps(item.facts or []),
            "superseded_by": item.superseded_by,
            "namespace": item.namespace,
            "event_id": item.event_id,
            "modality": item.modality,
            "expires_at": item.expires_at,
        })
        self._conn.commit()
        return item

    def update(self, item: MemoryItem) -> None:
        self._conn.execute("""
            UPDATE memories SET
                content=:content, domain=:domain, source=:source,
                repetition_count=:repetition_count,
                volatility_ema=:volatility_ema,
                surprise_ema=:surprise_ema,
                surprise_at=:surprise_at,
                mismatch_ema=:mismatch_ema,
                mismatch_var=:mismatch_var,
                mismatch_count=:mismatch_count,
                goal_delta=:goal_delta,
                last_confirmed_at=:last_confirmed_at,
                last_audited_at=:last_audited_at,
                tags=:tags,
                facts=:facts,
                superseded_by=:superseded_by,
                event_id=:event_id,
                modality=:modality,
                expires_at=:expires_at
            WHERE id=:id
        """, {
            "id": item.id,
            "content": item.content,
            "domain": item.domain,
            "source": item.source,
            "repetition_count": item.repetition_count,
            "volatility_ema": item.volatility_ema,
            "surprise_ema": item.surprise_ema,
            "surprise_at": item.surprise_at,
            "mismatch_ema": item.mismatch_ema,
            "mismatch_var": item.mismatch_var,
            "mismatch_count": item.mismatch_count,
            "goal_delta": item.goal_delta,
            "last_confirmed_at": item.last_confirmed_at,
            "last_audited_at": item.last_audited_at,
            "tags": json.dumps(item.tags),
            "facts": json.dumps(item.facts or []),
            "superseded_by": item.superseded_by,
            "event_id": item.event_id,
            "modality": item.modality,
            "expires_at": item.expires_at,
        })
        self._conn.commit()

    # ── read ──────────────────────────────────────────────────────────────────

    def get(self, item_id: str) -> Optional[MemoryItem]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id=?", (item_id,)
        ).fetchone()
        return _row_to_item(row) if row else None

    def all_active(
        self, namespace: str | None = None, domain: str | None = None,
        event_id: str | None = None,
    ) -> list[MemoryItem]:
        """Active (non-superseded) items. Scope to a tenant via `namespace`;
        pass namespace=None only for cross-tenant/admin queries.
        Optionally filter by `event_id` to retrieve facets of a multi-facet event.
        """
        clauses = ["superseded_by IS NULL"]
        params: list[object] = []
        if namespace is not None:
            clauses.append("namespace=?")
            params.append(namespace)
        if domain:
            clauses.append("domain=?")
            params.append(domain)
        if event_id:
            clauses.append("event_id=?")
            params.append(event_id)
        sql = "SELECT * FROM memories WHERE " + " AND ".join(clauses)
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_item(r) for r in rows]

    def get_by_event(self, namespace: str, event_id: str) -> list[MemoryItem]:
        """All items (active and superseded) for a given event, ordered by creation time."""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE namespace=? AND event_id=? ORDER BY created_at",
            (namespace, event_id),
        ).fetchall()
        return [_row_to_item(r) for r in rows]

    def list_expired_ids(
        self, namespace: str, *, now: float | None = None
    ) -> list[str]:
        """Return ids of items whose expires_at is in the past (no delete)."""
        ts = time.time() if now is None else now
        rows = self._conn.execute(
            "SELECT id FROM memories WHERE namespace=? AND expires_at IS NOT NULL AND expires_at < ?",
            (namespace, ts),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def purge_expired(self, namespace: str, *, now: float | None = None) -> list[str]:
        """Hard-delete items whose expires_at is in the past.

        Returns the deleted item ids so callers (``MemoryLayer``) can sync the
        vector index. Prefer ``MemoryLayer.purge_expired()`` when an index is in use.
        """
        ts = time.time() if now is None else now
        ids = self.list_expired_ids(namespace, now=ts)
        if not ids:
            return []
        self.delete_mismatch_evidence_for_items(ids)
        self._conn.execute(
            "DELETE FROM memories WHERE namespace=? AND expires_at IS NOT NULL AND expires_at < ?",
            (namespace, ts),
        )
        self._conn.commit()
        return ids

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
        if cur.rowcount > 0:
            self.delete_mismatch_evidence_for_items([item_id])
        self._conn.commit()
        return cur.rowcount > 0

    def delete_namespace(self, namespace: str) -> None:
        """Remove every row for a tenant (including superseded history)."""
        self._conn.execute("DELETE FROM memories WHERE namespace=?", (namespace,))
        self._conn.execute(
            "DELETE FROM domain_stats WHERE namespace=?", (namespace,))
        self._conn.execute(
            "DELETE FROM mismatch_evidence WHERE namespace=?", (namespace,))
        self._conn.commit()

    def list_namespaces(self, *, exclude: tuple[str, ...] = ("__sidecar__",)) -> list[str]:
        """Distinct tenant namespaces present in memories or domain_stats."""
        rows = self._conn.execute(
            """
            SELECT namespace FROM memories
            UNION
            SELECT namespace FROM domain_stats
            ORDER BY 1
            """
        ).fetchall()
        skip = set(exclude)
        return [str(r[0]) for r in rows if r[0] not in skip]

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
                n_supersedes, n_inserts, mismatch_sum
            ) VALUES (
                :namespace, :domain, :n_confirms, :n_mismatches,
                :n_supersedes, :n_inserts, :mismatch_sum
            )
            ON CONFLICT(namespace, domain) DO UPDATE SET
                n_confirms=:n_confirms,
                n_mismatches=:n_mismatches,
                n_supersedes=:n_supersedes,
                n_inserts=:n_inserts,
                mismatch_sum=:mismatch_sum
        """, row)
        self._conn.commit()

    def delete_domain_stats_namespace(self, namespace: str) -> None:
        self._conn.execute(
            "DELETE FROM domain_stats WHERE namespace=?", (namespace,))
        self._conn.commit()

    # ── mismatch evidence (consolidate / sleeptime) ───────────────────────────

    def append_mismatch_evidence(
        self,
        item_id: str,
        namespace: str,
        content: str,
        *,
        mismatch_magnitude: float,
        source: str,
        created_at: float | None = None,
        evidence_id: str | None = None,
    ) -> str:
        """Persist a rejected observation text for later consolidate review."""
        eid = evidence_id or str(uuid.uuid4())
        ts = time.time() if created_at is None else created_at
        self._conn.execute(
            """INSERT INTO mismatch_evidence
               (id, item_id, namespace, content, mismatch_magnitude, source, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (eid, item_id, namespace, content, mismatch_magnitude, source, ts),
        )
        self._conn.commit()
        return eid

    def list_mismatch_evidence(
        self, item_id: str, *, limit: int = 50
    ) -> list[dict]:
        """Evidence rows for an item, oldest first (capped by ``limit`` most recent)."""
        if limit <= 0:
            return []
        # Fetch newest ``limit`` rows, then return chronological order.
        rows = self._conn.execute(
            """SELECT * FROM mismatch_evidence
               WHERE item_id=?
               ORDER BY created_at DESC
               LIMIT ?""",
            (item_id, limit),
        ).fetchall()
        out = [dict(r) for r in reversed(rows)]
        return out

    def delete_mismatch_evidence_for_items(self, item_ids: list[str]) -> int:
        """Hard-delete evidence rows for purged memory ids. Returns rows removed."""
        if not item_ids:
            return 0
        placeholders = ",".join("?" * len(item_ids))
        cur = self._conn.execute(
            f"DELETE FROM mismatch_evidence WHERE item_id IN ({placeholders})",
            item_ids,
        )
        self._conn.commit()
        return cur.rowcount

    # ── maintenance ledger ────────────────────────────────────────────────────

    def start_maintenance_run(
        self, namespace: str, *, dry_run: bool = False, run_id: str | None = None
    ) -> str:
        rid = run_id or str(uuid.uuid4())
        self._conn.execute(
            """INSERT INTO maintenance_runs
               (run_id, namespace, dry_run, started_at)
               VALUES (?, ?, ?, ?)""",
            (rid, namespace, 1 if dry_run else 0, time.time()),
        )
        self._conn.commit()
        return rid

    def finish_maintenance_run(self, run_id: str) -> None:
        self._conn.execute(
            "UPDATE maintenance_runs SET finished_at=? WHERE run_id=?",
            (time.time(), run_id),
        )
        self._conn.commit()

    def mark_maintenance_rolled_back(self, run_id: str) -> None:
        self._conn.execute(
            "UPDATE maintenance_runs SET rolled_back_at=? WHERE run_id=?",
            (time.time(), run_id),
        )
        self._conn.commit()

    def get_maintenance_run(self, run_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM maintenance_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def record_maintenance_action(
        self,
        run_id: str,
        task: str,
        action_type: str,
        *,
        old_id: str | None = None,
        new_id: str | None = None,
        payload: dict | None = None,
    ) -> str:
        aid = str(uuid.uuid4())
        self._conn.execute(
            """INSERT INTO maintenance_actions
               (id, run_id, task, action_type, old_id, new_id, payload_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                aid,
                run_id,
                task,
                action_type,
                old_id,
                new_id,
                json.dumps(payload or {}),
                time.time(),
            ),
        )
        self._conn.commit()
        return aid

    def list_maintenance_actions(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM maintenance_actions WHERE run_id=? ORDER BY created_at",
            (run_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json") or "{}")
            out.append(d)
        return out

    def close(self):
        self._conn.close()
