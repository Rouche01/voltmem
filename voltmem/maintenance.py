"""
Maintenance window — asynchronous tasks for memory consolidation.

The maintenance runner operates on a MemoryLayer during idle cycles,
performing operations too expensive or speculative for the write path:

- expire_cleanup: purge rows past expires_at
- reclassify_ambiguous: review low-confidence classifications with full history
- pattern_audit: scan logged_mismatch clusters for emergent change signals

Usage::

    from voltmem import MemoryLayer, MaintenanceWindow

    mem = MemoryLayer("app.db")
    maintenance = MaintenanceWindow(mem)
    maintenance.register("expire_cleanup", expire_cleanup, interval=3600)
    maintenance.run_once("expire_cleanup")
    # or
    maintenance.run_all()   # runs every registered task once

Safety::

- Tasks never block the write path (run in separate thread or external process)
- Default: WAL mode SQLite + read-only snapshot for analysis tasks
- Tasks receive a read-only view; mutations go through the layer's normal API
"""

from __future__ import annotations

import copy
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .memory import MemoryLayer
from .domains import MemoryItem


# ── task registry ─────────────────────────────────────────────────────────────

@dataclass
class MaintenanceTask:
    name: str
    fn: Callable[["MaintenanceContext"], None]
    interval: float  # seconds between runs (0 = manual only)
    last_run: float = 0.0


class MaintenanceContext:
    """Read-only(ish) view passed to maintenance tasks.

    Provides safe access to the memory layer's state without exposing
    the full write API. Tasks that need to mutate call back through the
    layer's public methods (observe, write, etc.).
    """

    def __init__(self, layer: MemoryLayer):
        self._layer = layer
        self.namespace = layer.namespace

    @property
    def store(self):
        """Direct store access for read-only queries."""
        return self._layer._store

    @property
    def layer(self) -> MemoryLayer:
        """The underlying layer — tasks may call public methods for mutations."""
        return self._layer

    def all_active(self, domain: str | None = None) -> list[MemoryItem]:
        return self._layer._active(domain=domain)

    def domain_stats(self) -> dict[str, dict]:
        return self._layer.domain_stats()

    def summary(self) -> dict:
        return self._layer.summary()


class MaintenanceWindow:
    """Pluggable maintenance runner for a MemoryLayer."""

    def __init__(self, layer: MemoryLayer):
        self._layer = layer
        self._tasks: dict[str, MaintenanceTask] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ── registration ──────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        fn: Callable[[MaintenanceContext], None],
        interval: float = 0.0,
    ) -> "MaintenanceWindow":
        """Register a maintenance task.

        Parameters
        ----------
        name : str
            Unique task identifier.
        fn : callable(ctx)
            Task implementation receiving a MaintenanceContext.
        interval : float
            Minimum seconds between auto-runs (0 = manual only).
        """
        with self._lock:
            self._tasks[name] = MaintenanceTask(
                name=name, fn=fn, interval=interval)
        return self

    def unregister(self, name: str) -> bool:
        with self._lock:
            return self._tasks.pop(name, None) is not None

    def tasks(self) -> list[MaintenanceTask]:
        with self._lock:
            return list(self._tasks.values())

    # ── execution ─────────────────────────────────────────────────────────────

    def run_once(self, name: str):
        """Execute a single registered task immediately.
        Returns the task's return value."""
        with self._lock:
            task = self._tasks.get(name)
        if task is None:
            raise KeyError(f"No maintenance task registered: {name!r}")
        ctx = MaintenanceContext(self._layer)
        result = task.fn(ctx)
        task.last_run = time.time()
        return result

    def run_all(self) -> dict[str, bool]:
        """Run every registered task once. Returns {name: success}."""
        results: dict[str, bool] = {}
        for task in self.tasks():
            try:
                self.run_once(task.name)
                results[task.name] = True
            except Exception as e:
                results[task.name] = False
                # Log but don't crash other tasks
                print(f"[maintenance] {task.name} failed: {e}")
        return results

    def run_due(self) -> dict[str, bool]:
        """Run tasks whose interval has elapsed since last run."""
        now = time.time()
        results: dict[str, bool] = {}
        for task in self.tasks():
            if task.interval > 0 and (now - task.last_run) >= task.interval:
                try:
                    self.run_once(task.name)
                    results[task.name] = True
                except Exception as e:
                    results[task.name] = False
                    print(f"[maintenance] {task.name} failed: {e}")
        return results

    # ── background thread ─────────────────────────────────────────────────────

    def start(self, check_interval: float = 60.0) -> None:
        """Start a background thread that runs due tasks periodically."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()

        def _loop() -> None:
            while not self._stop_event.is_set():
                self.run_due()
                self._stop_event.wait(check_interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> "MaintenanceWindow":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


# ── built-in tasks ────────────────────────────────────────────────────────────

def expire_cleanup(ctx: MaintenanceContext) -> int:
    """Purge expired rows and sync the vector index. Returns number deleted."""
    deleted = ctx.layer.purge_expired()
    if deleted:
        print(f"[maintenance] expire_cleanup: purged {deleted} expired rows")
    return deleted


def reclassify_ambiguous(
    ctx: MaintenanceContext,
    *,
    min_mismatch_rate: float = 0.3,
    min_observations: int = 5,
) -> list[dict]:
    """Review domains with high mismatch rates — possible misclassification.

    Flags domains where logged_mismatch + audited exceeds confirm rate,
    suggesting the domain prior may be wrong (or the classifier is
    systematically misfiring).

    Returns a list of flag dicts for operator review; does NOT mutate.
    """
    stats = ctx.domain_stats()
    flags: list[dict] = []
    for domain, s in stats.items():
        total = s.get("inserted", 0) + s.get("confirmed", 0) + s.get("logged_mismatch", 0) + s.get("audited", 0)
        if total < min_observations:
            continue
        mismatch_rate = (s.get("logged_mismatch", 0) + s.get("audited", 0)) / total
        if mismatch_rate >= min_mismatch_rate:
            flags.append({
                "domain": domain,
                "mismatch_rate": round(mismatch_rate, 3),
                "observations": total,
                "prior": s.get("prior"),
                "suggestion": "review classifier or domain prior",
            })
    if flags:
        print(f"[maintenance] reclassify_ambiguous: flagged {len(flags)} domains")
    return flags


def pattern_audit(
    ctx: MaintenanceContext,
    *,
    min_cluster_size: int = 3,
    lookback_days: float = 14.0,
) -> list[dict]:
    """Scan for items with repeated logged_mismatch — possible emergent change.

    Looks at active items with mismatch_count >= min_cluster_size that
    haven't been audited. These are candidates for manual review or
    proactive escalation.

    Returns flag dicts; does NOT mutate.
    """
    now = time.time()
    cutoff = now - lookback_days * 86400
    items = ctx.all_active()
    flags: list[dict] = []
    for item in items:
        if item.mismatch_count < min_cluster_size:
            continue
        if item.last_audited_at and item.last_audited_at > cutoff:
            continue
        flags.append({
            "item_id": item.id,
            "domain": item.domain,
            "content": item.content,
            "mismatch_count": item.mismatch_count,
            "last_confirmed_days_ago": round((now - item.last_confirmed_at) / 86400, 1),
            "suggestion": "review for emergent change",
        })
    if flags:
        print(f"[maintenance] pattern_audit: flagged {len(flags)} items")
    return flags


def consolidate(
    ctx: MaintenanceContext,
    *,
    min_mismatch_count: int = 3,
    min_days_since_audit: float = 7.0,
    dry_run: bool = True,
    event_aware: bool = True,
) -> list[dict]:
    """Proactively reorganize memories that show emergent change signals.

    This is the core sleeptime-compute task: it looks at items with
    accumulated logged_mismatches that never crossed the real-time escalation
    threshold, and decides whether the pattern is strong enough to warrant
    a proactive update.

    When ``dry_run=True`` (default), returns what *would* be done without
    mutating. Set ``dry_run=False`` to actually supersede the old memory.

    When ``event_aware=True`` (default), if a reorganized item belongs to a
    multi-facet event, all other facets of that event are also flagged for
    review — they may need co-ordinated update (e.g. location change implies
    task context change).

    Returns a list of action dicts describing what happened (or would happen).
    """
    now = time.time()
    items = ctx.all_active()
    actions: list[dict] = []
    touched_events: set[str | None] = set()

    for item in items:
        if item.mismatch_count < min_mismatch_count:
            continue
        days_since_audit = (
            (now - item.last_audited_at) / 86400
            if item.last_audited_at else float("inf")
        )
        if days_since_audit < min_days_since_audit:
            continue

        # Build a synthetic observation that captures the emergent change.
        # In a real implementation this might use an LLM to summarize the
        # mismatch history into a new, consolidated fact.
        new_content = f"[consolidated] {item.content}"

        action = {
            "item_id": item.id,
            "domain": item.domain,
            "old_content": item.content,
            "new_content": new_content,
            "mismatch_count": item.mismatch_count,
            "days_since_audit": round(days_since_audit, 1),
            "dry_run": dry_run,
            "event_id": item.event_id,
        }

        if not dry_run:
            # Use the layer's public API to safely supersede
            # force_update=True because the maintenance task has already
            # done its own threshold checking (mismatch_count >= min).
            result = ctx.layer.observe(
                content=new_content,
                domain=item.domain,
                mismatch_magnitude=0.85,  # strong enough to override protection
                source="system_generated",  # lower reliability than explicit
                goal_delta=0.0,
                force_update=True,
            )
            action["result"] = result.action
            action["new_item_id"] = result.item.id

        actions.append(action)
        touched_events.add(item.event_id)

    # Event-aware: flag co-facets for review
    if event_aware:
        for item in items:
            if item.event_id in touched_events and item.id not in {a["item_id"] for a in actions}:
                actions.append({
                    "item_id": item.id,
                    "domain": item.domain,
                    "content": item.content,
                    "event_id": item.event_id,
                    "note": "co-facet of reorganized event — review for consistency",
                    "dry_run": dry_run,
                })

    if actions:
        mode = "would" if dry_run else "did"
        n_reorganized = len([a for a in actions if "mismatch_count" in a])
        n_cofacets = len(actions) - n_reorganized
        print(f"[maintenance] consolidate: {mode} reorganize {n_reorganized} items"
              f" ({n_cofacets} co-facets flagged)")
    return actions
