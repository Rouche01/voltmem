"""
Maintenance window — asynchronous tasks for memory consolidation.

The maintenance runner operates on a MemoryLayer during idle cycles,
performing operations too expensive or speculative for the write path:

- expire_cleanup: purge rows past expires_at
- reclassify_ambiguous: review low-confidence classifications with full history
- pattern_audit: scan logged_mismatch clusters for emergent change signals
- consolidate: reorganize memories with emergent change signals (ledgered)

Mutating tasks record a ``run_id`` ledger so ``MemoryLayer.rollback_maintenance``
can undo supersedes and restore purged snapshots.

Usage::

    from voltmem import MemoryLayer, MaintenanceWindow

    mem = MemoryLayer("app.db")
    maintenance = MaintenanceWindow(mem)
    maintenance.register("expire_cleanup", expire_cleanup, interval=3600)
    result = maintenance.run_once("expire_cleanup")
    # result["run_id"] can be passed to mem.rollback_maintenance(run_id)

Safety::

- Prefer running in a separate thread or process so the write path stays responsive
- Tasks share the layer's SQLite connection; file-backed DBs open in WAL mode by default
- Mutations go through the layer API and are ledgered when a run_id is active
- ``consolidate`` is still a stub content-wise; prefer explicit ``task=consolidate``
  until a real summarizer ships — default sidecar ``run_all`` skips it
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .memory import MemoryLayer, memory_item_to_payload
from .domains import MemoryItem


# ── task registry ─────────────────────────────────────────────────────────────

@dataclass
class MaintenanceTask:
    name: str
    fn: Callable[["MaintenanceContext"], Any]
    interval: float  # seconds between runs (0 = manual only)
    last_run: float = 0.0


class MaintenanceContext:
    """View passed to maintenance tasks.

    Provides store/layer access. When ``run_id`` is set, mutating helpers
    append to the maintenance ledger for possible later rollback.
    """

    def __init__(
        self,
        layer: MemoryLayer,
        *,
        run_id: str | None = None,
        dry_run: bool = False,
        task: str | None = None,
    ):
        self._layer = layer
        self.namespace = layer.namespace
        self.run_id = run_id
        self.dry_run = dry_run
        self.task = task

    @property
    def store(self):
        """Store access for queries and ledger helpers."""
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

    def record_supersede(self, old_id: str, new_id: str, **extra: Any) -> None:
        if not self.run_id or self.dry_run:
            return
        self.store.record_maintenance_action(
            self.run_id,
            self.task or "unknown",
            "supersede",
            old_id=old_id,
            new_id=new_id,
            payload=extra,
        )

    def record_purge(self, item: MemoryItem, **extra: Any) -> None:
        if not self.run_id or self.dry_run:
            return
        payload = memory_item_to_payload(item)
        payload.update(extra)
        self.store.record_maintenance_action(
            self.run_id,
            self.task or "unknown",
            "purge",
            old_id=item.id,
            payload=payload,
        )


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
        fn: Callable[[MaintenanceContext], Any],
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

    def run_once(
        self,
        name: str,
        *,
        dry_run: bool = False,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a single registered task.

        Opens a ledger run (unless ``run_id`` is provided). Returns
        ``{"run_id", "task", "dry_run", "result"}``.
        """
        with self._lock:
            task = self._tasks.get(name)
        if task is None:
            raise KeyError(f"No maintenance task registered: {name!r}")

        own_run = run_id is None
        rid = run_id or self._layer.start_maintenance_run(dry_run=dry_run)
        ctx = MaintenanceContext(
            self._layer, run_id=rid, dry_run=dry_run, task=name)
        try:
            result = task.fn(ctx)
            task.last_run = time.time()
            return {
                "run_id": rid,
                "task": name,
                "dry_run": dry_run,
                "result": result,
            }
        finally:
            if own_run:
                self._layer.finish_maintenance_run(rid)

    def run_all(self, *, dry_run: bool = False) -> dict[str, Any]:
        """Run every registered task once under a shared ``run_id``."""
        rid = self._layer.start_maintenance_run(dry_run=dry_run)
        results: dict[str, Any] = {}
        errors: dict[str, str] = {}
        try:
            for task in self.tasks():
                try:
                    out = self.run_once(
                        task.name, dry_run=dry_run, run_id=rid)
                    results[task.name] = out["result"]
                except Exception as e:
                    errors[task.name] = str(e)
                    print(f"[maintenance] {task.name} failed: {e}")
        finally:
            self._layer.finish_maintenance_run(rid)
        return {
            "run_id": rid,
            "dry_run": dry_run,
            "results": results,
            "errors": errors,
        }

    def run_due(self, *, dry_run: bool = False) -> dict[str, Any]:
        """Run tasks whose interval has elapsed since last run (shared run_id)."""
        now = time.time()
        due = [
            t for t in self.tasks()
            if t.interval > 0 and (now - t.last_run) >= t.interval
        ]
        if not due:
            return {"run_id": None, "dry_run": dry_run, "results": {}, "errors": {}}
        rid = self._layer.start_maintenance_run(dry_run=dry_run)
        results: dict[str, Any] = {}
        errors: dict[str, str] = {}
        try:
            for task in due:
                try:
                    out = self.run_once(
                        task.name, dry_run=dry_run, run_id=rid)
                    results[task.name] = out["result"]
                except Exception as e:
                    errors[task.name] = str(e)
                    print(f"[maintenance] {task.name} failed: {e}")
        finally:
            self._layer.finish_maintenance_run(rid)
        return {
            "run_id": rid,
            "dry_run": dry_run,
            "results": results,
            "errors": errors,
        }

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

def expire_cleanup(
    ctx: MaintenanceContext, *, dry_run: bool | None = None
) -> int:
    """Purge expired rows (and sync the vector index).

    When dry-run, counts rows that *would* be deleted without mutating.
    Snapshots purged rows into the ledger for ``rollback_maintenance``.
    """
    is_dry = ctx.dry_run if dry_run is None else dry_run
    ids = ctx.store.list_expired_ids(ctx.namespace)
    if is_dry:
        if ids:
            print(f"[maintenance] expire_cleanup: would purge {len(ids)} expired rows (dry_run)")
        return len(ids)

    deleted = 0
    for item_id in ids:
        item = ctx.store.get(item_id)
        if item is None or item.namespace != ctx.namespace:
            continue
        ctx.record_purge(item)
        if ctx.layer.remove(item_id):
            deleted += 1
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
    dry_run: bool | None = None,
    event_aware: bool = True,
) -> list[dict]:
    """Proactively reorganize memories that show emergent change signals.

    Stub content: prefixes ``[consolidated]`` today. Mutations are ledgered
    (``supersede``) for ``rollback_maintenance``.

    When dry-run, returns what *would* be done without mutating.
    """
    is_dry = ctx.dry_run if dry_run is None else dry_run
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

        # Stub: real implementation should summarize mismatch history.
        new_content = f"[consolidated] {item.content}"

        action = {
            "item_id": item.id,
            "domain": item.domain,
            "old_content": item.content,
            "new_content": new_content,
            "mismatch_count": item.mismatch_count,
            "days_since_audit": round(days_since_audit, 1),
            "dry_run": is_dry,
            "event_id": item.event_id,
        }

        if not is_dry:
            result = ctx.layer.observe(
                content=new_content,
                domain=item.domain,
                mismatch_magnitude=0.85,
                source="system_generated",
                goal_delta=0.0,
                force_update=True,
            )
            action["result"] = result.action
            action["new_item_id"] = result.item.id
            if result.action == "audited":
                ctx.record_supersede(item.id, result.item.id)

        actions.append(action)
        touched_events.add(item.event_id)

    if event_aware:
        action_ids = {a["item_id"] for a in actions}
        for item in items:
            if item.event_id in touched_events and item.id not in action_ids:
                actions.append({
                    "item_id": item.id,
                    "domain": item.domain,
                    "content": item.content,
                    "event_id": item.event_id,
                    "note": "co-facet of reorganized event — review for consistency",
                    "dry_run": is_dry,
                })

    if actions:
        mode = "would" if is_dry else "did"
        n_reorganized = len([a for a in actions if "mismatch_count" in a])
        n_cofacets = len(actions) - n_reorganized
        print(f"[maintenance] consolidate: {mode} reorganize {n_reorganized} items"
              f" ({n_cofacets} co-facets flagged)")
    return actions
