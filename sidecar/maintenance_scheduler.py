"""Background maintenance scheduler for the HTTP sidecar.

Runs due tasks (expire_cleanup, flag audits, consolidate, reconcile_twins)
on a daemon thread so HTTP request handlers are not blocked. Set
``VOLTMEM_CONSOLIDATE=0`` or ``VOLTMEM_RECONCILE_TWINS=0`` to leave those
out of the daemon while keeping manual trigger.
"""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING

from voltmem import MaintenanceWindow
from voltmem.maintenance import (
    consolidate,
    expire_cleanup,
    pattern_audit,
    reclassify_ambiguous,
    reconcile_twins_default,
)

if TYPE_CHECKING:
    from .memory_pool import MemoryPool


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class SidecarMaintenanceScheduler:
    """Per-tenant ``MaintenanceWindow.run_due`` loop on a daemon thread."""

    def __init__(
        self,
        pool: MemoryPool,
        *,
        check_interval: float | None = None,
        expire_interval: float | None = None,
        pattern_interval: float | None = None,
        reclassify_interval: float | None = None,
        consolidate_interval: float | None = None,
        consolidate_enabled: bool | None = None,
        reconcile_interval: float | None = None,
        reconcile_enabled: bool | None = None,
    ) -> None:
        self._pool = pool
        self._check_interval = (
            check_interval
            if check_interval is not None
            else _env_float("VOLTMEM_MAINTENANCE_CHECK_INTERVAL", 60.0)
        )
        self._expire_interval = (
            expire_interval
            if expire_interval is not None
            else _env_float("VOLTMEM_EXPIRE_INTERVAL", 3600.0)
        )
        self._pattern_interval = (
            pattern_interval
            if pattern_interval is not None
            else _env_float("VOLTMEM_PATTERN_AUDIT_INTERVAL", 3600.0)
        )
        self._reclassify_interval = (
            reclassify_interval
            if reclassify_interval is not None
            else _env_float("VOLTMEM_RECLASSIFY_INTERVAL", 86400.0)
        )
        self._consolidate_interval = (
            consolidate_interval
            if consolidate_interval is not None
            else _env_float("VOLTMEM_CONSOLIDATE_INTERVAL", 86400.0)
        )
        self._consolidate_enabled = (
            consolidate_enabled
            if consolidate_enabled is not None
            else _env_bool("VOLTMEM_CONSOLIDATE", True)
        )
        self._reconcile_interval = (
            reconcile_interval
            if reconcile_interval is not None
            else _env_float("VOLTMEM_RECONCILE_INTERVAL", 86400.0)
        )
        self._reconcile_enabled = (
            reconcile_enabled
            if reconcile_enabled is not None
            else _env_bool("VOLTMEM_RECONCILE_TWINS", True)
        )
        self._windows: dict[str, MaintenanceWindow] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _window_for(self, user_id: str) -> MaintenanceWindow:
        with self._lock:
            mw = self._windows.get(user_id)
            if mw is not None:
                return mw
            mem = self._pool.for_user(user_id)
            mw = MaintenanceWindow(mem.layer)
            mw.register(
                "expire_cleanup", expire_cleanup, interval=self._expire_interval
            )
            mw.register(
                "pattern_audit", pattern_audit, interval=self._pattern_interval
            )
            mw.register(
                "reclassify_ambiguous",
                reclassify_ambiguous,
                interval=self._reclassify_interval,
            )
            if self._consolidate_enabled:
                mw.register(
                    "consolidate",
                    consolidate,
                    interval=self._consolidate_interval,
                )
            if self._reconcile_enabled:
                mw.register(
                    "reconcile_twins",
                    reconcile_twins_default,
                    interval=self._reconcile_interval,
                )
            self._windows[user_id] = mw
            return mw

    def _namespaces(self) -> list[str]:
        root = self._pool._root.layer
        return root._store.list_namespaces(exclude=("__sidecar__",))

    def tick(self) -> dict[str, dict]:
        """Run due tasks once for every known tenant. Returns per-ns summaries."""
        out: dict[str, dict] = {}
        for ns in self._namespaces():
            try:
                mw = self._window_for(ns)
                out[ns] = mw.run_due(dry_run=False)
            except Exception as exc:  # noqa: BLE001 — isolate tenants
                print(f"[sidecar-maintenance] {ns} failed: {exc}")
                out[ns] = {"error": str(exc)}
        return out

    def _loop(self) -> None:
        # Run once shortly after start, then on the check interval.
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001
                print(f"[sidecar-maintenance] tick failed: {exc}")
            self._stop.wait(self._check_interval)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="voltmem-sidecar-maintenance",
            daemon=True,
        )
        self._thread.start()
        print(
            "[sidecar-maintenance] daemon started "
            f"(check={self._check_interval}s expire={self._expire_interval}s"
            f" consolidate={self._consolidate_interval if self._consolidate_enabled else 'off'}s"
            f" reconcile={self._reconcile_interval if self._reconcile_enabled else 'off'}s)"
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        with self._lock:
            self._windows.clear()

    @classmethod
    def maybe_start(cls, pool: MemoryPool) -> "SidecarMaintenanceScheduler | None":
        """Start from env unless ``VOLTMEM_MAINTENANCE=0``."""
        if not _env_bool("VOLTMEM_MAINTENANCE", True):
            return None
        sched = cls(pool)
        sched.start()
        return sched
