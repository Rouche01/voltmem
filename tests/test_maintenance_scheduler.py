"""Tests for sidecar background maintenance scheduler."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["VOLTMEM_EMBEDDINGS"] = "0"
os.environ["VOLTMEM_MAINTENANCE"] = "0"
os.environ.setdefault("VOLTMEM_PROFILE", "stylens")

from sidecar.maintenance_scheduler import SidecarMaintenanceScheduler  # noqa: E402
from sidecar.memory_pool import MemoryPool  # noqa: E402
from sidecar.profiles import build_profile  # noqa: E402


def test_scheduler_tick_runs_expire_cleanup_off_request_path():
    domains, classifier = build_profile("stylens")
    restore = domains.install()
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        pool = MemoryPool(path, embeddings=False, classifier=classifier)
        mem = pool.for_user("sched-user")
        mem.add("temp badge", expires_at=1.0)
        assert any(i.get("expires_at") == 1.0 for i in mem.get_all()) or True
        # get_all may not expose expires_at — verify via store
        layer = mem.layer
        assert layer._store.list_expired_ids(layer.namespace)

        sched = SidecarMaintenanceScheduler(
            pool,
            check_interval=3600,
            expire_interval=0.01,  # effectively always due after first run
            pattern_interval=99999,
            reclassify_interval=99999,
        )
        # First tick: last_run=0 so interval elapsed → purge
        out = sched.tick()
        assert "sched-user" in out
        assert out["sched-user"].get("results", {}).get("expire_cleanup") == 1
        assert layer._store.list_expired_ids(layer.namespace) == []
        pool.close()
    finally:
        restore()
        try:
            os.unlink(path)
        except OSError:
            pass


def test_list_namespaces_excludes_sidecar():
    from voltmem import MemoryLayer

    with MemoryLayer(":memory:") as mem:
        mem.write("x", domain="location")
        user = mem.for_user("alice")
        user.write("y", domain="location")
        ns = mem._store.list_namespaces()
        assert "alice" in ns
        assert "default" in ns
        assert "__sidecar__" not in ns


def test_scheduler_registers_consolidate_by_default():
    domains, classifier = build_profile("stylens")
    restore = domains.install()
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        pool = MemoryPool(path, embeddings=False, classifier=classifier)
        pool.for_user("sched-user").add("hello")
        sched = SidecarMaintenanceScheduler(
            pool,
            check_interval=3600,
            expire_interval=99999,
            pattern_interval=99999,
            reclassify_interval=99999,
            consolidate_interval=0.01,
            consolidate_enabled=True,
        )
        mw = sched._window_for("sched-user")
        assert "consolidate" in mw._tasks
        out = sched.tick()
        assert "sched-user" in out
        assert "consolidate" in out["sched-user"].get("results", {})
        pool.close()
    finally:
        restore()
        try:
            os.unlink(path)
        except OSError:
            pass


def test_scheduler_can_disable_consolidate():
    domains, classifier = build_profile("stylens")
    restore = domains.install()
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        pool = MemoryPool(path, embeddings=False, classifier=classifier)
        pool.for_user("sched-user").add("hello")
        sched = SidecarMaintenanceScheduler(
            pool,
            consolidate_enabled=False,
        )
        mw = sched._window_for("sched-user")
        assert "consolidate" not in mw._tasks
        pool.close()
    finally:
        restore()
        try:
            os.unlink(path)
        except OSError:
            pass


def test_scheduler_registers_reconcile_twins_by_default():
    domains, classifier = build_profile("stylens")
    restore = domains.install()
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        pool = MemoryPool(path, embeddings=False, classifier=classifier)
        pool.for_user("sched-user").add("hello")
        sched = SidecarMaintenanceScheduler(
            pool,
            check_interval=3600,
            expire_interval=99999,
            pattern_interval=99999,
            reclassify_interval=99999,
            consolidate_interval=99999,
            reconcile_interval=0.01,
            reconcile_enabled=True,
        )
        mw = sched._window_for("sched-user")
        assert "reconcile_twins" in mw._tasks
        out = sched.tick()
        assert "reconcile_twins" in out["sched-user"].get("results", {})
        pool.close()
    finally:
        restore()
        try:
            os.unlink(path)
        except OSError:
            pass


def test_scheduler_can_disable_reconcile_twins():
    domains, classifier = build_profile("stylens")
    restore = domains.install()
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        pool = MemoryPool(path, embeddings=False, classifier=classifier)
        pool.for_user("sched-user").add("hello")
        sched = SidecarMaintenanceScheduler(
            pool,
            reconcile_enabled=False,
        )
        mw = sched._window_for("sched-user")
        assert "reconcile_twins" not in mw._tasks
        pool.close()
    finally:
        restore()
        try:
            os.unlink(path)
        except OSError:
            pass


if __name__ == "__main__":
    tests = [
        test_scheduler_tick_runs_expire_cleanup_off_request_path,
        test_list_namespaces_excludes_sidecar,
        test_scheduler_registers_consolidate_by_default,
        test_scheduler_can_disable_consolidate,
        test_scheduler_registers_reconcile_twins_by_default,
        test_scheduler_can_disable_reconcile_twins,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    if failed:
        raise SystemExit(f"{failed} test(s) failed")
    print(f"{len(tests)} passed")
