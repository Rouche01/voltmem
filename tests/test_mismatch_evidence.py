"""Tests for mismatch_evidence store + observe wiring."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voltmem import MemoryLayer
from voltmem.store import MemoryStore


def test_append_and_list_mismatch_evidence():
    store = MemoryStore(":memory:")
    eid = store.append_mismatch_evidence(
        "item-1",
        "default",
        "User mentioned a new job",
        mismatch_magnitude=0.6,
        source="weak_inference",
        created_at=100.0,
    )
    assert eid
    store.append_mismatch_evidence(
        "item-1",
        "default",
        "User talked about work again",
        mismatch_magnitude=0.55,
        source="weak_inference",
        created_at=200.0,
    )
    rows = store.list_mismatch_evidence("item-1")
    assert len(rows) == 2
    assert rows[0]["content"] == "User mentioned a new job"
    assert rows[0]["created_at"] == 100.0
    assert rows[1]["content"] == "User talked about work again"
    assert rows[1]["mismatch_magnitude"] == 0.55
    store.close()


def test_list_mismatch_evidence_respects_limit():
    store = MemoryStore(":memory:")
    for i in range(5):
        store.append_mismatch_evidence(
            "item-1",
            "default",
            f"signal {i}",
            mismatch_magnitude=0.5,
            source="weak_inference",
            created_at=float(i),
        )
    rows = store.list_mismatch_evidence("item-1", limit=2)
    assert [r["content"] for r in rows] == ["signal 3", "signal 4"]
    store.close()


def test_logged_mismatch_persists_observation_text():
    with MemoryLayer(":memory:") as mem:
        item = mem.write("User works as a data analyst", domain="biographical")
        stored = mem._store.get(item.item.id)
        stored.volatility_ema = 0.12
        mem._store.update(stored)

        texts = (
            "User mentioned a different role in passing",
            "User said something else about work",
        )
        for text in texts:
            r = mem.observe(
                text,
                domain="biographical",
                mismatch_magnitude=0.65,
                source="weak_inference",
            )
            assert r.action == "logged_mismatch"

        evidence = mem._store.list_mismatch_evidence(item.item.id)
        assert [e["content"] for e in evidence] == list(texts)
        assert all(e["source"] == "weak_inference" for e in evidence)
        assert all(e["namespace"] == mem.namespace for e in evidence)


def test_confirm_and_audit_do_not_write_evidence():
    with MemoryLayer(":memory:") as mem:
        written = mem.write("User likes tea", domain="core_preference")
        confirm = mem.observe(
            "User likes tea",
            domain="core_preference",
            mismatch_magnitude=0.05,
            source="explicit_statement",
        )
        assert confirm.action == "confirmed"
        assert mem._store.list_mismatch_evidence(written.item.id) == []

        career = mem.write(
            "User works as a data analyst", domain="professional_context"
        )
        audited = mem.observe(
            "User explicitly said they changed careers and now work as a nurse",
            domain="professional_context",
            mismatch_magnitude=0.90,
            source="explicit_statement",
        )
        assert audited.action == "audited"
        assert mem._store.list_mismatch_evidence(career.item.id) == []
        assert mem._store.list_mismatch_evidence(audited.item.id) == []


def test_delete_purges_mismatch_evidence():
    with MemoryLayer(":memory:") as mem:
        item = mem.write("User works as a data analyst", domain="biographical")
        stored = mem._store.get(item.item.id)
        stored.volatility_ema = 0.12
        mem._store.update(stored)
        mem.observe(
            "User mentioned a different role",
            domain="biographical",
            mismatch_magnitude=0.65,
            source="weak_inference",
        )
        assert mem._store.list_mismatch_evidence(item.item.id)
        assert mem.remove(item.item.id)
        assert mem._store.list_mismatch_evidence(item.item.id) == []


def test_purge_expired_clears_mismatch_evidence():
    with MemoryLayer(":memory:") as mem:
        item = mem.write(
            "temp note",
            domain="current_task",
            expires_at=time.time() - 10,
        )
        mem._store.append_mismatch_evidence(
            item.item.id,
            mem.namespace,
            "stale signal",
            mismatch_magnitude=0.5,
            source="weak_inference",
        )
        assert mem.purge_expired() == 1
        assert mem._store.list_mismatch_evidence(item.item.id) == []


def test_delete_namespace_clears_mismatch_evidence():
    store = MemoryStore(":memory:")
    store.append_mismatch_evidence(
        "item-1",
        "tenant-a",
        "signal",
        mismatch_magnitude=0.5,
        source="weak_inference",
    )
    store.append_mismatch_evidence(
        "item-2",
        "tenant-b",
        "other",
        mismatch_magnitude=0.5,
        source="weak_inference",
    )
    store.delete_namespace("tenant-a")
    assert store.list_mismatch_evidence("item-1") == []
    assert len(store.list_mismatch_evidence("item-2")) == 1
    store.close()


def test_mismatch_evidence_migration_on_existing_db():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "legacy.db"
        store = MemoryStore(db)
        store.close()
        store2 = MemoryStore(db)
        store2.append_mismatch_evidence(
            "item-1",
            "default",
            "after migrate",
            mismatch_magnitude=0.4,
            source="weak_inference",
        )
        assert store2.list_mismatch_evidence("item-1")[0]["content"] == "after migrate"
        store2.close()


if __name__ == "__main__":
    tests = [
        test_append_and_list_mismatch_evidence,
        test_list_mismatch_evidence_respects_limit,
        test_logged_mismatch_persists_observation_text,
        test_confirm_and_audit_do_not_write_evidence,
        test_delete_purges_mismatch_evidence,
        test_purge_expired_clears_mismatch_evidence,
        test_delete_namespace_clears_mismatch_evidence,
        test_mismatch_evidence_migration_on_existing_db,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
