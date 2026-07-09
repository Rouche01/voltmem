"""Tests for the product-facing Memory API."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voltmem import Memory, create_memory  # noqa: E402


def test_create_memory_and_add_search():
    with create_memory(":memory:", user_id="u1", embeddings=False) as mem:
        mem.add("I live in Berlin")
        mem.add("I live in Paris now")
        hits = mem.search("where does the user live", limit=3)
        assert hits
        assert any("Paris" in h["memory"] for h in hits)


def test_add_messages_list():
    with Memory(user_id="u2", db_path=":memory:") as mem:
        out = mem.add([
            {"role": "user", "content": "I prefer dark mode"},
            {"role": "assistant", "content": "Noted."},
        ])
        assert len(out) >= 1
        assert mem.get_all()


def test_delete_and_clear():
    with Memory(user_id="u3", db_path=":memory:") as mem:
        row = mem.add("temporary fact")
        mid = row["id"]
        assert mem.delete(mid)
        assert mem.get(mid) is None
        mem.add("another")
        mem.clear()
        assert mem.get_all() == []


def test_multi_tenant_isolation():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        with Memory(user_id="alice", db_path=path) as a, \
             Memory(user_id="bob", db_path=path) as b:
            a.add("I live in Berlin")
            b.add("I live in Paris")
            assert "Berlin" in a.search("where live")[0]["memory"]
            assert "Paris" in b.search("where live")[0]["memory"]
    finally:
        os.unlink(path)


if __name__ == "__main__":
    tests = [
        test_create_memory_and_add_search,
        test_add_messages_list,
        test_delete_and_clear,
        test_multi_tenant_isolation,
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
