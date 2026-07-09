"""
Tests for vector index backends and MemoryLayer ANN retrieval.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voltmem import MemoryLayer
from voltmem.embeddings import _cosine
from voltmem.vector_index import BruteForceVectorIndex, SqliteVectorIndex, cosine_similarity


# ── deterministic toy embedder for parity tests ───────────────────────────────

_KEYWORDS = ("berlin", "paris", "concise", "stressed", "billing", "auth")


def _toy_embed(text: str) -> list[float]:
    t = text.lower()
    vec = [1.0 if kw in t else 0.0 for kw in _KEYWORDS]
    return vec if any(vec) else [0.1] * len(_KEYWORDS)


def _toy_similarity(a: str, b: str) -> float:
    return max(0.0, _cosine(_toy_embed(a), _toy_embed(b)))


def _layer_with_index(mode: str = "brute"):
    return MemoryLayer(
        ":memory:",
        similarity_fn=_toy_similarity,
        embed_fn=_toy_embed,
        vector_index=mode,
        namespace="test",
    )


# ── index unit tests ──────────────────────────────────────────────────────────

def test_brute_force_search_orders_by_similarity():
    idx = BruteForceVectorIndex()
    q = [1.0, 0.0, 0.0]
    idx.upsert("a", "u1", "location", [1.0, 0.0, 0.0])
    idx.upsert("b", "u1", "location", [0.8, 0.2, 0.0])
    idx.upsert("c", "u1", "location", [0.0, 1.0, 0.0])
    hits = idx.search(q, "u1", top_k=2)
    assert [h[0] for h in hits] == ["a", "b"]


def test_sqlite_index_namespace_isolation():
    idx = SqliteVectorIndex(":memory:")
    idx.upsert("a", "alice", "location", [1.0, 0.0])
    idx.upsert("b", "bob", "location", [0.0, 1.0])
    alice = idx.search([1.0, 0.0], "alice", top_k=5)
    bob = idx.search([1.0, 0.0], "bob", top_k=5)
    assert alice[0][0] == "a"
    assert bob[0][0] == "b"
    idx.close()


def test_cosine_similarity_clamps_negative():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == 0.0


# ── MemoryLayer integration ───────────────────────────────────────────────────

def test_retrieve_with_brute_index_matches_full_scan():
    with _layer_with_index("brute") as mem:
        mem.write("User lives in Berlin", domain="location")
        mem.write("User prefers concise answers", domain="core_preference")
        mem.write("User is stressed this week", domain="emotional_context")

        indexed = mem.retrieve("concise communication", top_k=2)
        mem._vector_index = None
        full = mem.retrieve("concise communication", top_k=2)

        assert [i.content for i in indexed.items] == [i.content for i in full.items]


def test_retrieve_with_sqlite_index_matches_full_scan():
    with _layer_with_index("sqlite") as mem:
        mem.write("User lives in Berlin", domain="location")
        mem.write("User lives in Paris now", domain="location")
        mem.write("building the billing service", domain="current_project")

        indexed = mem.retrieve("where does user live", top_k=1)
        mem._vector_index = None
        full = mem.retrieve("where does user live", top_k=1)

        assert indexed.items[0].content == full.items[0].content


def test_supersede_removes_old_vector():
    with _layer_with_index("brute") as mem:
        mem.write("User lives in Berlin", domain="location")
        mem.observe(
            "User lives in Paris now",
            domain="location",
            mismatch_magnitude=0.9,
            source="explicit_statement",
        )
        active = mem._active(domain="location")
        assert len(active) == 1
        assert "Paris" in active[0].content
        hits = mem._vector_index.search(_toy_embed("Berlin"), "test", top_k=5)
        for item_id, _ in hits:
            item = mem._store.get(item_id)
            assert item is None or "Berlin" not in item.content


def test_remove_and_clear_drop_vectors():
    with _layer_with_index("sqlite") as mem:
        r = mem.write("User lives in Berlin", domain="location")
        mem.remove(r.item.id)
        hits = mem._vector_index.search(_toy_embed("Berlin"), "test", top_k=5)
        assert hits == []

        mem.write("User lives in Paris", domain="location")
        mem.clear()
        hits = mem._vector_index.search(_toy_embed("Paris"), "test", top_k=5)
        assert hits == []


def test_vector_index_off_preserves_behavior():
    with MemoryLayer(":memory:", vector_index="off") as mem:
        mem.remember("I live in Berlin")
        res = mem.remember("I live in Paris")
        assert res.action in ("audited", "logged_mismatch")
        assert len(mem._active(domain="location")) == 1


if __name__ == "__main__":
    tests = [
        test_brute_force_search_orders_by_similarity,
        test_sqlite_index_namespace_isolation,
        test_cosine_similarity_clamps_negative,
        test_retrieve_with_brute_index_matches_full_scan,
        test_retrieve_with_sqlite_index_matches_full_scan,
        test_supersede_removes_old_vector,
        test_remove_and_clear_drop_vectors,
        test_vector_index_off_preserves_behavior,
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
