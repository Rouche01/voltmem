"""
TTL haystack eval — hard expiry must never surface in retrieval.

OPEN_PROBLEMS acceptance: haystack with expired items → 0% retrieval past
``expires_at``. Soft staleness can down-rank old facts; TTL is a hard cutoff.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voltmem import MemoryLayer

DAY = 86400.0


@dataclass
class TtlSlot:
    query: str
    current: str
    domain: str
    expired_decoys: list[str]
    distractors: list[tuple[str, str]]  # (text, domain)
    # Tokens that mark query↔slot content as highly similar (expired decoys share these)
    match_tokens: tuple[str, ...]


SLOTS: list[TtlSlot] = [
    TtlSlot(
        query="what is the user working on right now",
        current="User is currently building the billing service",
        domain="current_project",
        expired_decoys=[
            "User is currently building the postgres migration",
            "User is currently building the auth refactor",
            "User is currently building the mobile app rewrite",
            "User is currently building the search indexer",
            "User is currently building the onboarding flow",
            "User is currently building the metrics dashboard",
        ],
        distractors=[
            ("User prefers concise bullet points", "core_preference"),
            ("User lives in Berlin", "location"),
            ("User feels stressed this week", "emotional_context"),
        ],
        match_tokens=("building", "currently"),
    ),
    TtlSlot(
        query="how is the user feeling lately",
        current="User is feeling focused and energized this week",
        domain="emotional_context",
        expired_decoys=[
            "User is feeling stressed and overwhelmed this week",
            "User is feeling anxious about deadlines this week",
            "User is feeling tired and burned out this week",
            "User is feeling frustrated with the rollout this week",
        ],
        distractors=[
            ("User is working on the billing service", "current_project"),
            ("User prefers direct communication", "core_preference"),
        ],
        match_tokens=("feeling",),
    ),
    TtlSlot(
        query="where does the user live",
        current="User currently lives in Paris",
        domain="location",
        expired_decoys=[
            "User currently lives in Berlin",
            "User currently lives in London",
            "User currently lives in Madrid",
            "User currently lives in Amsterdam",
        ],
        distractors=[
            ("User prefers dark mode", "core_preference"),
            ("User is feeling calm today", "emotional_context"),
        ],
        match_tokens=("lives", "live"),
    ),
]


def _haystack_sim(a: str, b: str) -> float:
    """High sim for query↔slot content; low for cross-slot distractors."""
    al, bl = a.lower(), b.lower()

    def keyed(q: str, c: str) -> float | None:
        for slot in SLOTS:
            ql = slot.query.lower()
            if q != ql and c != ql:
                continue
            other = c if q == ql else q
            if any(tok in other for tok in slot.match_tokens):
                # Prefer exact current slightly so @1 is stable after TTL filter
                if other == slot.current.lower():
                    return 0.98
                if other in {d.lower() for d in slot.expired_decoys}:
                    return 0.95
                return 0.85
        return None

    hit = keyed(al, bl)
    if hit is not None:
        return hit
    # Weak fallback overlap
    ta = {t for t in al.split() if len(t) > 2}
    tb = {t for t in bl.split() if len(t) > 2}
    if not ta or not tb:
        return 0.05
    return 0.05 + 0.4 * (len(ta & tb) / len(ta | tb))


def _seed_slot(mem: MemoryLayer, slot: TtlSlot, *, now: float) -> str:
    """Write current (live) + expired decoys + live distractors. Returns current id."""
    past = now - 7 * DAY
    for text in slot.expired_decoys:
        mem.write(
            text,
            domain=slot.domain,
            expires_at=past,  # already expired
            at_time=past - DAY,
        )
    for text, domain in slot.distractors:
        mem.write(text, domain=domain, at_time=now - DAY)
    cur = mem.write(
        slot.current,
        domain=slot.domain,
        expires_at=now + 14 * DAY,  # still valid
        at_time=now,
    )
    return cur.item.id


def _expired_in_hits(items, *, now: float) -> list:
    out = []
    for item in items:
        if item.expires_at is not None and now > item.expires_at:
            out.append(item)
    return out


def test_ttl_haystack_zero_expired_in_topk():
    """Across slots: no expired row may appear in top-k (freshness on or off)."""
    now = time.time()
    expired_hits = 0
    total_hits = 0
    current_at_1 = 0

    with MemoryLayer(
        ":memory:", similarity_fn=_haystack_sim, vector_index="off"
    ) as mem:
        current_ids: dict[str, str] = {}
        for slot in SLOTS:
            current_ids[slot.query] = _seed_slot(mem, slot, now=now)

        for slot in SLOTS:
            for use_staleness in (True, False):
                result = mem.retrieve(
                    slot.query, top_k=5, now=now, use_staleness=use_staleness
                )
                total_hits += len(result.items)
                bad = _expired_in_hits(result.items, now=now)
                expired_hits += len(bad)
                assert not bad, (
                    f"expired leaked under use_staleness={use_staleness} "
                    f"query={slot.query!r}: {[i.content for i in bad]}"
                )
                if result.items and result.items[0].id == current_ids[slot.query]:
                    current_at_1 += 1

    assert total_hits > 0
    assert expired_hits == 0, f"expected 0% expired retrieval, got {expired_hits}"
    # With competitive decoys filtered out, current should usually win @1
    assert current_at_1 >= len(SLOTS), (
        f"current@1 too weak: {current_at_1} wins across "
        f"{len(SLOTS) * 2} retrieves (staleness on+off)"
    )


def test_ttl_haystack_expired_would_win_without_cutoff():
    """Sanity: expired decoys are similarity-competitive if not filtered."""
    now = time.time()
    slot = SLOTS[0]
    with MemoryLayer(
        ":memory:", similarity_fn=_haystack_sim, vector_index="off"
    ) as mem:
        _seed_slot(mem, slot, now=now)
        # Raw candidate pool includes expired (store still has them)
        active_including_expired = [
            i for i in mem._store.all_active(namespace=mem.namespace)
            if i.domain == slot.domain
        ]
        expired = [
            i for i in active_including_expired
            if i.expires_at is not None and now > i.expires_at
        ]
        assert len(expired) == len(slot.expired_decoys)

        # Best expired decoy similarity should be high vs the query
        best_expired_sim = max(
            _haystack_sim(slot.query, i.content) for i in expired
        )
        current = next(
            i for i in active_including_expired
            if i.expires_at is not None and now <= i.expires_at
        )
        current_sim = _haystack_sim(slot.query, current.content)
        assert best_expired_sim >= 0.3, "decoys should be competitive on sim"
        assert current_sim >= 0.3

        # retrieve still returns zero expired
        hits = mem.retrieve(slot.query, top_k=10, now=now)
        assert _expired_in_hits(hits.items, now=now) == []
        assert all(i.id != e.id for i in hits.items for e in expired)


def test_ttl_haystack_with_vector_index():
    """ANN candidates may include expired rows; retrieve must still drop them."""
    now = time.time()

    def embed(text: str) -> list[float]:
        # Bag-of-hash dims so related "building the X" strings cluster
        vec = [0.0] * 32
        for tok in text.lower().split():
            vec[hash(tok) % 32] += 1.0
        return vec

    def sim(a: str, b: str) -> float:
        from voltmem.embeddings import _cosine
        return max(0.0, _cosine(embed(a), embed(b)))

    slot = SLOTS[0]
    with MemoryLayer(
        ":memory:",
        similarity_fn=sim,
        embed_fn=embed,
        vector_index="brute",
    ) as mem:
        current_id = _seed_slot(mem, slot, now=now)
        # Confirm index still holds expired ids (not purged yet)
        indexed_ids = {item_id for item_id, _ in mem._vector_index.search(
            embed(slot.query), mem.namespace, top_k=20
        )}
        store_expired = [
            i.id for i in mem._store.all_active(namespace=mem.namespace)
            if i.expires_at is not None and now > i.expires_at
        ]
        assert any(eid in indexed_ids for eid in store_expired), (
            "precondition: expired vectors should still be indexed before purge"
        )

        result = mem.retrieve(slot.query, top_k=5, now=now)
        assert _expired_in_hits(result.items, now=now) == []
        assert result.items[0].id == current_id


def test_ttl_haystack_purge_removes_expired_noise():
    now = time.time()
    slot = SLOTS[1]
    with MemoryLayer(
        ":memory:", similarity_fn=_haystack_sim, vector_index="off"
    ) as mem:
        current_id = _seed_slot(mem, slot, now=now)
        before = len(mem._store.all_active(namespace=mem.namespace))
        deleted = mem.purge_expired(now=now)
        assert deleted == len(slot.expired_decoys)
        after = mem._store.all_active(namespace=mem.namespace)
        assert len(after) == before - deleted
        assert all(
            i.expires_at is None or now <= i.expires_at for i in after
        )
        hits = mem.retrieve(slot.query, top_k=3, now=now)
        assert hits.items[0].id == current_id
        assert _expired_in_hits(hits.items, now=now) == []


if __name__ == "__main__":
    tests = [
        test_ttl_haystack_zero_expired_in_topk,
        test_ttl_haystack_expired_would_win_without_cutoff,
        test_ttl_haystack_with_vector_index,
        test_ttl_haystack_purge_removes_expired_noise,
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
    print(f"\n{passed}/{passed + failed} TTL haystack probes passed")
    if failed:
        sys.exit(1)
    print("PASS: 0% retrieval past expires_at")
