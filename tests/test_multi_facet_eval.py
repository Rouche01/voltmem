"""
Synthetic multi-facet eval (Problem 4).

Asserts the OPEN_PROBLEMS acceptance criteria:
  linked retrieval + independent stale@k / audit behavior

Sketch: one robot tick → stable map + volatile battery + task, shared event_id.
Each facet keeps its own V_d — linkage must not force a shared forget/audit rate.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voltmem import DomainRegistry, MemoryLayer, retrieval_score
from voltmem.domains import MemoryItem

EVENT_ID = "tick-50ms-001"

FACETS = [
    {
        "content": "corridor map patch A12",
        "domain": "spatial_map",
        "modality": "structured",
    },
    {
        "content": "battery 37%",
        "domain": "power_state",
        "modality": "sensor",
    },
    {
        "content": "go to charging dock",
        "domain": "current_task",
        "modality": "text",
    },
]


def _robot_domains() -> DomainRegistry:
    """Match OPEN_PROBLEMS sketch: stable map vs volatile power."""
    return (
        DomainRegistry()
        .register("spatial_map", 0.10)
        .register("power_state", 0.85)
    )


def _with_robot_domains(fn):
    restore = _robot_domains().install()
    try:
        return fn()
    finally:
        restore()


# ── 1. Linked retrieve ────────────────────────────────────────────────────────

def test_multi_facet_linked_retrieve():
    def run():
        with MemoryLayer(":memory:") as mem:
            results = mem.add_event(EVENT_ID, FACETS)
            assert len(results) == 3
            assert all(r.action == "inserted" for r in results)

            linked = mem.retrieve_by_event(EVENT_ID)
            assert len(linked) == 3
            assert {i.event_id for i in linked} == {EVENT_ID}
            assert [i.domain for i in linked] == [
                "spatial_map",
                "power_state",
                "current_task",
            ]
            assert [i.modality for i in linked] == [
                "structured",
                "sensor",
                "text",
            ]
            # Creation order preserved
            assert linked[0].created_at <= linked[1].created_at <= linked[2].created_at

    _with_robot_domains(run)


# ── 2. Independent audit ──────────────────────────────────────────────────────

def test_multi_facet_independent_audit():
    """Volatile facet audits; stable co-facet retains under the same tick."""

    def run():
        with MemoryLayer(":memory:") as mem:
            mem.add_event(EVENT_ID, FACETS)

            batt = mem.observe(
                "battery 12%",
                domain="power_state",
                mismatch_magnitude=0.85,
                source="explicit_statement",
            )
            assert batt.action == "audited", (
                f"volatile power_state should audit, got {batt.action}"
            )
            assert batt.item.event_id == EVENT_ID
            assert "12%" in batt.item.content

            mapping = mem.observe(
                "corridor map patch B99",
                domain="spatial_map",
                mismatch_magnitude=0.85,
                source="explicit_statement",
            )
            assert mapping.action == "logged_mismatch", (
                f"stable spatial_map should retain, got {mapping.action}"
            )

            active_map = [i for i in mem._active(domain="spatial_map") if i.event_id == EVENT_ID]
            assert len(active_map) == 1
            assert active_map[0].content == "corridor map patch A12"

            active_batt = [i for i in mem._active(domain="power_state") if i.event_id == EVENT_ID]
            assert len(active_batt) == 1
            assert "12%" in active_batt[0].content

            # Event still reassembles (history + tip)
            linked = mem.retrieve_by_event(EVENT_ID)
            assert any(i.domain == "spatial_map" and i.is_active for i in linked)
            assert any(
                i.domain == "power_state" and i.is_active and "12%" in i.content
                for i in linked
            )
            assert any(i.domain == "current_task" and i.is_active for i in linked)

    _with_robot_domains(run)


# ── 3. Independent stale@k ────────────────────────────────────────────────────

def test_multi_facet_independent_stale_at_k_scoring():
    """Same age + sim: volatile facet ranks worse than stable co-facet."""

    def run():
        now = time.time()
        age = now - 30 * 86400
        stable = MemoryItem(
            id="map",
            content="corridor map patch A12",
            domain="spatial_map",
            source="explicit_statement",
            event_id=EVENT_ID,
            created_at=age,
            last_confirmed_at=age,
        )
        volatile = MemoryItem(
            id="batt",
            content="battery 37%",
            domain="power_state",
            source="explicit_statement",
            event_id=EVENT_ID,
            created_at=age,
            last_confirmed_at=age,
        )
        score_map = retrieval_score(stable, semantic_similarity=1.0, now=now)
        score_batt = retrieval_score(volatile, semantic_similarity=1.0, now=now)
        assert score_batt < score_map, (
            f"stale@k independence failed: power={score_batt:.3f} map={score_map:.3f}"
        )
        # Stable should barely be crushed by age; volatile should take a real hit
        assert score_map > 0.85
        assert score_batt < 0.75

    _with_robot_domains(run)


def test_multi_facet_independent_stale_at_1_retrieve():
    """Haystack-style: old volatile tip loses @1 to a fresh same-domain fact."""

    def run():
        def sim(a: str, b: str) -> float:
            # Deterministic: battery queries prefer battery strings
            al, bl = a.lower(), b.lower()
            if "battery" in al and "battery" in bl:
                return 0.95
            if "map" in al and "map" in bl:
                return 0.95
            if "dock" in al and "dock" in bl:
                return 0.9
            return 0.2

        with MemoryLayer(":memory:", similarity_fn=sim, vector_index="off") as mem:
            old = time.time() - 90 * 86400
            mem.add_event(EVENT_ID, FACETS, at_time=old)
            # Back-date confirmations so age is real (write sets last_confirmed=now)
            for item in mem._active():
                item.last_confirmed_at = old
                item.created_at = old
                mem._store.update(item)

            fresh = mem.write(
                "battery 88% after charge",
                domain="power_state",
                event_id="tick-charge-002",
            )
            assert fresh.action == "inserted"

            result = mem.retrieve("battery level", top_k=3)
            assert result.items, "expected retrieval hits"
            top = result.items[0]
            assert top.id == fresh.item.id, (
                f"stale@1: expected fresh battery tip, got {top.content!r}"
            )
            # Stale co-event battery must not win @1
            assert "37%" not in top.content

    _with_robot_domains(run)


# ── 4. Causal smoke: swap priors invert stale ranking ─────────────────────────

def test_multi_facet_swap_priors_invert_stale_ranking():
    """Under swapped V_d, the 'map' facet becomes the stale trap."""
    from contextlib import contextmanager
    import voltmem.domains as dom

    @contextmanager
    def swapped():
        restore = _robot_domains().install()
        try:
            # Invert the two custom priors after install
            v_map = dom.DOMAIN_VOLATILITY["spatial_map"]
            v_pwr = dom.DOMAIN_VOLATILITY["power_state"]
            dom.DOMAIN_VOLATILITY["spatial_map"] = v_pwr
            dom.DOMAIN_VOLATILITY["power_state"] = v_map
            yield
        finally:
            restore()

    now = time.time()
    age = now - 30 * 86400
    with swapped():
        stable_named = MemoryItem(
            id="map",
            content="corridor map patch A12",
            domain="spatial_map",
            source="explicit_statement",
            event_id=EVENT_ID,
            created_at=age,
            last_confirmed_at=age,
        )
        volatile_named = MemoryItem(
            id="batt",
            content="battery 37%",
            domain="power_state",
            source="explicit_statement",
            event_id=EVENT_ID,
            created_at=age,
            last_confirmed_at=age,
        )
        # After swap, spatial_map is high-V → should score worse when aged
        score_map = retrieval_score(stable_named, 1.0, now=now)
        score_batt = retrieval_score(volatile_named, 1.0, now=now)
        assert score_map < score_batt, (
            f"swap control failed: map={score_map:.3f} batt={score_batt:.3f}"
        )


if __name__ == "__main__":
    tests = [
        test_multi_facet_linked_retrieve,
        test_multi_facet_independent_audit,
        test_multi_facet_independent_stale_at_k_scoring,
        test_multi_facet_independent_stale_at_1_retrieve,
        test_multi_facet_swap_priors_invert_stale_ranking,
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
    print(f"\n{passed}/{passed + failed} multi-facet eval probes passed")
    if failed:
        sys.exit(1)
    print("PASS: linked retrieve + independent audit + independent stale@k")
