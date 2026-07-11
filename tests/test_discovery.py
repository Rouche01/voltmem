"""Tests for domain auto-discovery (Tier 1)."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voltmem import (  # noqa: E402
    MemoryLayer,
    DomainStats,
    VolatilityTracker,
    blend_volatility,
    update_volatility_ema,
    VOL_DRIFT_MAX,
    DOMAIN_VOLATILITY,
)
from voltmem.domains import MemoryItem  # noqa: E402
from voltmem.scoring import should_escalate  # noqa: E402


def test_domain_stats_empirical_volatility_rises_with_supersedes():
    stats = DomainStats(domain="location")
    for _ in range(5):
        stats.record("audited", mismatch=0.8)
    assert stats.empirical_volatility > 0.4


def test_blend_volatility_requires_min_observations():
    prior = 0.08
    assert blend_volatility(prior, 0.9, n_observations=1) == prior
    blended = blend_volatility(prior, 0.9, n_observations=10)
    assert blended > prior


def test_prior_anchored_ema_clamps_drift():
    item = MemoryItem(
        id="x",
        content="pref",
        domain="personality_trait",
        source="weak_inference",
        volatility_ema=-1.0,
        created_at=0.0,
        last_confirmed_at=0.0,
    )
    prior = DOMAIN_VOLATILITY["personality_trait"]
    updated = update_volatility_ema(item, observed_mismatch=0.9, source="weak_inference")
    assert updated <= prior + VOL_DRIFT_MAX + 1e-9
    assert updated >= prior - VOL_DRIFT_MAX - 1e-9


def test_auto_discover_records_observations():
    with MemoryLayer(":memory:", auto_discover=True) as mem:
        mem.write("I live in Berlin", domain="location")
        mem.observe(
            "I moved to Paris",
            domain="location",
            mismatch_magnitude=0.85,
            source="explicit_statement",
        )
        summary = mem.summary()
        assert summary["auto_discover"] is True
        loc = summary["domain_discovery"]["location"]
        assert loc["n_supersedes"] >= 1
        assert loc["empirical"] > 0.0


def test_auto_discover_persists_across_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "discover.db")
        with MemoryLayer(db, namespace="alice", auto_discover=True) as mem:
            mem.write("I prefer concise answers", domain="core_preference")
            for _ in range(4):
                mem.observe(
                    "I prefer concise answers",
                    domain="core_preference",
                    mismatch_magnitude=0.05,
                    source="explicit_statement",
                )
        with MemoryLayer(db, namespace="alice", auto_discover=True) as mem2:
            stats = mem2.summary()["domain_discovery"]["core_preference"]
            assert stats["n_confirms"] >= 4


def test_auto_discover_off_by_default():
    with MemoryLayer(":memory:") as mem:
        assert mem.auto_discover is False
        assert mem._tracker is None
        mem.write("I live in Berlin", domain="location")
        summary = mem.summary()
        assert summary["domain_discovery"] == {}


def test_weak_inference_does_not_supersede_stable_with_auto_discover():
    """Repeated weak contradictions should not corrupt a stable domain."""
    with MemoryLayer(":memory:", auto_discover=True) as mem:
        mem.write("User is a careful planner", domain="personality_trait")
        for _ in range(8):
            res = mem.observe(
                "User did something out of character",
                domain="personality_trait",
                mismatch_magnitude=0.6,
                source="weak_inference",
            )
        assert res.action == "logged_mismatch"
        items = mem._active(domain="personality_trait")
        assert len(items) == 1
        assert "careful planner" in items[0].content


def test_create_memory_auto_discover_flag():
    from voltmem import create_memory

    with create_memory(":memory:", auto_discover=True, embeddings=False) as mem:
        assert mem.layer.auto_discover is True
        row = mem.add("I live in Berlin")
        assert row["action"] == "inserted"


if __name__ == "__main__":
    tests = [
        test_domain_stats_empirical_volatility_rises_with_supersedes,
        test_blend_volatility_requires_min_observations,
        test_prior_anchored_ema_clamps_drift,
        test_auto_discover_records_observations,
        test_auto_discover_persists_across_sessions,
        test_auto_discover_off_by_default,
        test_weak_inference_does_not_supersede_stable_with_auto_discover,
        test_create_memory_auto_discover_flag,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
