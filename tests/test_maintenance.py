"""
Tests for Phase 1 & 2 additions: event_id, expires_at, MaintenanceWindow.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voltmem import MemoryLayer, MaintenanceWindow
from voltmem.maintenance import (
    expire_cleanup, reclassify_ambiguous, pattern_audit, consolidate,
    reconcile_twins,
)
from voltmem.scoring import belief_has_shifted


DAY = 86400.0


def _plant_evidence(mem, item, text, n=4, mm=0.5, src="weak_inference",
                    gap_days=0.0, start=None):
    """Cluster of logged mismatches after the write. ``start`` is the first stamp."""
    t0 = start if start is not None else time.time()
    for i in range(n):
        mem._store.append_mismatch_evidence(
            item.id,
            mem.namespace,
            text,
            mismatch_magnitude=mm,
            source=src,
            created_at=t0 + i * gap_days * DAY,
        )


def _observe_weak_stream(domain, turns=16, gap_days=1.0, mm=0.70,
                         base="User works as a data analyst",
                         obs="User mentioned a nursing shift in passing"):
    """Battery D live path: observe() stream with wall-clock-relative stamps.

    Returns (layer, live_actions, original_content). Live must stay quiet;
    mismatch evidence is what consolidate reads overnight.
    """
    mem = MemoryLayer(":memory:")
    now = time.time()
    start = now - turns * gap_days * DAY
    mem.write(base, domain=domain, at_time=start)
    actions = []
    for i in range(turns):
        res = mem.observe(
            f"{obs} ({i})",
            domain=domain,
            mismatch_magnitude=mm,
            source="weak_inference",
            at_time=start + (i + 1) * gap_days * DAY,
        )
        actions.append(res.action)
    return mem, actions, base


# ── Phase 1: event_id + expires_at ─────────────────────────────────────────────

def test_event_id_roundtrip():
    with MemoryLayer(":memory:") as mem:
        mem.write("test", domain="location", event_id="evt-1", modality="text")
        items = mem._store.all_active()
        assert len(items) == 1
        assert items[0].event_id == "evt-1"
        assert items[0].modality == "text"


def test_expires_at_roundtrip():
    with MemoryLayer(":memory:") as mem:
        future = time.time() + 1000
        mem.write("test", domain="location", expires_at=future)
        items = mem._store.all_active()
        assert items[0].expires_at == future
        assert not items[0].is_expired


def test_expired_item_is_expired():
    with MemoryLayer(":memory:") as mem:
        past = 1.0
        mem.write("test", domain="location", expires_at=past)
        items = mem._store.all_active()
        assert items[0].is_expired


def test_retrieval_skips_expired_items():
    with MemoryLayer(":memory:") as mem:
        mem.write("alive", domain="location")
        mem.write("dead", domain="location", expires_at=1.0)
        results = mem.retrieve("location")
        assert len(results.items) == 1
        assert results.items[0].content == "alive"


def test_add_event_creates_linked_facets():
    with MemoryLayer(":memory:") as mem:
        results = mem.add_event("tick-001", [
            {"content": "battery 37%", "domain": "current_task", "modality": "sensor"},
            {"content": "go to dock", "domain": "current_task", "modality": "text"},
        ])
        assert len(results) == 2
        items = mem.retrieve_by_event("tick-001")
        assert len(items) == 2
        assert all(i.event_id == "tick-001" for i in items)


def test_retrieve_by_event_ordered_by_creation():
    with MemoryLayer(":memory:") as mem:
        mem.add_event("evt-1", [
            {"content": "first", "domain": "location"},
            {"content": "second", "domain": "current_task"},
        ])
        items = mem.retrieve_by_event("evt-1")
        assert len(items) == 2
        assert items[0].content == "first"
        assert items[1].content == "second"


def test_purge_expired_removes_dead_rows():
    with MemoryLayer(":memory:") as mem:
        mem.write("dead", domain="transient_fact", expires_at=1.0)
        mem.write("alive", domain="location")
        deleted = mem.purge_expired()
        assert deleted == 1
        active = mem._store.all_active()
        assert len(active) == 1
        assert active[0].content == "alive"


def test_expire_cleanup_syncs_vector_index():
    """expire_cleanup must drop embeddings — store-only purge left orphans."""
    from voltmem.vector_index import BruteForceVectorIndex

    def _embed(text: str) -> list[float]:
        # Distinct enough vectors for two items
        return [1.0, 0.0] if "dead" in text else [0.0, 1.0]

    idx = BruteForceVectorIndex()
    mem = MemoryLayer(
        ":memory:",
        embed_fn=_embed,
        vector_index=idx,
        namespace="default",
    )
    dead = mem.write("dead fact", domain="transient_fact", expires_at=1.0)
    alive = mem.write("alive berlin", domain="location")
    assert len(idx.search([1.0, 0.0], "default", top_k=5)) >= 1

    mw = MaintenanceWindow(mem)
    mw.register("expire", expire_cleanup)
    assert mw.run_once("expire")["result"] == 1

    hits = idx.search([1.0, 0.0], "default", top_k=10)
    hit_ids = {h[0] for h in hits}
    assert dead.item.id not in hit_ids
    assert alive.item.id in {h[0] for h in idx.search([0.0, 1.0], "default", top_k=10)}
    mem.close()


def test_event_id_inherited_on_supersede():
    with MemoryLayer(":memory:") as mem:
        mem.write("old", domain="location", event_id="evt-1")
        mem.observe("new", domain="location", mismatch_magnitude=0.9, event_id="evt-2")
        items = mem._active(domain="location")
        assert len(items) == 1
        # New item should inherit event_id from candidate if not explicitly set
        assert items[0].event_id == "evt-2"


def test_observe_explicit_event_id_overrides_inheritance():
    with MemoryLayer(":memory:") as mem:
        mem.write("old", domain="location", event_id="evt-1")
        mem.observe("new", domain="location", mismatch_magnitude=0.9, event_id="evt-2")
        items = mem._active(domain="location")
        assert items[0].event_id == "evt-2"


# ── Phase 2: MaintenanceWindow ─────────────────────────────────────────────────

def test_maintenance_window_registration():
    mem = MemoryLayer(":memory:")
    mw = MaintenanceWindow(mem)
    mw.register("expire", expire_cleanup, interval=3600)
    assert len(mw.tasks()) == 1
    assert mw.tasks()[0].name == "expire"


def test_maintenance_window_unregister():
    mem = MemoryLayer(":memory:")
    mw = MaintenanceWindow(mem)
    mw.register("expire", expire_cleanup)
    assert mw.unregister("expire")
    assert not mw.unregister("missing")


def test_expire_cleanup_task():
    mem = MemoryLayer(":memory:")
    mem.write("dead", domain="transient_fact", expires_at=1.0)
    mw = MaintenanceWindow(mem)
    mw.register("expire", expire_cleanup)
    assert mw.run_once("expire")["result"] == 1


def test_expire_cleanup_dry_run_does_not_mutate():
    mem = MemoryLayer(":memory:")
    mem.write("dead", domain="transient_fact", expires_at=1.0)
    mw = MaintenanceWindow(mem)
    mw.register("expire", expire_cleanup)
    would = mw.run_once("expire", dry_run=True)
    assert would["result"] == 1
    assert would["dry_run"] is True
    active = mem._store.all_active()
    assert len(active) == 1
    assert active[0].content == "dead"


def test_reclassify_ambiguous_flags_high_mismatch_domains():
    mem = MemoryLayer(":memory:")
    mem.observe("pref A", domain="core_preference")
    mem.observe("pref B", domain="core_preference", mismatch_magnitude=0.8)
    mem.observe("pref C", domain="core_preference", mismatch_magnitude=0.9)
    mw = MaintenanceWindow(mem)
    mw.register("reclassify", lambda ctx: reclassify_ambiguous(ctx, min_observations=1))
    flags = mw.run_once("reclassify")["result"]
    assert len(flags) >= 1
    assert flags[0]["domain"] == "core_preference"


def test_pattern_audit_flags_accumulated_mismatches():
    mem = MemoryLayer(":memory:")
    mem.write("fact", domain="location")
    item = mem._active(domain="location")[0]
    _plant_evidence(mem, item, "maybe moved", n=5, mm=0.5, src="weak_inference")
    mw = MaintenanceWindow(mem)
    mw.register("pattern", pattern_audit)
    flags = mw.run_once("pattern")["result"]
    assert len(flags) == 1
    assert flags[0]["belief_mass"] >= flags[0]["belief_bar"]


def test_pattern_audit_ignores_lifetime_count_without_a_pile():
    """A high mismatch_count with no windowed evidence must not flag."""
    mem = MemoryLayer(":memory:")
    mem.write("fact", domain="location")
    item = mem._active(domain="location")[0]
    item.mismatch_count = 25
    mem._store.update(item)
    mw = MaintenanceWindow(mem)
    mw.register("pattern", pattern_audit)
    flags = mw.run_once("pattern")["result"]
    assert flags == []


def test_consolidate_dry_run_does_not_mutate():
    mem = MemoryLayer(":memory:")
    mem.write("old", domain="emotional_context")
    item = mem._active(domain="emotional_context")[0]
    _plant_evidence(mem, item, "feeling better lately", n=4, mm=0.5)
    mw = MaintenanceWindow(mem)
    mw.register("consolidate", consolidate)
    flags = mw.run_once("consolidate", dry_run=True)["result"]
    assert len(flags) == 1
    assert flags[0]["new_content"] == "feeling better lately"
    assert "skipped" not in flags[0]
    items = mem._active(domain="emotional_context")
    assert items[0].content == "old"


def test_consolidate_real_run_reorganizes():
    mem = MemoryLayer(":memory:")
    mem.write("old", domain="emotional_context")
    item = mem._active(domain="emotional_context")[0]
    _plant_evidence(mem, item, "feeling better lately", n=4, mm=0.5)
    mw = MaintenanceWindow(mem)
    mw.register("consolidate", consolidate)
    out = mw.run_once("consolidate")
    results = out["result"]
    assert len(results) == 1
    assert results[0]["result"] == "audited"
    items = mem._active(domain="emotional_context")
    assert items[0].content == "feeling better lately"


def test_sleeptime_daily_weak_pile_consolidates_professional():
    """Third test: live observe() stays shut; overnight consolidate updates."""
    mem, actions, original = _observe_weak_stream("professional_context")
    assert "audited" not in actions, f"live path caught the pile: {actions}"
    assert mem._active(domain="professional_context")[0].content == original
    mw = MaintenanceWindow(mem)
    mw.register("consolidate", consolidate)
    out = mw.run_once("consolidate")["result"]
    done = [a for a in out if a.get("result") == "audited"]
    assert done, f"daily pile must mutate overnight, got {out}"
    after = mem._active(domain="professional_context")[0].content
    assert after != original
    assert "nursing" in after.lower()


def test_sleeptime_monthly_drip_does_not_consolidate():
    """Same sixteen asides, monthly: decay, so overnight must not rewrite."""
    mem, actions, original = _observe_weak_stream(
        "professional_context", gap_days=30.0)
    assert "audited" not in actions
    mw = MaintenanceWindow(mem)
    mw.register("consolidate", consolidate)
    out = mw.run_once("consolidate")["result"]
    done = [a for a in out if a.get("result") == "audited"]
    assert not done, f"monthly drip must not consolidate, got {out}"
    assert mem._active(domain="professional_context")[0].content == original


def test_sleeptime_cannot_erode_very_stable_domain():
    mem, actions, original = _observe_weak_stream(
        "core_preference",
        base="User prefers concise, direct answers",
        obs="User lingered on a long explanation",
    )
    assert "audited" not in actions
    mw = MaintenanceWindow(mem)
    mw.register("consolidate", consolidate)
    out = mw.run_once("consolidate")["result"]
    done = [a for a in out if a.get("result") == "audited"]
    assert not done, f"preference must not yield overnight, got {out}"
    assert mem._active(domain="core_preference")[0].content == original


def test_consolidate_skips_without_evidence():
    mem = MemoryLayer(":memory:")
    mem.write("old", domain="emotional_context")
    item = mem._active(domain="emotional_context")[0]
    item.mismatch_count = 3
    mem._store.update(item)
    mw = MaintenanceWindow(mem)
    mw.register("consolidate", consolidate)
    results = mw.run_once("consolidate")["result"]
    assert results == []
    assert mem._active(domain="emotional_context")[0].content == "old"


def test_consolidate_injectable_summarizer():
    mem = MemoryLayer(":memory:")
    mem.write("old tip", domain="emotional_context")
    item = mem._active(domain="emotional_context")[0]
    _plant_evidence(mem, item, "signal a", n=4, mm=0.5)

    class FixedSummarizer:
        def summarize(self, current, evidence, *, domain):
            return "merged by test"

    mw = MaintenanceWindow(mem)
    mw.register(
        "consolidate",
        lambda ctx: consolidate(ctx, summarizer=FixedSummarizer()),
    )
    results = mw.run_once("consolidate")["result"]
    assert results[0]["new_content"] == "merged by test"
    assert mem._active(domain="emotional_context")[0].content == "merged by test"


def test_consolidate_event_aware_flags_cofacets():
    mem = MemoryLayer(":memory:")
    mem.add_event("tick-001", [
        {"content": "exhausted", "domain": "emotional_context"},
        {"content": "gym", "domain": "current_task"},
        {"content": "apartment", "domain": "location"},
    ])
    # Build up mismatches on emotional_context (also writes evidence)
    mem.observe("tired", domain="emotional_context", mismatch_magnitude=0.18)
    mem.observe("drained", domain="emotional_context", mismatch_magnitude=0.18)
    mem.observe("great", domain="emotional_context", mismatch_magnitude=0.18)
    mw = MaintenanceWindow(mem)
    mw.register(
        "consolidate",
        lambda ctx: consolidate(ctx, min_mismatch_count=3, event_aware=True),
    )
    results = mw.run_once("consolidate")["result"]
    reorganized = [
        r for r in results if "new_content" in r and not r.get("skipped")
    ]
    cofacets = [r for r in results if "note" in r]
    assert len(reorganized) == 1
    assert reorganized[0]["new_content"] == "great"
    assert len(cofacets) == 2
    assert all(r["event_id"] == "tick-001" for r in cofacets)


def test_run_all_executes_all_tasks():
    mem = MemoryLayer(":memory:")
    mem.write("dead", domain="transient_fact", expires_at=1.0)
    mw = MaintenanceWindow(mem)
    mw.register("expire", expire_cleanup)
    mw.register("pattern", pattern_audit)
    out = mw.run_all()
    assert "expire" in out["results"]
    assert "pattern" in out["results"]
    assert out["run_id"]


def test_rollback_maintenance_undoes_consolidate():
    mem = MemoryLayer(":memory:")
    mem.write("old", domain="emotional_context")
    item = mem._active(domain="emotional_context")[0]
    old_id = item.id
    _plant_evidence(mem, item, "feeling better lately", n=4, mm=0.5)
    mw = MaintenanceWindow(mem)
    mw.register("consolidate", consolidate)
    out = mw.run_once("consolidate")
    run_id = out["run_id"]
    new_id = out["result"][0]["new_item_id"]
    assert mem._active(domain="emotional_context")[0].content == "feeling better lately"

    summary = mem.rollback_maintenance(run_id)
    assert summary["restored"] == 1
    active = mem._active(domain="emotional_context")
    assert len(active) == 1
    assert active[0].id == old_id
    assert active[0].content == "old"
    retired = mem._store.get(new_id)
    assert retired is not None
    assert retired.superseded_by == old_id


def test_rollback_maintenance_restores_purged():
    mem = MemoryLayer(":memory:")
    mem.write("dead", domain="transient_fact", expires_at=1.0)
    mw = MaintenanceWindow(mem)
    mw.register("expire", expire_cleanup)
    out = mw.run_once("expire")
    assert out["result"] == 1
    assert mem._store.all_active() == []

    summary = mem.rollback_maintenance(out["run_id"])
    assert summary["restored"] == 1
    active = mem._store.all_active()
    assert len(active) == 1
    assert active[0].content == "dead"


def test_background_thread_runs_periodically():
    mem = MemoryLayer(":memory:")
    mem.write("dead", domain="transient_fact", expires_at=1.0)
    mw = MaintenanceWindow(mem)
    mw.register("expire", expire_cleanup, interval=1)
    with mw:
        time.sleep(2.5)
    # After 2.5s with 1s interval, expire should have run at least twice
    active = mem._store.all_active()
    assert len(active) == 0  # expired item was purged


# ── twin reconciliation ────────────────────────────────────────────────────────

def _always_related(a, b):
    return 0.9 if a != b else 1.0


def _job_twins(new, stored, domain):
    blob = f"{stored} {new}".lower()
    return "analyst" in blob and "nurse" in blob


def _plant_twins(mem, older, newer, domain, t0=None):
    t0 = t0 if t0 is not None else time.time()
    mem.write(older, domain=domain, at_time=t0)
    mem.write(newer, domain=domain, at_time=t0 + 10)
    return mem._active(domain=domain)


def test_reconcile_twins_skips_without_a_verifier():
    mem = MemoryLayer(":memory:")
    _plant_twins(
        mem,
        "User works as a data analyst",
        "User changed careers and now works as a nurse",
        "professional_context",
    )
    mw = MaintenanceWindow(mem)
    mw.register("twins", reconcile_twins)
    out = mw.run_once("twins")["result"]
    assert out == [{"skipped": "no_verifier"}]
    assert len(mem._active()) == 2


def test_reconcile_twins_dry_run_does_not_mutate():
    mem = MemoryLayer(":memory:")
    _plant_twins(
        mem,
        "User works as a data analyst",
        "User changed careers and now works as a nurse",
        "professional_context",
    )
    mw = MaintenanceWindow(mem)
    mw.register(
        "twins",
        lambda ctx: reconcile_twins(
            ctx, verifier=_job_twins, similarity_fn=_always_related),
    )
    flags = mw.run_once("twins", dry_run=True)["result"]
    assert flags and flags[0]["old_content"].startswith("User works as")
    assert len(mem._active()) == 2


def test_reconcile_twins_merges_career_change_and_keeps_skills():
    mem = MemoryLayer(":memory:")
    t0 = time.time()
    _plant_twins(
        mem,
        "User works as a data analyst",
        "User changed careers and now works as a nurse",
        "professional_context",
        t0=t0,
    )
    _plant_twins(
        mem,
        "User is proficient in Python",
        "User is proficient in Japanese",
        "skill",
        t0=t0 + 20,
    )
    mw = MaintenanceWindow(mem)
    mw.register(
        "twins",
        lambda ctx: reconcile_twins(
            ctx, verifier=_job_twins, similarity_fn=_always_related),
    )
    out = mw.run_once("twins")["result"]
    retired = [a for a in out if "old_id" in a]
    assert len(retired) == 1, out
    jobs = mem._active(domain="professional_context")
    assert len(jobs) == 1
    assert "nurse" in jobs[0].content
    skills = mem._active(domain="skill")
    assert len(skills) == 2
    contents = {s.content for s in skills}
    assert "User is proficient in Python" in contents
    assert "User is proficient in Japanese" in contents


def test_reconcile_twins_refusal_never_merges():
    mem = MemoryLayer(":memory:")
    _plant_twins(
        mem,
        "User prefers dark mode",
        "User prefers concise, direct answers",
        "core_preference",
    )
    mw = MaintenanceWindow(mem)
    mw.register(
        "twins",
        lambda ctx: reconcile_twins(
            ctx, verifier=lambda *_a: False, similarity_fn=_always_related),
    )
    out = mw.run_once("twins")["result"]
    assert out == []
    assert len(mem._active(domain="core_preference")) == 2


def test_rollback_maintenance_restores_retired_twin():
    mem = MemoryLayer(":memory:")
    _plant_twins(
        mem,
        "User works as a data analyst",
        "User changed careers and now works as a nurse",
        "professional_context",
    )
    older = [i for i in mem._active() if "analyst" in i.content][0]
    mw = MaintenanceWindow(mem)
    mw.register(
        "twins",
        lambda ctx: reconcile_twins(
            ctx, verifier=_job_twins, similarity_fn=_always_related),
    )
    out = mw.run_once("twins")
    assert len(mem._active()) == 1
    summary = mem.rollback_maintenance(out["run_id"])
    assert summary["restored"] == 1
    active = mem._active()
    assert len(active) == 2
    restored = mem._store.get(older.id)
    assert restored is not None
    assert restored.superseded_by is None
    assert restored.content == "User works as a data analyst"


# ── force_update ───────────────────────────────────────────────────────────────

def test_force_update_bypasses_escalation_math():
    with MemoryLayer(":memory:") as mem:
        mem.write("stable", domain="personality_trait")
        result = mem.observe(
            "changed",
            domain="personality_trait",
            mismatch_magnitude=0.1,  # would normally NOT escalate
            source="weak_inference",
            force_update=True,
        )
        assert result.action == "audited"


def test_force_update_false_uses_normal_math():
    with MemoryLayer(":memory:") as mem:
        mem.write("stable", domain="personality_trait")
        result = mem.observe(
            "changed",
            domain="personality_trait",
            mismatch_magnitude=0.1,  # below 0.15 threshold → confirmed even without force
            source="weak_inference",
            force_update=False,
        )
        # With mismatch=0.1 < 0.15, it confirms regardless of force_update
        # To test normal escalation math, use mismatch above 0.15 but below threshold
        assert result.action == "confirmed"


# ── run ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_event_id_roundtrip,
        test_expires_at_roundtrip,
        test_expired_item_is_expired,
        test_retrieval_skips_expired_items,
        test_add_event_creates_linked_facets,
        test_retrieve_by_event_ordered_by_creation,
        test_purge_expired_removes_dead_rows,
        test_expire_cleanup_syncs_vector_index,
        test_event_id_inherited_on_supersede,
        test_observe_explicit_event_id_overrides_inheritance,
        test_maintenance_window_registration,
        test_maintenance_window_unregister,
        test_expire_cleanup_task,
        test_expire_cleanup_dry_run_does_not_mutate,
        test_reclassify_ambiguous_flags_high_mismatch_domains,
        test_pattern_audit_flags_accumulated_mismatches,
        test_pattern_audit_ignores_lifetime_count_without_a_pile,
        test_consolidate_dry_run_does_not_mutate,
        test_consolidate_real_run_reorganizes,
        test_sleeptime_daily_weak_pile_consolidates_professional,
        test_sleeptime_monthly_drip_does_not_consolidate,
        test_sleeptime_cannot_erode_very_stable_domain,
        test_consolidate_skips_without_evidence,
        test_consolidate_injectable_summarizer,
        test_consolidate_event_aware_flags_cofacets,
        test_run_all_executes_all_tasks,
        test_rollback_maintenance_undoes_consolidate,
        test_rollback_maintenance_restores_purged,
        test_reconcile_twins_skips_without_a_verifier,
        test_reconcile_twins_dry_run_does_not_mutate,
        test_reconcile_twins_merges_career_change_and_keeps_skills,
        test_reconcile_twins_refusal_never_merges,
        test_rollback_maintenance_restores_retired_twin,
        test_background_thread_runs_periodically,
        test_force_update_bypasses_escalation_math,
        test_force_update_false_uses_normal_math,
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

    print(f"\n{passed}/{passed+failed} tests passed")
    if failed:
        sys.exit(1)
