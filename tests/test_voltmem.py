"""
Tests for VoltMem — covering all core equation behaviours.
"""
import math
import sys
import time
sys.path.insert(0, "/home/claude/voltmem")

from voltmem import MemoryLayer, DOMAIN_VOLATILITY
from voltmem.domains import MemoryItem
from voltmem.scoring import (
    escalation_score, staleness, retrieval_score,
    protection_weight, should_escalate,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def make_item(domain="core_preference", rep=1, vol_ema=-1.0):
    now = time.time()
    return MemoryItem(
        id="test-01",
        content="test content",
        domain=domain,
        source="explicit_statement",
        repetition_count=rep,
        volatility_ema=vol_ema,
        created_at=now,
        last_confirmed_at=now,
    )


# ── 1. Protection weight ──────────────────────────────────────────────────────

def test_stable_domain_gets_high_protection():
    stable = make_item(domain="core_preference")    # V_d = 0.08
    volatile = make_item(domain="current_task")     # V_d = 0.90
    assert protection_weight(stable) > protection_weight(volatile), \
        "Stable domain should have higher protection weight than volatile"

def test_protection_weight_clamped():
    item = make_item(domain="personality_trait")    # very low V_d → very high w
    w = protection_weight(item)
    assert w <= 20.0, "Protection weight should be clamped to 20"

# ── 2. Staleness ──────────────────────────────────────────────────────────────

def test_freshly_confirmed_item_low_staleness():
    item = make_item(domain="current_task")         # high V_d
    s = staleness(item)
    assert s < 0.02, f"Fresh item should have very low staleness, got {s:.4f}"

def test_volatile_item_goes_stale_faster_than_stable():
    now = time.time()
    one_week_ago = now - 7 * 86400

    stable = MemoryItem(
        id="s1", content="x", domain="personality_trait",
        source="explicit_statement",
        created_at=one_week_ago, last_confirmed_at=one_week_ago,
    )
    volatile = MemoryItem(
        id="v1", content="x", domain="current_task",
        source="explicit_statement",
        created_at=one_week_ago, last_confirmed_at=one_week_ago,
    )
    assert staleness(volatile, now) > staleness(stable, now), \
        "Volatile domain should go stale faster than stable domain"

# ── 3. Escalation score ───────────────────────────────────────────────────────

def test_high_mismatch_volatile_escalates():
    item = make_item(domain="current_task")         # V_d = 0.90
    escalated = should_escalate(
        item, mismatch_magnitude=0.9, source="explicit_statement")
    assert escalated, "High mismatch on volatile domain should escalate"

def test_low_mismatch_stable_does_not_escalate():
    item = make_item(domain="personality_trait")    # V_d = 0.05
    escalated = should_escalate(
        item, mismatch_magnitude=0.2, source="weak_inference")
    assert not escalated, \
        "Low mismatch on stable domain with weak source should not escalate"

def test_high_repetition_suppresses_escalation():
    """High C raises the denominator in E_t, making escalation harder."""
    low_rep  = make_item(domain="stated_preference", rep=1)
    high_rep = make_item(domain="stated_preference", rep=50)
    E_low,  _ = escalation_score(low_rep,  mismatch_magnitude=0.5)
    E_high, _ = escalation_score(high_rep, mismatch_magnitude=0.5)
    assert E_low > E_high, \
        "Higher repetition count should produce lower escalation score"

def test_threshold_scales_inversely_with_volatility():
    """theta_t = theta_0 * (1/V_d) * L; high V_d → low threshold."""
    stable   = make_item(domain="personality_trait")   # V_d=0.05
    volatile = make_item(domain="current_task")         # V_d=0.90
    _, theta_stable   = escalation_score(stable,   mismatch_magnitude=0.5)
    _, theta_volatile = escalation_score(volatile, mismatch_magnitude=0.5)
    assert theta_stable > theta_volatile, \
        "Stable domain should have higher audit threshold (harder to trigger)"

# ── 4. Retrieval score ────────────────────────────────────────────────────────

def test_stale_volatile_item_ranked_lower_than_fresh():
    now = time.time()
    old = time.time() - 30 * 86400   # 30 days ago

    fresh = MemoryItem(
        id="f1", content="x", domain="current_task",
        source="explicit_statement",
        created_at=now, last_confirmed_at=now,
    )
    stale = MemoryItem(
        id="s1", content="x", domain="current_task",
        source="explicit_statement",
        created_at=old, last_confirmed_at=old,
    )
    score_fresh = retrieval_score(fresh, semantic_similarity=0.8, now=now)
    score_stale = retrieval_score(stale, semantic_similarity=0.8, now=now)
    assert score_fresh > score_stale, \
        "Fresh volatile item should score higher than stale volatile item"

def test_stable_item_age_barely_penalised():
    now = time.time()
    one_year_ago = now - 365 * 86400

    old_stable = MemoryItem(
        id="os1", content="x", domain="personality_trait",
        source="explicit_statement",
        created_at=one_year_ago, last_confirmed_at=one_year_ago,
    )
    score = retrieval_score(old_stable, semantic_similarity=0.8, now=now)
    # personality_trait V_d=0.05; staleness after 1yr ≈ 1-exp(-0.05*365) ≈ 1.0
    # but weight=0.05 so penalty = 0.05 * ~1.0 = 0.05 → score ≈ 0.8*0.95=0.76
    assert score > 0.70, \
        f"Stable item should barely be penalised for age; got {score:.3f}"

# ── 5. MemoryLayer integration ────────────────────────────────────────────────

def test_write_and_retrieve():
    with MemoryLayer(":memory:") as mem:
        mem.write("User prefers concise answers", domain="core_preference")
        results = mem.retrieve("communication style preference")
        assert len(results.items) > 0
        assert any("concise" in item.content for item in results.items)

def test_low_mismatch_confirms_not_supersedes():
    with MemoryLayer(":memory:") as mem:
        r1 = mem.write("User is a software engineer", domain="professional_context")
        r2 = mem.observe(
            "User mentioned working in software again",
            domain="professional_context",
            mismatch_magnitude=0.05,
            source="weak_inference",
        )
        assert r2.action == "confirmed", \
            f"Low mismatch should confirm, got {r2.action}"
        assert r2.item.repetition_count == 2

def test_high_mismatch_volatile_supersedes():
    with MemoryLayer(":memory:") as mem:
        mem.write("User is job hunting", domain="current_project")
        result = mem.observe(
            "User accepted a job offer, no longer job hunting",
            domain="current_project",
            mismatch_magnitude=0.9,
            source="explicit_statement",
        )
        assert result.action == "audited", \
            f"High mismatch on volatile domain should audit/supersede, got {result.action}"
        # old item should be superseded, new item active
        all_items = mem._store.all_active(domain="current_project")
        assert len(all_items) == 1
        assert "accepted" in all_items[0].content or "no longer" in all_items[0].content

def test_high_mismatch_stable_does_not_supersede():
    """
    A highly stable domain (personality_trait) should resist superseding
    even under significant mismatch, because theta_t is very high for it.
    """
    with MemoryLayer(":memory:") as mem:
        mem.write("User is introverted", domain="personality_trait")
        result = mem.observe(
            "User seemed very outgoing in this session",
            domain="personality_trait",
            mismatch_magnitude=0.6,
            source="weak_inference",
        )
        # Should log mismatch or confirm, NOT supersede
        assert result.action in ("logged_mismatch", "confirmed"), \
            (f"Stable domain with moderate mismatch from weak source "
             f"should not supersede; got {result.action}")

def test_inspect_returns_scoring_breakdown():
    with MemoryLayer(":memory:") as mem:
        r = mem.write("User lives in Berlin", domain="location")
        info = mem.inspect(r.item.id)
        assert "effective_volatility" in info
        assert "protection_weight" in info
        assert "staleness" in info
        assert info["effective_volatility"] == DOMAIN_VOLATILITY["location"]

def test_summary():
    with MemoryLayer(":memory:") as mem:
        mem.write("A", domain="core_preference")
        mem.write("B", domain="core_preference")
        mem.write("C", domain="emotional_context")
        s = mem.summary()
        assert s["total_active_memories"] == 3
        assert s["by_domain"]["core_preference"] == 2
        assert s["by_domain"]["emotional_context"] == 1


# ── content-level matching (multi-fact domains) ────────────────────────────────

def test_observe_matches_right_item_in_multi_fact_domain():
    """Two distinct facts in one domain must not collide: an update should
    supersede the semantically-matching item, leaving the other untouched."""
    # use a volatile domain so a high-mismatch update actually audits
    with MemoryLayer(":memory:") as mem:   # default keyword similarity
        a = mem.write("building the billing service", domain="current_project")
        b = mem.write("migrating the user database", domain="current_project")
        res = mem.observe("now building the billing dashboard",
                          domain="current_project", mismatch_magnitude=0.9,
                          source="explicit_statement")
        assert res.action == "audited", f"expected audit, got {res.action}"
        # the database project must still be active and unchanged
        active = {i.content for i in mem._store.all_active(domain="current_project")}
        assert "migrating the user database" in active
        # the billing item (the semantic match) was the one superseded
        assert mem._store.get(a.item.id).superseded_by is not None
        assert mem._store.get(b.item.id).superseded_by is None


# ── batteries-included remember() / recall() ───────────────────────────────────

def test_remember_classifies_domain_for_new_facts():
    with MemoryLayer(":memory:") as mem:
        assert mem.remember("I was born in Spain").item.domain == "biographical"
        assert mem.remember("I feel anxious today").item.domain \
            == "emotional_context"
        assert mem.remember("I am working on the payments project").item.domain \
            == "current_project"


def test_remember_updates_related_fact():
    """A follow-up statement about the same fact should update it, not duplicate."""
    with MemoryLayer(":memory:") as mem:   # keyword similarity is enough here
        mem.remember("I live in Berlin")
        res = mem.remember("I live in Paris")
        assert res.action in ("audited", "logged_mismatch"), res.action
        locs = mem._store.all_active(domain="location")
        assert len(locs) == 1, "should update in place, not create a 2nd location"


def test_recall_returns_plain_strings():
    with MemoryLayer(":memory:") as mem:
        mem.remember("I prefer concise answers")
        out = mem.recall("how should responses be written", top_k=3)
        assert isinstance(out, list)
        assert all(isinstance(s, str) for s in out)


# ── multi-tenant namespacing ───────────────────────────────────────────────────

def test_for_user_isolates_memories():
    """Two tenants in one database must not see or overwrite each other's facts."""
    with MemoryLayer(":memory:") as mem:
        alice = mem.for_user("alice")
        bob = mem.for_user("bob")

        alice.remember("I live in Berlin")
        bob.remember("I live in Paris")

        assert alice.recall("where live", top_k=1) == ["I live in Berlin"]
        assert bob.recall("where live", top_k=1) == ["I live in Paris"]

        assert alice.summary()["namespace"] == "alice"
        assert bob.summary()["namespace"] == "bob"
        assert alice.summary()["total_active_memories"] == 1
        assert bob.summary()["total_active_memories"] == 1


def test_cross_tenant_observe_does_not_match():
    """Bob's update must not supersede Alice's memory even in the same domain."""
    with MemoryLayer(":memory:") as mem:
        alice = mem.for_user("alice")
        bob = mem.for_user("bob")

        alice.write("User prefers dark mode", domain="core_preference")
        bob.observe("User prefers light mode", domain="core_preference",
                    mismatch_magnitude=0.9, source="explicit_statement")

        alice_items = alice._active(domain="core_preference")
        bob_items = bob._active(domain="core_preference")
        assert len(alice_items) == 1
        assert alice_items[0].content == "User prefers dark mode"
        assert len(bob_items) == 1
        assert bob_items[0].content == "User prefers light mode"


def test_inspect_hides_other_namespace():
    with MemoryLayer(":memory:") as mem:
        alice = mem.for_user("alice")
        bob = mem.for_user("bob")
        r = alice.write("secret", domain="biographical")
        info = bob.inspect(r.item.id)
        assert "error" in info


def test_clear_removes_namespace_memories():
    with MemoryLayer(":memory:") as mem:
        mem.remember("I live in Berlin")
        assert mem.recall("where do I live")
        mem.clear()
        assert mem.recall("where do I live") == []


def test_langchain_memory_roundtrip():
    try:
        from voltmem.integrations.langchain import VoltMemMemory
    except ImportError:
        print("  SKIP  test_langchain_memory_roundtrip (install requirements-integrations.txt)")
        return

    memory = VoltMemMemory(session_id="lc-test", db_path=":memory:", top_k=3)
    try:
        assert memory.load_memory_variables({"input": "hello"})["history"] == ""

        memory.save_context(
            {"input": "I live in Berlin"},
            {"output": "Noted."},
        )
        vars_after = memory.load_memory_variables(
            {"input": "Where do I live?"}
        )
        history = vars_after["history"]
        assert "Berlin" in history

        memory.save_context(
            {"input": "Actually I moved to Paris"},
            {"output": "Updated."},
        )
        vars_final = memory.load_memory_variables(
            {"input": "Where do I live?"}
        )
        assert "Paris" in vars_final["history"]
    finally:
        memory.close()


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_stable_domain_gets_high_protection,
        test_protection_weight_clamped,
        test_freshly_confirmed_item_low_staleness,
        test_volatile_item_goes_stale_faster_than_stable,
        test_high_mismatch_volatile_escalates,
        test_low_mismatch_stable_does_not_escalate,
        test_high_repetition_suppresses_escalation,
        test_threshold_scales_inversely_with_volatility,
        test_stale_volatile_item_ranked_lower_than_fresh,
        test_stable_item_age_barely_penalised,
        test_write_and_retrieve,
        test_low_mismatch_confirms_not_supersedes,
        test_high_mismatch_volatile_supersedes,
        test_high_mismatch_stable_does_not_supersede,
        test_inspect_returns_scoring_breakdown,
        test_summary,
        test_observe_matches_right_item_in_multi_fact_domain,
        test_remember_classifies_domain_for_new_facts,
        test_remember_updates_related_fact,
        test_recall_returns_plain_strings,
        test_for_user_isolates_memories,
        test_cross_tenant_observe_does_not_match,
        test_inspect_hides_other_namespace,
        test_clear_removes_namespace_memories,
        test_langchain_memory_roundtrip,
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
