"""
Tests for VoltMem — covering all core equation behaviours.
"""
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voltmem import MemoryLayer, DOMAIN_VOLATILITY
from voltmem.domains import MemoryItem
from voltmem.scoring import (
    escalation_score, staleness, retrieval_score,
    protection_weight, should_escalate,
    similarity_spread, freshness_mix,
    EXPLICIT_MIN_VD, MIX_MIN, SIM_SPREAD_FLAT, SIM_SPREAD_FULL,
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


def make_item_v(v_d: float, domain="professional_context", rep=1):
    """Item with explicit effective volatility (simulates auto_discover drift)."""
    return make_item(domain=domain, rep=rep, vol_ema=v_d)


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

def test_explicit_high_mismatch_updates_stable_professional_context():
    """P0: career change with explicit statement must escalate despite low V_d."""
    item = make_item(domain="professional_context")  # V_d = 0.30
    # Raw score alone fails (E≈0.28 < θ=0.5); band θ-cap must let it through.
    E_t, theta_t = escalation_score(
        item, mismatch_magnitude=0.90, source="explicit_statement")
    assert E_t <= theta_t, "precondition: formula alone should block"
    assert should_escalate(
        item, mismatch_magnitude=0.90, source="explicit_statement"), \
        "High-M explicit correction must update professional_context"


def test_escalation_medium_stable_v_grid_explicit_updates():
    """Drift-safe: explicit high-M should escalate across medium-stable V_d."""
    for v in [0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.55]:
        item = make_item_v(v)
        assert should_escalate(
            item, mismatch_magnitude=0.90, source="explicit_statement"), \
            f"medium-stable V_d={v} should escalate on explicit high-M"


def test_escalation_very_stable_v_grid_explicit_retains():
    """Very-stable band: explicit high-M must not one-shot update (pref blips)."""
    for v in [0.05, 0.08, 0.10, 0.12]:
        assert v < EXPLICIT_MIN_VD
        item = make_item_v(v, domain="core_preference")
        assert not should_escalate(
            item, mismatch_magnitude=0.90, source="explicit_statement"), \
            f"very-stable V_d={v} must retain on explicit high-M"


def test_explicit_cap_scales_with_drifted_volatility():
    """V_d drifted down within band (0.20) still updates; below band uses cumulative."""
    drifted = make_item_v(0.20)
    assert should_escalate(
        drifted, mismatch_magnitude=0.90, source="explicit_statement")
    below_band = make_item_v(0.12, domain="biographical")
    assert not should_escalate(
        below_band, mismatch_magnitude=0.90, source="explicit_statement")
    below_band.mismatch_count = 3
    assert should_escalate(
        below_band, mismatch_magnitude=0.70, source="strong_inference")

def test_weak_evidence_still_retained_on_stable_domain():
    """θ-cap must not weaken retain-on-noise / very-stable behaviour."""
    item = make_item(domain="personality_trait")
    assert not should_escalate(
        item, mismatch_magnitude=0.60, source="weak_inference")
    assert not should_escalate(
        item, mismatch_magnitude=0.75, source="strong_inference")
    # remember() uses explicit_statement + heuristic M≈0.9 on pref blips
    pref = make_item(domain="core_preference")
    assert not should_escalate(
        pref, mismatch_magnitude=0.90, source="explicit_statement"), \
        "core_preference paraphrase must still retain under θ-cap"

def test_cumulative_mismatches_eventually_escalate():
    item = make_item(domain="professional_context")
    item.mismatch_count = 3
    assert should_escalate(
        item, mismatch_magnitude=0.70, source="strong_inference"), \
        "After enough logged mismatches, further conflict should escalate"

def test_observe_audits_explicit_career_change():
    with MemoryLayer(":memory:") as mem:
        mem.write("User works as a data analyst", domain="professional_context")
        res = mem.observe(
            "User explicitly said they changed careers and now work as a nurse",
            domain="professional_context",
            mismatch_magnitude=0.90,
            source="explicit_statement",
        )
        assert res.action == "audited", f"expected audited, got {res.action}"


def test_cumulative_mismatches_integration_audits_career_change():
    """Below-band V_d: three logged conflicts then strong inference escalates."""
    with MemoryLayer(":memory:") as mem:
        item = mem.write(
            "User works as a data analyst", domain="biographical")
        stored = mem._store.get(item.item.id)
        stored.volatility_ema = 0.12
        mem._store.update(stored)

        for text in (
            "User mentioned a different role in passing",
            "User said something else about work",
            "User brought up career again",
        ):
            r = mem.observe(
                text,
                domain="biographical",
                mismatch_magnitude=0.65,
                source="weak_inference",
            )
            assert r.action == "logged_mismatch"

        final = mem.observe(
            "User explicitly said they changed careers and now work as a nurse",
            domain="biographical",
            mismatch_magnitude=0.75,
            source="strong_inference",
        )
        assert final.action == "audited", f"expected audited, got {final.action}"

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


def test_similarity_spread_and_freshness_mix():
    assert similarity_spread([]) == 0.0
    assert similarity_spread([0.5]) == 0.0
    assert abs(similarity_spread([0.70, 0.72]) - 0.02) < 1e-9
    assert freshness_mix(0.0) == MIX_MIN
    assert freshness_mix(SIM_SPREAD_FLAT) == MIX_MIN
    assert freshness_mix(SIM_SPREAD_FULL) == 1.0
    assert freshness_mix(0.50) == 1.0
    mid = freshness_mix((SIM_SPREAD_FLAT + SIM_SPREAD_FULL) / 2)
    assert MIX_MIN < mid < 1.0


def test_retrieval_score_mix_dampens_staleness_penalty():
    """On a plateau, lower mix shrinks freshness-driven score gaps."""
    now = time.time()
    old = now - 20 * 86400
    volatile = MemoryItem(
        id="v1", content="mood", domain="current_task",
        source="explicit_statement",
        created_at=old, last_confirmed_at=old,
    )
    stable = MemoryItem(
        id="s1", content="pref", domain="core_preference",
        source="explicit_statement",
        created_at=old, last_confirmed_at=old,
    )
    gap_full = abs(
        retrieval_score(volatile, 0.71, now=now, mix=1.0)
        - retrieval_score(stable, 0.70, now=now, mix=1.0)
    )
    gap_damp = abs(
        retrieval_score(volatile, 0.71, now=now, mix=MIX_MIN)
        - retrieval_score(stable, 0.70, now=now, mix=MIX_MIN)
    )
    assert gap_damp < gap_full, \
        f"dampened mix should shrink cross-domain score gap; {gap_damp} vs {gap_full}"


def test_plateau_retrieve_dampens_vs_clear_gap():
    """Near-equal sims → mix < 1; large sim gap → full freshness (mix = 1)."""
    now = time.time()
    old = now - 14 * 86400

    def plateau_sim(query, content):
        return {"volatile fact": 0.705, "stable fact": 0.700}.get(content, 0.0)

    def clear_sim(query, content):
        return {"volatile fact": 0.90, "stable fact": 0.40}.get(content, 0.0)

    with MemoryLayer(":memory:", similarity_fn=plateau_sim) as mem:
        mem.write("volatile fact", domain="current_task", at_time=old)
        mem.write("stable fact", domain="core_preference", at_time=old)
        # Force-confirm timestamps (write may refresh)
        for it in mem._active():
            it.last_confirmed_at = old
            it.created_at = old
            mem._store.update(it)
        plateau = mem.retrieve("what was I doing", top_k=2, now=now)
        assert len(plateau.items) == 2
        # Spread 0.005 → MIX_MIN; volatile penalty reduced vs mix=1
        vol = next(i for i in mem._active() if i.content == "volatile fact")
        stab = next(i for i in mem._active() if i.content == "stable fact")
        score_damp_v = retrieval_score(vol, 0.705, now=now, mix=MIX_MIN)
        score_full_v = retrieval_score(vol, 0.705, now=now, mix=1.0)
        assert score_damp_v > score_full_v
        # Ranking uses dampened path: scores should match mix=MIX_MIN
        by_id = {i.content: s for i, s in zip(plateau.items, plateau.scores)}
        assert abs(by_id["volatile fact"] - score_damp_v) < 1e-9

    with MemoryLayer(":memory:", similarity_fn=clear_sim) as mem:
        # Fresh volatile + clear sim gap → full mix; volatile should win.
        mem.write("volatile fact", domain="current_task", at_time=now)
        mem.write("stable fact", domain="core_preference", at_time=old)
        for it in mem._active():
            if it.content == "stable fact":
                it.last_confirmed_at = old
                it.created_at = old
                mem._store.update(it)
        clear = mem.retrieve("volatile fact query", top_k=2, now=now)
        vol = next(i for i in mem._active() if i.content == "volatile fact")
        expected = retrieval_score(vol, 0.90, now=now, mix=1.0)
        by_id = {i.content: s for i, s in zip(clear.items, clear.scores)}
        assert abs(by_id["volatile fact"] - expected) < 1e-9
        assert clear.items[0].content == "volatile fact"

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


def _mock_similarity(pairs: dict[tuple[str, str], float]):
    """Build a deterministic similarity fn for linking tests."""
    def sim(a: str, b: str) -> float:
        if a == b:
            return 1.0
        if (a, b) in pairs:
            return pairs[(a, b)]
        if (b, a) in pairs:
            return pairs[(b, a)]
        qa, qb = set(a.lower().split()), set(b.lower().split())
        if not qa or not qb:
            return 0.0
        return len(qa & qb) / max(len(qa), len(qb))
    return sim


def test_remember_slot_fallback_links_volatile_mood():
    """Paraphrased mood below global threshold still routes through observe()."""
    mood_a = "I'm feeling great today"
    mood_b = "I'm pretty stressed this week"
    sim_fn = _mock_similarity({(mood_a, mood_b): 0.44})
    with MemoryLayer(":memory:", similarity_fn=sim_fn) as mem:
        mem.remember(mood_a)
        res = mem.remember(mood_b)
        assert res.action == "audited", res.action
        assert len(mem._active(domain="emotional_context")) == 1
        assert "stressed" in res.item.content


def test_remember_slot_fallback_protects_stable_pref():
    """Stable pref blip links in-slot but volatility engine keeps original."""
    pref = "I prefer concise, direct answers"
    blip = "I really like short replies"
    sim_fn = _mock_similarity({(pref, blip): 0.50})
    with MemoryLayer(":memory:", similarity_fn=sim_fn) as mem:
        mem.remember(pref)
        res = mem.remember(blip)
        assert res.action == "logged_mismatch", res.action
        assert len(mem._active()) == 1
        assert "concise" in res.item.content
        top = mem.recall("how should I format replies", top_k=1)
        assert top and "concise" in top[0].lower()


def test_remember_slot_fallback_updates_location():
    """Location paraphrase below global threshold still supersedes stale city."""
    berlin = "I live in Berlin"
    paris = "I live in Paris now"
    sim_fn = _mock_similarity({(berlin, paris): 0.53})
    with MemoryLayer(":memory:", similarity_fn=sim_fn) as mem:
        mem.remember(berlin)
        res = mem.remember(paris)
        assert res.action == "audited", res.action
        assert len(mem._active(domain="location")) == 1
        assert "Paris" in res.item.content


def test_remember_cross_domain_no_false_link():
    """Unrelated domains must not be merged by slot fallback."""
    with MemoryLayer(":memory:") as mem:
        mem.remember("I live in Berlin")
        mem.remember("I prefer concise, direct answers")
        assert len(mem._active()) == 2
        assert len(mem._active(domain="location")) == 1
        assert len(mem._active(domain="core_preference")) == 1


def test_remember_preference_sibling_domains_link():
    """Prefs split across core/stated classifiers still share one slot."""
    pref = "I prefer concise, direct answers"
    blip = "I really like short replies"
    sim_fn = _mock_similarity({(pref, blip): 0.50})
    with MemoryLayer(":memory:", similarity_fn=sim_fn) as mem:
        mem.remember(pref)
        res = mem.remember(blip)
        assert res.action == "logged_mismatch", res.action
        domains = {i.domain for i in mem._active()}
        assert domains == {"core_preference"}


def test_remember_multi_fact_domain_ambiguous_no_link():
    """Two distinct projects must not collide on a weak, ambiguous update."""
    a = "building the billing service"
    b = "migrating the user database"
    vague = "working on infrastructure improvements"
    sim_fn = _mock_similarity({
        (vague, a): 0.40,
        (vague, b): 0.38,
    })
    with MemoryLayer(":memory:", similarity_fn=sim_fn) as mem:
        mem.write(a, domain="current_project")
        mem.write(b, domain="current_project")
        res = mem.remember(vague)
        assert res.action == "inserted", res.action
        assert len(mem._active(domain="current_project")) == 3


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


# ── Battery A regression (experiments/voltmem_eval.py) ────────────────────────

def test_voltmem_eval_battery_a_real_profile():
    """CI gate: expanded escalation probes must stay green under real priors."""
    from experiments.voltmem_eval import (
        CUMULATIVE_ESCALATION_PROBES,
        ESCALATION_PROBES,
        run_escalation,
    )

    correct, n, rows = run_escalation("real")
    expected_n = len(ESCALATION_PROBES) + len(CUMULATIVE_ESCALATION_PROBES)
    failed = [r for r in rows if not r[3]]
    assert n == expected_n and correct == n, (
        f"Battery A real: {correct}/{n} (expected {expected_n}); failures="
        + ", ".join(f"{r[0]} want={r[1]} got={r[2]} ({r[5]})" for r in failed)
    )


def test_voltmem_eval_battery_a_real_beats_controls():
    """Causal check: true priors outperform flat and swap on the expanded suite."""
    from experiments.voltmem_eval import run_escalation

    c_real, n_real, _ = run_escalation("real")
    c_flat, n_flat, _ = run_escalation("flat")
    c_swap, n_swap, _ = run_escalation("swap")
    a_real, a_flat, a_swap = c_real / n_real, c_flat / n_flat, c_swap / n_swap
    assert a_real > a_flat and a_real >= a_swap and a_real > 0.5, (
        f"causal fail: real={a_real:.0%} flat={a_flat:.0%} swap={a_swap:.0%}"
    )


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
        test_explicit_high_mismatch_updates_stable_professional_context,
        test_escalation_medium_stable_v_grid_explicit_updates,
        test_escalation_very_stable_v_grid_explicit_retains,
        test_explicit_cap_scales_with_drifted_volatility,
        test_weak_evidence_still_retained_on_stable_domain,
        test_cumulative_mismatches_eventually_escalate,
        test_observe_audits_explicit_career_change,
        test_cumulative_mismatches_integration_audits_career_change,
        test_voltmem_eval_battery_a_real_profile,
        test_voltmem_eval_battery_a_real_beats_controls,
        test_stale_volatile_item_ranked_lower_than_fresh,
        test_stable_item_age_barely_penalised,
        test_similarity_spread_and_freshness_mix,
        test_retrieval_score_mix_dampens_staleness_penalty,
        test_plateau_retrieve_dampens_vs_clear_gap,
        test_write_and_retrieve,
        test_low_mismatch_confirms_not_supersedes,
        test_high_mismatch_volatile_supersedes,
        test_high_mismatch_stable_does_not_supersede,
        test_inspect_returns_scoring_breakdown,
        test_summary,
        test_observe_matches_right_item_in_multi_fact_domain,
        test_remember_classifies_domain_for_new_facts,
        test_remember_updates_related_fact,
        test_remember_slot_fallback_links_volatile_mood,
        test_remember_slot_fallback_protects_stable_pref,
        test_remember_slot_fallback_updates_location,
        test_remember_cross_domain_no_false_link,
        test_remember_preference_sibling_domains_link,
        test_remember_multi_fact_domain_ambiguous_no_link,
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
