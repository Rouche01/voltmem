"""
Tests for VoltMem — covering all core equation behaviours.
"""
import math
import sys
import tempfile
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
    recent_surprise, surprise_mode_scale, update_surprise_ema,
    resolve_escalation_law,
    EXPLICIT_MIN_VD, MIX_MIN, SIM_SPREAD_FLAT, SIM_SPREAD_FULL,
    SURPRISE_HALFLIFE_DAYS,
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


def test_allostatic_one_shot_matches_battery_a_labels():
    """Fresh items: allostatic must not flip Battery A retain/update labels."""
    assert should_escalate(
        make_item("current_task"), 0.90, "explicit_statement", mode="allostatic")
    assert should_escalate(
        make_item("professional_context"), 0.90, "explicit_statement",
        mode="allostatic")
    assert not should_escalate(
        make_item("core_preference"), 0.90, "explicit_statement",
        mode="allostatic")
    assert not should_escalate(
        make_item("personality_trait"), 0.60, "weak_inference",
        mode="allostatic")


def test_allostatic_reopens_entrenched_professional_after_mismatches():
    """C=6 + 2 logged mismatches: homeostatic retains, allostatic updates."""
    item = make_item(domain="professional_context", rep=6)
    item.mismatch_count = 2
    assert not should_escalate(
        item, 0.90, "explicit_statement", mode="homeostatic"), \
        "homeostatic should still block high-C career change before N=3 cliff"
    assert should_escalate(
        item, 0.90, "explicit_statement", mode="allostatic"), \
        "allostatic should reopen professional_context"


def test_mode_scale_settles_with_quiet_time():
    """s(m) must decay back toward 1 — a lifetime counter could only ratchet open."""
    now = time.time()
    hot = make_item(domain="professional_context")
    hot.surprise_ema = 0.8
    hot.surprise_at = now
    cold = make_item(domain="professional_context")
    cold.surprise_ema = 0.8
    # six half-lives of quiet, expressed in half-lives so a retune cannot
    # turn this into a boundary case
    cold.surprise_at = now - 6 * SURPRISE_HALFLIFE_DAYS * 86400

    s_hot = surprise_mode_scale(hot, now)
    s_cold = surprise_mode_scale(cold, now)
    assert s_hot < 0.5, f"recent surprise should reopen the bar, got s={s_hot:.3f}"
    assert s_cold > 0.95, f"quiet time should re-settle the bar, got s={s_cold:.3f}"
    assert recent_surprise(cold, now) < 0.05


def test_mode_scale_ignores_lifetime_mismatch_count():
    """The ratchet: mismatch_count never decreases, so it must not drive s(m)."""
    now = time.time()
    item = make_item(domain="professional_context")
    item.mismatch_count = 25       # accumulated over the item's life
    item.surprise_ema = 0.0        # but nothing surprising recently
    assert surprise_mode_scale(item, now) == 1.0


def test_confirms_and_time_both_settle_surprise():
    """Confirms pull the EMA down; elapsed time decays what is left."""
    now = time.time()
    item = make_item(domain="professional_context")
    item.surprise_ema = 0.9
    item.surprise_at = now
    after_confirm = update_surprise_ema(
        item, 0.05, source="explicit_statement", now=now)
    assert after_confirm < 0.9

    stale = make_item(domain="professional_context")
    stale.surprise_ema = 0.9
    stale.surprise_at = now - 2 * SURPRISE_HALFLIFE_DAYS * 86400
    after_decay = update_surprise_ema(
        stale, 0.05, source="explicit_statement", now=now)
    assert after_decay < after_confirm, (
        "a stale spike should not survive a quiet stretch: "
        f"decayed={after_decay:.3f} fresh={after_confirm:.3f}")


def test_residual_surprise_is_distance_from_predicted_mismatch():
    """Same M is surprising after quiet confirms, unsurprising in a weak stream."""
    from voltmem.scoring import residual_surprise, update_mismatch_expectation

    quiet = make_item(domain="professional_context")
    spike = residual_surprise(quiet, 0.90)
    habituated = make_item(domain="professional_context")
    for _ in range(12):
        habituated.mismatch_ema, habituated.mismatch_var = (
            update_mismatch_expectation(habituated, 0.40, "weak_inference"))
    same_stream = residual_surprise(habituated, 0.40)
    assert spike > 0.7, f"quiet-then-shift should be unexpected, got {spike:.3f}"
    assert same_stream < 0.25, (
        f"a predicted weak stream should not look surprising, got {same_stream:.3f}")
    assert spike > same_stream


def test_residual_surprise_widens_with_domain_volatility():
    """V_d is anticipated noise: the same M is less surprising on a volatile slot."""
    from voltmem.scoring import residual_surprise

    trait = residual_surprise(make_item("personality_trait"), 0.60)
    mood = residual_surprise(make_item("emotional_context"), 0.60)
    assert trait > mood, (
        f"weak noise should be more unexpected on a trait ({trait:.3f}) "
        f"than on mood ({mood:.3f})")


def _belief_stream(domain, n, gap_days, mm=0.70, src="weak_inference",
                   confirm_at_day=None):
    from voltmem.scoring import belief_has_shifted
    start = 1_700_000_000.0
    item = make_item(domain)
    item.created_at = start
    item.last_confirmed_at = start
    if confirm_at_day is not None:
        item.last_confirmed_at = start + confirm_at_day * 86400.0
    evidence = [
        {
            "mismatch_magnitude": mm,
            "source": src,
            "created_at": start + i * gap_days * 86400.0,
        }
        for i in range(n)
    ]
    now = start + (n - 1) * gap_days * 86400.0
    return belief_has_shifted(item, evidence, now=now)


def test_belief_shift_daily_weak_pile_moves_professional():
    """Sleeptime job: sixteen daily asides should move belief in a medium-band fact."""
    moved, mass, bar = _belief_stream("professional_context", 16, 1.0)
    assert moved, f"daily pile should consolidate, mass={mass:.3f} bar={bar:.3f}"


def test_belief_shift_monthly_drip_does_not_move():
    """Same sixteen asides spread monthly must decay, not accumulate."""
    moved, mass, bar = _belief_stream("professional_context", 16, 30.0)
    assert not moved, f"monthly drip must not consolidate, mass={mass:.3f} bar={bar:.3f}"


def test_belief_shift_cannot_erode_very_stable_domain():
    moved, mass, bar = _belief_stream("core_preference", 16, 1.0)
    assert not moved, f"preference must hold, mass={mass:.3f} bar={bar:.3f}"


def test_belief_shift_later_confirm_resets_the_pile():
    moved, mass, bar = _belief_stream(
        "professional_context", 16, 1.0, confirm_at_day=15.0)
    assert not moved, f"a late confirm should wipe the pile, mass={mass:.3f}"


def test_allostatic_stable_survives_sustained_weak_contradiction():
    """Erosion guard: s(m) must not let weak inferences grind down a trait."""
    with MemoryLayer(":memory:", escalation_mode="allostatic") as mem:
        mem.write("User is a careful, risk-averse planner",
                  domain="personality_trait")
        actions = [
            mem.observe(
                f"User did something a bit out of character ({i})",
                domain="personality_trait",
                mismatch_magnitude=0.60,
                source="weak_inference",
            ).action
            for i in range(12)
        ]
        assert "audited" not in actions, (
            "sustained weak contradiction overwrote a stable trait: "
            f"{actions}")


def _slow_burn(escalation_mode, domain, gap_days, turns=16, mm=0.70):
    """Weak-evidence stream; returns the 1-based turn that escalated, or None.

    weak_inference (R=0.4) sits below both the explicit θ-cap and the cumulative
    N-strike override, so only s(m) can reopen the bar here.
    """
    start = 1_700_000_000.0
    with MemoryLayer(":memory:", escalation_mode=escalation_mode) as mem:
        mem.write("User works as a data analyst", domain=domain, at_time=start)
        for i in range(turns):
            res = mem.observe(
                f"User mentioned a nursing shift in passing ({i})",
                domain=domain,
                mismatch_magnitude=mm,
                source="weak_inference",
                at_time=start + (i + 1) * gap_days * 86400.0,
            )
            if res.action == "audited":
                return i + 1
    return None


def test_allostatic_does_not_treat_a_predicted_weak_stream_as_surprise():
    """First-test finding: r_t habituates, so s(m) no longer catches Battery D.

    Sixteen daily weak mentions of a job change used to ride an EMA of raw M
    sitting 2e-4 from the trigger. Scoring surprise as |M − Ê| / σ makes that
    stream predicted after a few hits, so the bar stays closed. Catching the
    pile is now a cumulative-belief / sleeptime job, not online s(m).
    """
    assert _slow_burn("allostatic", "professional_context", gap_days=1.0) is None, \
        "a predicted weak stream must not reopen via s(m) alone"
    assert _slow_burn("homeostatic", "professional_context", gap_days=1.0) is None


def test_slow_burn_needs_recent_surprise_not_just_many():
    """Same evidence, same count, spread thin: must decay, not accumulate."""
    assert _slow_burn("allostatic", "professional_context", gap_days=30.0) is None, \
        "mentions spread over months should decay between hits, not pile up"


def test_slow_burn_cannot_erode_very_stable_domain():
    assert _slow_burn("allostatic", "core_preference", gap_days=1.0) is None, \
        "a deep preference must not yield to weak inference alone"


def test_label_error_insurance_is_what_allostatic_gives_up():
    """The documented cost of allostatic mode (Battery E), pinned deterministically.

    A personality trait (V=0.05) misread by the classifier as a relationship
    (V=0.35) — a real confusion pair from the corpus — then contradicted by
    weak-ish evidence. homeostatic survives because it ALSO discounts the score by
    V_d, so a mislabeled stable fact still scores low. allostatic drops that
    discount and overwrites the trait. The 'double V_d penalty' is functioning
    as insurance against label error, which is why allostatic is not the default.
    """
    mislabeled = make_item(domain="relationship")   # true domain: personality_trait
    assert not should_escalate(
        mislabeled, 0.75, "strong_inference", mode="homeostatic"), \
        "homeostatic's V_d discount should protect a mislabeled stable fact"
    assert should_escalate(
        mislabeled, 0.75, "strong_inference", mode="allostatic"), \
        "if this stops holding, Battery E's tradeoff has changed — re-run it"
    assert not should_escalate(
        mislabeled, 0.75, "strong_inference", mode="composite"), \
        "composite on a fresh item must keep homeostatic's insurance"


def test_composite_uses_allostatic_for_explicit_career_change():
    """Battery C via the gate: explicit high-M after confirms → allostatic."""
    item = make_item(domain="professional_context", rep=6)
    item.mismatch_ema = 0.05
    item.mismatch_var = 0.001
    assert should_escalate(
        item, 0.90, "explicit_statement", mode="composite")
    assert resolve_escalation_law(
        item, 0.90, "explicit_statement", "composite") == "allostatic"


def test_composite_stays_homeostatic_on_expected_weak_noise():
    """Learned stream of weak M: r_t is low, so the gate must not open."""
    from voltmem.scoring import update_mismatch_expectation
    item = make_item(domain="professional_context")
    for _ in range(8):
        item.mismatch_ema, item.mismatch_var = update_mismatch_expectation(
            item, 0.40, "weak_inference")
    assert resolve_escalation_law(
        item, 0.40, "weak_inference", "composite") == "homeostatic"
    assert not should_escalate(
        item, 0.40, "weak_inference", mode="composite")


def test_allostatic_very_stable_retains_after_two_mismatches():
    """Negative control: s(m) must not one-shot core_preference after 2 strikes."""
    item = make_item(domain="core_preference", rep=1)
    item.mismatch_count = 2
    assert not should_escalate(
        item, 0.90, "explicit_statement", mode="allostatic")


def test_allostatic_observe_resettles_after_audit():
    """After an update, surprise_ema resets so a weak blip does not flip back."""
    with MemoryLayer(":memory:", escalation_mode="allostatic") as mem:
        mem.write("User works as a data analyst", domain="professional_context")
        for i in range(5):
            r = mem.observe(
                f"User is still a data analyst ({i})",
                domain="professional_context",
                mismatch_magnitude=0.05,
                source="explicit_statement",
            )
            assert r.action == "confirmed"
        for text in ("mentioned a different role", "said something else about work"):
            r = mem.observe(
                text,
                domain="professional_context",
                mismatch_magnitude=0.65,
                source="weak_inference",
            )
            assert r.action == "logged_mismatch"
        updated = mem.observe(
            "User explicitly said they changed careers and now work as a nurse",
            domain="professional_context",
            mismatch_magnitude=0.90,
            source="explicit_statement",
        )
        assert updated.action == "audited", f"got {updated.action}"
        assert updated.item.surprise_ema == 0.0
        blip = mem.observe(
            "Someone mentioned the old analyst job in passing",
            domain="professional_context",
            mismatch_magnitude=0.60,
            source="weak_inference",
        )
        assert blip.action != "audited", f"re-settle failed, got {blip.action}"


def test_invalid_escalation_mode_rejected():
    try:
        MemoryLayer(":memory:", escalation_mode="nope")
    except ValueError as e:
        assert "escalation_mode" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_default_escalation_mode_is_composite():
    with MemoryLayer(":memory:") as mem:
        assert mem.escalation_mode == "composite"


def test_current_escalation_mode_alias():
    from voltmem.scoring import normalize_escalation_mode
    assert normalize_escalation_mode("current") == "homeostatic"
    with MemoryLayer(":memory:", escalation_mode="current") as mem:
        assert mem.escalation_mode == "homeostatic"

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
        assert "surprise_ema" in info
        assert info["escalation_mode"] == "composite"
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


# ── two-stage linking (recall then verify) ───────────────────────────────────

def _counting_verifier(decide):
    """A verifier plus a record of what it was asked, so recall can be asserted."""
    seen = []

    def verify(new_text, stored_text, domain):
        seen.append((stored_text, new_text))
        return decide(new_text, stored_text, domain)

    verify.seen = seen
    return verify


def test_auto_verifier_stays_off_without_an_embedder():
    with MemoryLayer(":memory:") as mem:
        assert mem._link_verifier is None


def test_auto_verifier_attaches_when_an_embedder_is_present():
    from voltmem.verify import LLMLinkVerifier

    class _Emb:
        def __call__(self, a, b):
            return MemoryLayer._similarity(a, b)

        def embed(self, text):
            return [1.0, 0.0]

    with MemoryLayer(":memory:", similarity_fn=_Emb()) as mem:
        assert isinstance(mem._link_verifier, LLMLinkVerifier)
        assert mem.verify_on_write is False
    with MemoryLayer(":memory:", similarity_fn=_Emb(),
                     verify_on_write=True) as mem:
        assert mem.verify_on_write is True
    with MemoryLayer(":memory:", similarity_fn=_Emb(),
                     link_verifier=None) as mem:
        assert mem._link_verifier is None


def test_sleeptime_default_inserts_grey_without_asking():
    """Embedder + auto verifier: grey writes insert; 14B waits for sleeptime."""
    mood_a = "I'm feeling great today"
    mood_b = "I'm pretty stressed this week"
    sim_fn = _mock_similarity({(mood_a, mood_b): 0.70})

    class _Emb:
        def __call__(self, a, b):
            return sim_fn(a, b)

        def embed(self, text):
            return [1.0, 0.0]

    verifier = _counting_verifier(lambda *_a: True)
    with MemoryLayer(":memory:", similarity_fn=_Emb()) as mem:
        from voltmem.verify import as_verifier
        mem._link_verifier = as_verifier(verifier)
        mem.remember(mood_a)
        res = mem.remember(mood_b)
        assert res.action == "inserted", res.action
        assert len(mem._active()) == 2
        assert verifier.seen == []


def test_verify_on_write_asks_grey_not_heuristic_refusal():
    """Hatch: grey frames ask now; a heuristic KEEP_BOTH never reaches 14B."""
    mood_a = "I'm feeling great today"
    mood_b = "I'm pretty stressed this week"
    py = "User is proficient in Python"
    ja = "User is proficient in Japanese"
    sim_fn = _mock_similarity({
        (mood_a, mood_b): 0.70,
        (py, ja): 0.80,
    })

    class _Emb:
        def __call__(self, a, b):
            return sim_fn(a, b)

        def embed(self, text):
            return [1.0, 0.0]

    grey = _counting_verifier(lambda *_a: True)
    with MemoryLayer(":memory:", similarity_fn=_Emb(),
                     verify_on_write=True) as mem:
        from voltmem.verify import as_verifier
        mem._link_verifier = as_verifier(grey)
        mem.remember(mood_a)
        res = mem.remember(mood_b)
        assert res.action in ("audited", "logged_mismatch"), res.action
        assert len(mem._active()) == 1
        assert grey.seen == [(mood_a, mood_b)]

    skills = _counting_verifier(lambda *_a: True)
    with MemoryLayer(":memory:", similarity_fn=_Emb(),
                     verify_on_write=True) as mem:
        from voltmem.verify import as_verifier
        mem._link_verifier = as_verifier(skills)
        mem.remember(py)
        res = mem.remember(ja)
        assert res.action == "inserted", res.action
        assert len(mem._active()) == 2
        assert skills.seen == []


def test_verifier_links_below_the_threshold_ladder():
    """A pair the ladder cannot reach still links: heuristic join, or a verifier.

    0.25 clears no bar in the shipped ladder — global relate is 0.55 and the
    professional_context slot bar is 0.40. Battery F failed here. Known-frame
    cards (work as X) now join without a model; an injected verifier still
    sees pairs the heuristic does not cover.
    """
    stored = "User works as a data analyst"
    new = "User changed careers and now works as a nurse"
    sim_fn = _mock_similarity({(stored, new): 0.25})

    with MemoryLayer(":memory:", similarity_fn=sim_fn) as plain:
        plain.write(stored, domain="professional_context")
        res = plain.remember(new)
        assert res.action in ("audited", "logged_mismatch"), res.action
        assert len(plain._active()) == 1

    verifier = _counting_verifier(lambda *_a: True)
    with MemoryLayer(":memory:", similarity_fn=sim_fn,
                     link_verifier=verifier, link_recall_bar=0.20) as mem:
        mem.write(stored, domain="professional_context")
        res = mem.remember(new)
        assert res.action in ("audited", "logged_mismatch"), res.action
        assert len(mem._active()) == 1
        assert verifier.seen == [(stored, new)]


def test_verifier_refusal_prevents_a_false_merge():
    """A high-similarity distinct fact must survive. This is the merge Battery G
    showed no threshold can prevent: 0.80 similarity, two coexisting skills."""
    stored = "User is proficient in Python"
    new = "User is proficient in Japanese"
    sim_fn = _mock_similarity({(stored, new): 0.80})

    verifier = _counting_verifier(lambda *_a: False)
    with MemoryLayer(":memory:", similarity_fn=sim_fn,
                     link_verifier=verifier, link_recall_bar=0.20) as mem:
        mem.write(stored, domain="skill")
        res = mem.remember(new)
        assert res.action == "inserted", res.action
        assert len(mem._active()) == 2, "a true memory was destroyed"
        assert verifier.seen == [(stored, new)]


def test_verifier_failure_falls_back_to_the_ladder():
    """A dead model must not skip a pair the threshold ladder would have linked."""
    from voltmem.verify import LLMLinkVerifier

    verifier = LLMLinkVerifier(ollama_url="http://127.0.0.1:9", timeout=0.5)
    with MemoryLayer(":memory:", link_verifier=verifier,
                     link_recall_bar=0.20) as mem:
        mem.write("I live in Berlin", domain="location")
        res = mem.remember("I live in Paris")
        assert res.action in ("audited", "logged_mismatch"), res.action
        assert len(mem._active()) == 1
        assert "Paris" in mem._active()[0].content
    assert verifier.failures >= 1


def test_verifier_failure_below_the_ladder_is_still_a_duplicate():
    """If the ladder cannot link either, a dead verifier still inserts."""
    from voltmem.verify import LLMLinkVerifier

    stored = "User works as a data analyst"
    new = "User changed careers and now works as a nurse"
    sim_fn = _mock_similarity({(stored, new): 0.25})
    verifier = LLMLinkVerifier(ollama_url="http://127.0.0.1:9", timeout=0.5)
    with MemoryLayer(":memory:", similarity_fn=sim_fn, link_verifier=verifier,
                     link_recall_bar=0.20) as mem:
        mem.write(stored, domain="professional_context")
        res = mem.remember(new)
        assert res.action == "inserted", res.action
        assert len(mem._active()) == 2
    assert verifier.failures >= 1


def test_verifier_is_asked_in_similarity_order_and_stops_at_the_first_yes():
    """Candidates are charged for most-likely-first, and only until one agrees."""
    near = "User is building the billing service"
    far = "User has a sister named Alice"
    new = "User is building the payments service"
    sim_fn = _mock_similarity({(near, new): 0.70, (far, new): 0.35})

    verifier = _counting_verifier(lambda *_a: True)
    with MemoryLayer(":memory:", similarity_fn=sim_fn,
                     link_verifier=verifier, link_recall_bar=0.20) as mem:
        mem.write(far, domain="relationship")
        mem.write(near, domain="current_project")
        mem.remember(new)
        assert verifier.seen == [(near, new)], verifier.seen


def test_verifier_respects_the_recall_bar_and_top_k():
    """Below the bar nothing is asked; above it, at most top_k are."""
    sim_fn = _mock_similarity({
        ("a one", "z two"): 0.05,
    })
    verifier = _counting_verifier(lambda *_a: False)
    with MemoryLayer(":memory:", similarity_fn=sim_fn,
                     link_verifier=verifier, link_recall_bar=0.50,
                     link_recall_top_k=2) as mem:
        mem.write("a one", domain="skill")
        res = mem.remember("z two")
        assert res.action == "inserted"
        assert verifier.seen == [], "nothing above the recall bar should be asked"

    verifier = _counting_verifier(lambda *_a: False)
    with MemoryLayer(":memory:", link_verifier=verifier,
                     link_recall_bar=0.0, link_recall_top_k=2) as mem:
        for n, text in enumerate(["one two three", "one two four",
                                  "one two five", "one two six"]):
            mem.write(text, domain="skill")
        mem.remember("one two seven")
        assert len(verifier.seen) == 2, verifier.seen


def test_verdict_parsing_is_conservative():
    """Anything unreadable must mean keep-both, never merge."""
    from voltmem.verify import parse_verdict

    assert parse_verdict('{"decision": "UPDATE"}') is True
    assert parse_verdict('{"decision":"KEEP_BOTH"}') is False
    assert parse_verdict('```json\n{"decision": "UPDATE"}\n```') is True
    assert parse_verdict("the answer is KEEP_BOTH") is False
    assert parse_verdict("") is None
    assert parse_verdict("no idea") is None
    assert parse_verdict('{"decision": "MAYBE"}') is None


class _StubCrossEncoder:
    """Fixed scores so the CE verifier can be unit-tested without a download."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        n = len(pairs)
        out = self.scores[:n]
        self.scores = self.scores[n:]
        return out


def test_cross_encoder_verifier_uses_fitted_threshold():
    from voltmem.verify import CrossEncoderVerifier

    ce = CrossEncoderVerifier(encoder=_StubCrossEncoder([0.4, 0.9]), threshold=0.5)
    assert ce.verify("new", "stored", "skill") is False
    assert ce.verify("new", "stored", "skill") is True


def test_cross_encoder_verifier_refuses_unfitted_threshold():
    from voltmem.verify import CrossEncoderVerifier

    ce = CrossEncoderVerifier(encoder=_StubCrossEncoder([0.9]))
    try:
        ce.verify("new", "stored", "skill")
    except ValueError as exc:
        assert "threshold" in str(exc)
    else:
        raise AssertionError("unfitted verifier must refuse to merge")


def test_fit_score_threshold_prefers_fewer_false_merges():
    from voltmem.verify import fit_score_threshold

    t, missed, merged = fit_score_threshold([4.0, 5.0, 6.0], [1.0, 2.0, 3.0])
    assert missed == 0 and merged == 0
    assert 3.0 < t <= 4.0

    # Overlap: the conservative cut (fewer merges) wins the tie on total errors.
    t, missed, merged = fit_score_threshold([2.0, 5.0], [1.0, 3.0])
    assert merged <= 1
    assert t > 1.0


def _fact(subject, attribute, value, cardinality="slot", replaces=False):
    from voltmem.structure import StructuredFact
    return StructuredFact(subject, attribute, value, cardinality, replaces)


def test_structured_join_updates_a_slot():
    from voltmem.structure import join_structured
    stored = [_fact("user", "residence_city", "Berlin")]
    new = [_fact("I", "residence_city", "Paris")]
    assert join_structured(stored, new) is True


def test_structured_join_keeps_two_skills():
    from voltmem.structure import join_structured
    stored = [_fact("user", "skill_python", "proficient", "multi")]
    new = [_fact("user", "skill_japanese", "proficient", "multi")]
    assert join_structured(stored, new) is False


def test_structured_join_updates_multi_only_with_a_marker():
    from voltmem.structure import join_structured
    stored = [_fact("user", "skill_python", "proficient", "multi")]
    ended = [_fact("user", "skill_python", "ended", "multi", replaces=True)]
    rust = [_fact("user", "skill_rust", "proficient", "multi")]
    assert join_structured(stored, ended) is True
    assert join_structured(stored, rust, "User no longer uses Python") is False
    assert join_structured(stored, rust) is False
    # Same attribute, marker on the new text, no replaces flag.
    assert join_structured(stored, [_fact("user", "skill_python", "ended", "multi")],
                           "User no longer uses Python") is True


def test_structured_join_rejects_a_different_subject():
    from voltmem.structure import join_structured
    stored = [_fact("user", "residence_city", "Berlin")]
    new = [_fact("user_parents", "residence_city", "Hamburg")]
    assert join_structured(stored, new) is False


def test_structured_join_empty_extract_never_merges():
    from voltmem.structure import join_structured
    stored = [_fact("user", "occupation", "analyst")]
    assert join_structured(stored, []) is False
    assert join_structured([], stored) is False


def test_parse_structured_reads_facts_array():
    from voltmem.structure import parse_structured
    facts = parse_structured(
        '{"facts": [{"subject": "user", "attribute": "birth_year",'
        ' "value": "1991", "cardinality": "slot", "replaces": false}]}')
    assert len(facts) == 1
    assert facts[0].attribute == "birth_year"
    assert facts[0].is_slot
    assert parse_structured("not json") == []
    assert parse_structured("") == []


def test_conservative_join_blocks_named_ending_on_wrong_entity():
    from voltmem.structure import join_structured
    stored = [_fact("user", "current_manager", "Dana")]
    new = [_fact("user", "current_manager", "Miguel", "slot", True)]
    naive = join_structured(
        stored, new, "User no longer reports to Miguel",
        "User reports to Dana", conservative=False)
    safe = join_structured(
        stored, new, "User no longer reports to Miguel",
        "User reports to Dana")
    assert naive is True
    assert safe is False


def test_conservative_join_updates_when_ended_name_is_the_stored_value():
    from voltmem.structure import join_structured
    stored = [_fact("user", "employer", "Stripe")]
    new = [_fact("user", "employer", "Figma", "slot", True)]
    assert join_structured(
        stored, new, "User left Stripe and joined Figma",
        "User works at Stripe") is True


def test_conservative_join_parks_generic_slots_without_overlap():
    from voltmem.structure import join_structured
    stored = [_fact("user", "current_task", "drafting the quarterly report")]
    new = [_fact("user", "current_task", "needs to renew their passport")]
    assert join_structured(
        stored, new, "User needs to renew their passport",
        "User is drafting the quarterly report") is False
    # Same drawer, shared content word — the slides handoff.
    slides = [_fact("user", "current_task", "Monday slides")]
    done = [_fact("user", "current_task", "writing the report")]
    assert join_structured(
        slides, done, "User finished the slides and is now writing the report",
        "User is preparing the Monday slides") is True


def test_heuristic_extractor_covers_known_frames_only():
    from voltmem.structure import HeuristicStructuredExtractor, join_structured

    hx = HeuristicStructuredExtractor()
    city = hx.extract("I live in Berlin")
    assert city and city[0].attribute == "residence_city" and city[0].value == "Berlin"
    year = hx.extract("User was born in 1990")
    assert year and year[0].attribute == "birth_year" and year[0].value == "1990"
    py = hx.extract("User is proficient in Python")
    ja = hx.extract("User is proficient in Japanese")
    assert py[0].attribute == "skill_python"
    assert ja[0].attribute == "skill_japanese"
    assert join_structured(py, ja) is False
    job = hx.extract("User works as a data analyst")
    nurse = hx.extract(
        "User explicitly said they changed careers and now work as a nurse")
    assert job[0].attribute == "occupation"
    assert nurse[0].attribute == "occupation"
    assert join_structured(
        job, nurse,
        "User explicitly said they changed careers and now work as a nurse",
        "User works as a data analyst") is True
    assert hx.extract("User reports to Dana") == []
    parents = hx.extract("User's parents live in Hamburg")
    user = hx.extract("User lives in Berlin")
    assert parents[0].subject == "user_parents"
    assert join_structured(user, parents,
                           "User's parents live in Hamburg",
                           "User lives in Berlin") is False


def test_heuristic_remember_updates_city_and_keeps_skills():
    with MemoryLayer(":memory:") as mem:
        mem.remember("I live in Berlin")
        res = mem.remember("I live in Paris")
        assert res.action in ("audited", "logged_mismatch"), res.action
        assert len(mem._active(domain="location")) == 1
    with MemoryLayer(":memory:") as mem:
        mem.remember("User is proficient in Python")
        res = mem.remember("User is proficient in Japanese")
        assert res.action == "inserted", res.action
        assert len(mem._active(domain="skill")) == 2


def test_write_stamps_heuristic_facts():
    with MemoryLayer(":memory:") as mem:
        city = mem.write("User lives in Berlin", domain="location")
        assert city.item.facts
        assert city.item.facts[0]["attribute"] == "residence_city"
        assert city.item.facts[0]["value"] == "Berlin"
        mood = mem.write("I feel anxious today", domain="emotional_context")
        assert mood.item.facts == []
        info = mem.inspect(city.item.id)
        assert info["facts"][0]["attribute"] == "residence_city"


def test_heuristic_links_same_subject_below_the_recall_bar():
    """Persisted SAV recall is by subject, not cosine."""
    stored = "User works as a data analyst"
    new = "User changed careers and now works as a nurse"
    sim_fn = _mock_similarity({(stored, new): 0.0})
    with MemoryLayer(":memory:", similarity_fn=sim_fn,
                     link_recall_bar=0.20) as mem:
        mem.write(stored, domain="professional_context")
        res = mem.remember(new)
        assert res.action in ("audited", "logged_mismatch"), res.action
        assert len(mem._active()) == 1


def test_heuristic_keeps_different_subject_at_full_similarity():
    stored = "User lives in Berlin"
    new = "User's parents live in Hamburg"
    sim_fn = _mock_similarity({(stored, new): 1.0})
    with MemoryLayer(":memory:", similarity_fn=sim_fn) as mem:
        mem.remember(stored)
        res = mem.remember(new)
        assert res.action == "inserted", res.action
        assert len(mem._active()) == 2


def test_heuristic_facts_survive_reopen():
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "facts.db")
        with MemoryLayer(path) as mem:
            mem.remember("I live in Berlin")
        with MemoryLayer(path) as mem:
            items = mem._active(domain="location")
            assert items[0].facts
            assert items[0].facts[0]["attribute"] == "residence_city"
            res = mem.remember("I live in Paris")
            assert res.action in ("audited", "logged_mismatch"), res.action
            assert len(mem._active(domain="location")) == 1


def test_conservative_join_allows_positive_manager_fill():
    from voltmem.structure import join_structured
    stored = [_fact("user", "current_manager", "Dana")]
    new = [_fact("user", "current_manager", "Priya")]
    assert join_structured(
        stored, new, "User now reports to Priya",
        "User reports to Dana") is True


def test_structured_join_verifier_uses_injected_extracts():
    from voltmem.structure import StructuredJoinVerifier, StructuredFact

    class _Stub:
        def extract(self, text):
            table = {
                "User works as a data analyst": [
                    StructuredFact("user", "occupation", "analyst", "slot")],
                "User now works as a nurse": [
                    StructuredFact("user", "occupation", "nurse", "slot")],
                "User is proficient in Japanese": [
                    StructuredFact("user", "skill_japanese", "proficient", "multi")],
            }
            return table.get(text, [])

    v = StructuredJoinVerifier(extractor=_Stub())
    assert v.verify("User now works as a nurse",
                    "User works as a data analyst", "professional_context")
    assert not v.verify("User is proficient in Japanese",
                        "User works as a data analyst", "skill")


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


def test_voltmem_eval_battery_a_allostatic_real_profile():
    """Allostatic trigger must not regress Battery A under real priors."""
    from experiments.voltmem_eval import (
        CUMULATIVE_ESCALATION_PROBES,
        ESCALATION_PROBES,
        run_escalation,
    )

    correct, n, rows = run_escalation("real", escalation_mode="allostatic")
    expected_n = len(ESCALATION_PROBES) + len(CUMULATIVE_ESCALATION_PROBES)
    failed = [r for r in rows if not r[3]]
    assert n == expected_n and correct == n, (
        f"Battery A allostatic real: {correct}/{n} (expected {expected_n}); failures="
        + ", ".join(f"{r[0]} want={r[1]} got={r[2]} ({r[5]})" for r in failed)
    )


def test_voltmem_eval_battery_a_composite_real_profile():
    """Composite must not regress Battery A under real priors."""
    from experiments.voltmem_eval import (
        CUMULATIVE_ESCALATION_PROBES,
        ESCALATION_PROBES,
        run_escalation,
    )

    correct, n, rows = run_escalation("real", escalation_mode="composite")
    expected_n = len(ESCALATION_PROBES) + len(CUMULATIVE_ESCALATION_PROBES)
    failed = [r for r in rows if not r[3]]
    assert n == expected_n and correct == n, (
        f"Battery A composite real: {correct}/{n} (expected {expected_n}); failures="
        + ", ".join(f"{r[0]} want={r[1]} got={r[2]} ({r[5]})" for r in failed)
    )


def test_voltmem_eval_recency_shift_diagnostic():
    """Allostatic wins the professional noisy-shift case; preference stays closed."""
    from experiments.voltmem_eval import run_recency_shift

    rows = run_recency_shift()
    by_name = {row[0]["name"]: row for row in rows}
    prof = by_name["noisy-then-shift professional"]
    pref = by_name["noisy-then-shift preference"]
    assert prof[1] == "R" and prof[4] == "U", (
        f"professional diagnostic fail: homeostatic={prof[1]} allostatic={prof[4]}")
    assert pref[1] == "R" and pref[4] == "R", (
        f"preference negative control fail: homeostatic={pref[1]} allostatic={pref[4]}")
    quiet = by_name["quiet-then-shift professional"]
    assert quiet[1] == "R" and quiet[4] == "U", (
        f"quiet-shift fail: homeostatic={quiet[1]} allostatic={quiet[4]}")
    from experiments.voltmem_eval import _run_recency_probe, RECENCY_SHIFT_PROBES
    for probe in RECENCY_SHIFT_PROBES:
        got, _act = _run_recency_probe(probe, "composite")
        assert got == probe["want_allostatic"], (
            f"composite {probe['name']}: want={probe['want_allostatic']} got={got}")


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
        test_allostatic_one_shot_matches_battery_a_labels,
        test_allostatic_reopens_entrenched_professional_after_mismatches,
        test_mode_scale_settles_with_quiet_time,
        test_mode_scale_ignores_lifetime_mismatch_count,
        test_confirms_and_time_both_settle_surprise,
        test_residual_surprise_is_distance_from_predicted_mismatch,
        test_residual_surprise_widens_with_domain_volatility,
        test_belief_shift_daily_weak_pile_moves_professional,
        test_belief_shift_monthly_drip_does_not_move,
        test_belief_shift_cannot_erode_very_stable_domain,
        test_belief_shift_later_confirm_resets_the_pile,
        test_allostatic_stable_survives_sustained_weak_contradiction,
        test_allostatic_does_not_treat_a_predicted_weak_stream_as_surprise,
        test_slow_burn_needs_recent_surprise_not_just_many,
        test_slow_burn_cannot_erode_very_stable_domain,
        test_label_error_insurance_is_what_allostatic_gives_up,
        test_composite_uses_allostatic_for_explicit_career_change,
        test_composite_stays_homeostatic_on_expected_weak_noise,
        test_allostatic_very_stable_retains_after_two_mismatches,
        test_allostatic_observe_resettles_after_audit,
        test_invalid_escalation_mode_rejected,
        test_default_escalation_mode_is_composite,
        test_current_escalation_mode_alias,
        test_voltmem_eval_battery_a_real_profile,
        test_voltmem_eval_battery_a_real_beats_controls,
        test_voltmem_eval_battery_a_allostatic_real_profile,
        test_voltmem_eval_battery_a_composite_real_profile,
        test_voltmem_eval_recency_shift_diagnostic,
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
        test_auto_verifier_stays_off_without_an_embedder,
        test_auto_verifier_attaches_when_an_embedder_is_present,
        test_sleeptime_default_inserts_grey_without_asking,
        test_verify_on_write_asks_grey_not_heuristic_refusal,
        test_verifier_links_below_the_threshold_ladder,
        test_verifier_refusal_prevents_a_false_merge,
        test_verifier_failure_falls_back_to_the_ladder,
        test_verifier_failure_below_the_ladder_is_still_a_duplicate,
        test_verifier_is_asked_in_similarity_order_and_stops_at_the_first_yes,
        test_verifier_respects_the_recall_bar_and_top_k,
        test_verdict_parsing_is_conservative,
        test_cross_encoder_verifier_uses_fitted_threshold,
        test_cross_encoder_verifier_refuses_unfitted_threshold,
        test_fit_score_threshold_prefers_fewer_false_merges,
        test_structured_join_updates_a_slot,
        test_structured_join_keeps_two_skills,
        test_structured_join_updates_multi_only_with_a_marker,
        test_structured_join_rejects_a_different_subject,
        test_structured_join_empty_extract_never_merges,
        test_parse_structured_reads_facts_array,
        test_conservative_join_blocks_named_ending_on_wrong_entity,
        test_conservative_join_updates_when_ended_name_is_the_stored_value,
        test_conservative_join_parks_generic_slots_without_overlap,
        test_conservative_join_allows_positive_manager_fill,
        test_heuristic_extractor_covers_known_frames_only,
        test_heuristic_remember_updates_city_and_keeps_skills,
        test_write_stamps_heuristic_facts,
        test_heuristic_links_same_subject_below_the_recall_bar,
        test_heuristic_keeps_different_subject_at_full_similarity,
        test_heuristic_facts_survive_reopen,
        test_structured_join_verifier_uses_injected_extracts,
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
