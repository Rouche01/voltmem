"""Lock `voltmem.scoring` to the Lean control-law spec (`lean/`).

Named cases match `VoltMem/Oracle.lean` `#guard`s. Algebraic identities match
`VoltMem/Algebra.lean`. Run: `python tests/test_lean_oracle.py`.
Rebuild the spec: `cd lean && lake build && lake exe voltmem_oracle`.
"""
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voltmem.domains import MemoryItem
from voltmem.memory import MemoryLayer
from voltmem.scoring import (
    ALPHA, THETA_0, SOURCE_RELIABILITY,
    escalation_score, escalation_decision, resolve_escalation_law,
    belief_shift_mass, belief_shift_bar,
)


def _g0() -> float:
    return 0.1 + 1.9 / (1.0 + math.exp(0.0))


def _item(
    domain: str,
    *,
    rep: int = 1,
    mismatch_count: int = 0,
    mismatch_ema: float = -1.0,
    mismatch_var: float = -1.0,
    surprise_ema: float = 0.0,
) -> MemoryItem:
    now = time.time()
    return MemoryItem(
        id="oracle",
        content="x",
        domain=domain,
        source="explicit_statement",
        repetition_count=rep,
        mismatch_count=mismatch_count,
        mismatch_ema=mismatch_ema,
        mismatch_var=mismatch_var,
        surprise_ema=surprise_ema,
        created_at=now,
        last_confirmed_at=now,
    )


def _run(item, M, src, mode, v_exp=None, mode_scale=None):
    esc, E, theta = escalation_decision(
        item, M, src, 0.0, 1.0, mode=mode, v_exp=v_exp, mode_scale=mode_scale)
    law = resolve_escalation_law(item, M, src, mode)
    return esc, E, theta, law


def test_homeostatic_double_charge_cancels_V():
    """E · θ = res · θ₀ · L. V drops out (Algebra.homeostatic_double_charge)."""
    item = _item("professional_context", rep=6)
    M, src, L = 0.90, "explicit_statement", 1.0
    E, theta = escalation_score(item, M, src, 0.0, L, mode="homeostatic")
    R = SOURCE_RELIABILITY[src]
    C = max(item.repetition_count, 1)
    res = (M * R / (C ** ALPHA)) * _g0()
    assert abs(E * theta - res * THETA_0 * L) < 1e-12


def test_career_cliff_p0_vs_p25():
    """Battery C: p=0 clears, p=0.25 misses. Matches Oracle.careerP0 / careerP25."""
    item = _item(
        "professional_context", rep=6, mismatch_ema=0.05, mismatch_var=0.001)
    esc0, _, _, _ = _run(
        item, 0.90, "explicit_statement", "allostatic",
        v_exp=0.0, mode_scale=False)
    esc25, _, _, _ = _run(
        item, 0.90, "explicit_statement", "homeostatic",
        v_exp=0.25, mode_scale=False)
    assert esc0 is True
    assert esc25 is False


def test_career_composite_opens_allostatic():
    item = _item(
        "professional_context", rep=6, mismatch_ema=0.05, mismatch_var=0.001)
    esc, _, _, law = _run(item, 0.90, "explicit_statement", "composite")
    assert law == "allostatic"
    assert esc is True
    esc_h, _, _, law_h = _run(item, 0.90, "explicit_statement", "homeostatic")
    assert law_h == "homeostatic"
    assert esc_h is False


def test_mislabel_insurance_stays_homeostatic():
    """Battery E: fresh mislabel, composite keeps the double-charge insurance."""
    item = _item("relationship")
    assert _run(item, 0.75, "strong_inference", "homeostatic")[0] is False
    assert _run(item, 0.75, "strong_inference", "allostatic")[0] is True
    esc, _, _, law = _run(item, 0.75, "strong_inference", "composite")
    assert law == "homeostatic"
    assert esc is False


def test_fresh_weak_does_not_open_composite():
    item = _item("professional_context")
    esc, _, _, law = _run(item, 0.40, "weak_inference", "composite")
    assert law == "homeostatic"
    assert esc is False


def test_very_stable_allostatic_does_not_one_shot():
    item = _item("core_preference", mismatch_count=2)
    assert _run(item, 0.90, "explicit_statement", "allostatic")[0] is False


def test_sleeptime_spacing_daily_vs_monthly():
    """Same 16 weaks: daily mass clears the job bar; monthly does not."""
    now = 1_700_000_000.0
    n, M, src = 16, 0.40, "weak_inference"

    def pile(gap_days: float):
        return [
            {
                "created_at": now - i * gap_days * 86400.0,
                "mismatch_magnitude": M,
                "source": src,
            }
            for i in range(n)
        ]

    daily = belief_shift_mass(pile(1.0), now=now)
    monthly = belief_shift_mass(pile(30.0), now=now)
    job_bar = belief_shift_bar("professional_context")
    pref_bar = belief_shift_bar("core_preference")
    assert daily > job_bar
    assert monthly < job_bar
    assert daily < pref_bar


def test_observe_miss_never_audits():
    """Matcher in front: no candidate ⇒ insert; the law is not consulted."""
    with MemoryLayer(":memory:") as mem:
        r = mem.observe("I live in Berlin", domain="location")
        assert r.action == "inserted"


def test_confirm_and_log_do_not_replace_content():
    with MemoryLayer(":memory:") as mem:
        mem.write("User is a data analyst", domain="professional_context")
        c = mem.observe(
            "User is a data analyst",
            domain="professional_context",
            mismatch_magnitude=0.05,
            source="explicit_statement",
        )
        assert c.action == "confirmed"
        assert "data analyst" in c.item.content
        logged = mem.observe(
            "User might be a nurse",
            domain="professional_context",
            mismatch_magnitude=0.40,
            source="weak_inference",
        )
        assert logged.action == "logged_mismatch"
        assert "data analyst" in logged.item.content


def main():
    tests = [
        test_homeostatic_double_charge_cancels_V,
        test_career_cliff_p0_vs_p25,
        test_career_composite_opens_allostatic,
        test_mislabel_insurance_stays_homeostatic,
        test_fresh_weak_does_not_open_composite,
        test_very_stable_allostatic_does_not_one_shot,
        test_sleeptime_spacing_daily_vs_monthly,
        test_observe_miss_never_audits,
        test_confirm_and_log_do_not_replace_content,
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


if __name__ == "__main__":
    main()
