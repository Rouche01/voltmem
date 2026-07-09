"""
Multi-turn EMA erosion test — does stable protection survive repeated weak hits?
================================================================================

The VoltMem eval (voltmem_eval.py) used a fresh store per probe, i.e. a SINGLE
observation. That hid a real weakness in the original observe() logic:

  * the volatility EMA was updated IGNORING source reliability, and
  * it was updated TWICE per observe() call (once at the top, once in the
    confirm / logged_mismatch branch),

so a genuinely stable memory (e.g. a personality trait) could be dragged
"volatile" by a handful of weak, low-trust contradictions over several turns —
and once volatility climbs, the audit threshold drops and the stable fact can be
wrongly overwritten.

This script simulates many turns of weak, contradictory observations against one
stable memory and tracks:

  * the memory's effective volatility over time, and
  * whether it ever gets superseded (protection failure).

It compares the LEGACY behaviour (reproduced inline) against the CURRENT library
(the fix: reliability-weighted, single EMA update). A pass = the current library
keeps volatility bounded and never wrongly supersedes the stable memory.

Run:
    .venv/bin/python experiments/ema_erosion_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voltmem import MemoryLayer                       # noqa: E402
from voltmem.domains import (                          # noqa: E402
    MemoryItem, DOMAIN_VOLATILITY, SOURCE_RELIABILITY,
)
from voltmem.scoring import should_escalate, BETA      # noqa: E402

DOMAIN = "personality_trait"     # V_d = 0.05, should be very hard to overwrite
MISMATCH = 0.60                  # moderate contradiction each turn
SOURCE = "weak_inference"        # R = 0.4, low-trust — should barely move things
N_TURNS = 12


def legacy_update(current, mismatch):
    """Original EMA step: no reliability weighting, base rate (1 - BETA)."""
    return BETA * current + (1 - BETA) * mismatch


def run_legacy():
    """Reproduce the OLD observe() path for a stable item under repeated weak
    contradictions: double EMA update, reliability ignored."""
    v = DOMAIN_VOLATILITY[DOMAIN]          # start from the domain prior
    traj, superseded_turn = [], None
    for t in range(1, N_TURNS + 1):
        # top-of-observe update (mismatch > 0)
        v = legacy_update(v, MISMATCH)
        probe = MemoryItem(id="x", content="x", domain=DOMAIN, source=SOURCE,
                           volatility_ema=v)
        escalate = should_escalate(probe, MISMATCH, source=SOURCE)
        if escalate and superseded_turn is None:
            superseded_turn = t
        else:
            # logged_mismatch branch also updated the EMA a SECOND time
            v = legacy_update(v, MISMATCH)
        traj.append(v)
    return traj, superseded_turn


def run_fixed():
    """Run the CURRENT library over the same sequence of weak contradictions."""
    traj, superseded_turn = [], None
    with MemoryLayer(":memory:") as mem:
        r = mem.write("User is a careful, risk-averse planner", domain=DOMAIN)
        item_id = r.item.id
        for t in range(1, N_TURNS + 1):
            res = mem.observe("User did something a bit out of character",
                              domain=DOMAIN, mismatch_magnitude=MISMATCH,
                              source=SOURCE)
            if res.action == "audited" and superseded_turn is None:
                superseded_turn = t
            active = mem._store.all_active(domain=DOMAIN)
            # effective volatility of the (still active) stable memory
            cur = next((i for i in active if i.id == item_id), None)
            v = cur.effective_volatility if cur else float("nan")
            traj.append(v)
    return traj, superseded_turn


def single_hit(source, mismatch=MISMATCH):
    """Effective volatility of a stable memory after ONE observation, via the
    current library. Returns (delta_from_prior)."""
    with MemoryLayer(":memory:") as mem:
        r = mem.write("User is a careful, risk-averse planner", domain=DOMAIN)
        mem.observe("one-off out-of-character moment", domain=DOMAIN,
                    mismatch_magnitude=mismatch, source=source)
        active = mem._store.all_active(domain=DOMAIN)
        cur = next((i for i in active if i.id == r.item.id), None)
        return (cur.effective_volatility if cur else float("nan"))


def turns_until(traj, thresh):
    for i, v in enumerate(traj, start=1):
        if v >= thresh:
            return i
    return None


def main():
    prior = DOMAIN_VOLATILITY[DOMAIN]
    print("=" * 72)
    print("EMA EROSION / PROPORTIONALITY TEST — stable memory under contradiction")
    print("=" * 72)
    print(f"  domain={DOMAIN} (prior V_d={prior}), mismatch={MISMATCH}")

    # ── check 1: single-hit proportionality + backward compatibility ──────────
    legacy_weak_hit = legacy_update(prior, MISMATCH)            # ignores reliability
    fixed_weak_hit = single_hit("weak_inference")               # R=0.4
    fixed_expl_hit = single_hit("explicit_statement")           # R=1.0
    legacy_expl_hit = legacy_update(prior, MISMATCH)            # legacy ignores source

    print("\n  CHECK 1 — one observation, effect on a stable memory's volatility:")
    print(f"    legacy (any source)      : {prior:.3f} -> {legacy_weak_hit:.3f} "
          f"(+{legacy_weak_hit - prior:.3f})")
    print(f"    fixed  (weak_inference)  : {prior:.3f} -> {fixed_weak_hit:.3f} "
          f"(+{fixed_weak_hit - prior:.3f})  <- low-trust should barely move")
    print(f"    fixed  (explicit)        : {prior:.3f} -> {fixed_expl_hit:.3f} "
          f"(+{fixed_expl_hit - prior:.3f})  <- reliable == legacy (back-compat)")

    weak_jump_ratio = (legacy_weak_hit - prior) / max(fixed_weak_hit - prior, 1e-9)
    backcompat_ok = abs(fixed_expl_hit - legacy_expl_hit) < 1e-6

    # ── check 2: sustained stream — erosion rate + content protection ─────────
    leg_traj, leg_super = run_legacy()
    fix_traj, fix_super = run_fixed()

    print(f"\n  CHECK 2 — {N_TURNS} sustained {SOURCE} (R={SOURCE_RELIABILITY[SOURCE]}) "
          "contradictions:")
    print(f"    {'turn':>4}{'legacy V':>12}{'fixed V':>12}")
    print("    " + "-" * 28)
    for t in range(N_TURNS):
        print(f"    {t + 1:>4}{leg_traj[t]:>12.3f}{fix_traj[t]:>12.3f}")

    leg_cross = turns_until(leg_traj, 0.5)
    fix_cross = turns_until(fix_traj, 0.5)
    print(f"\n    turns until V>=0.50 : legacy={leg_cross}  fixed={fix_cross}")
    print(f"    content superseded  : legacy={'turn ' + str(leg_super) if leg_super else 'never'}"
          f"   fixed={'turn ' + str(fix_super) if fix_super else 'never'}")

    # ── verdict ───────────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("VERDICT:")
    proportional = weak_jump_ratio >= 2.0
    slower = (fix_cross is None) or (leg_cross is not None and fix_cross > leg_cross)
    content_safe = fix_super is None
    if proportional and backcompat_ok and content_safe and slower:
        print(f"  PASS. The fix makes a low-trust observation move a stable memory "
              f"{weak_jump_ratio:.1f}x LESS than the legacy logic, while leaving "
              "reliable-source behaviour unchanged (backward compatible). Under a "
              f"sustained weak stream it erodes far slower (V>=0.5 at "
              f"{'never' if fix_cross is None else 'turn ' + str(fix_cross)} vs legacy "
              f"{'turn ' + str(leg_cross) if leg_cross else 'never'}) and the stable "
              "content is never wrongly overwritten.")
        print("  NOTE (honest): reliability scales the STEP SIZE, not the EMA's fixed "
              "point — so persistent, repeated contradictions will still raise "
              "volatility over time. That is arguably correct (sustained conflict IS "
              "evidence of volatility); the fix ensures it happens slowly and in "
              "proportion to how much the evidence can be trusted.")
    else:
        print(f"  needs review: proportional={proportional} (ratio {weak_jump_ratio:.1f}x), "
              f"backcompat={backcompat_ok}, content_safe={content_safe}, slower={slower}")


if __name__ == "__main__":
    main()
