"""
Allostatic ablation — which ingredient actually earns the recency-shift win?
===========================================================================

The allostatic escalation mode changes two things at once relative to the
current control law:

  1. V_d leaves the escalation score. Current charges volatility twice —
     once in the numerator of E_t and again in the denominator of theta_t —
     so a stable-ish domain is penalised on both sides. Allostatic charges
     it once, on the threshold only.

  2. theta_t is scaled by s(m), a time-decayed readout of how surprising
     this memory has been lately. Sustained surprise reopens the bar;
     confirms and quiet time settle it again.

Battery C showed allostatic recovering career changes that current misses,
but a two-factor change cannot tell us WHICH factor did the work. If (1)
alone is responsible then s(m) is unnecessary machinery and the honest fix
is a one-line exponent change.

This script sweeps them independently:

    E_t     = [M_t * R_t / C^alpha] * G_factor * V_d^p
    theta_t = theta_0 * (1 / V_prior) * L_t * [s(m) if mode_scale else 1]

  p = 1.0, mode_scale off  ~ the current law (see caveat below)
  p = 0.0, mode_scale on   = allostatic as shipped

Caveat on nesting: the sweep runs inside allostatic mode, which reads the
threshold off the DOMAIN PRIOR rather than the drifted per-item EMA. For a
fresh item the two are identical, but the Battery C probes deliberately
drift the EMA, so `p=1, scale=off` is close to but not exactly the current
law. The true current law is reported as a separate reference row.

Scoring — a config is only interesting if it does BOTH:
  * holds Battery A at 20/20 under real priors, and keeps real > flat >= swap
    (the negative control that gates every VoltMem claim), and
  * wins the recency-shift diagnostic without flipping the very-stable
    negative control.

Run:
    .venv/bin/python experiments/allostatic_ablation.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import contextlib                                             # noqa: E402

import voltmem.scoring as sc                                  # noqa: E402
from voltmem_eval import (                                    # noqa: E402
    RECENCY_SHIFT_PROBES,
    SLOW_BURN_PROBES,
    _run_recency_probe,
    _run_slow_burn,
    run_escalation,
)

V_EXPS = [0.0, 0.25, 0.5, 0.75, 1.0]
MODE_SCALES = [True, False]

PROBE_KEYS = [
    "noisy-then-shift professional",
    "noisy-then-shift preference",
    "quiet-then-shift professional",
]
PROBE_LABELS = {
    "noisy-then-shift professional": "prof/noisy",
    "noisy-then-shift preference": "pref/noisy",
    "quiet-then-shift professional": "prof/quiet",
}


def battery_a(mode, v_exp=None, mode_scale=None):
    out = {}
    for profile in ("real", "flat", "swap"):
        correct, n, _ = run_escalation(
            profile, escalation_mode=mode, v_exp=v_exp, mode_scale=mode_scale)
        out[profile] = correct / n
    out["n"] = n
    out["causal"] = (
        out["real"] > out["flat"]
        and out["real"] >= out["swap"]
        and out["real"] > 0.5
    )
    return out


def battery_c(mode, v_exp=None, mode_scale=None):
    by_name = {p["name"]: p for p in RECENCY_SHIFT_PROBES}
    out = {}
    for key in PROBE_KEYS:
        got, _action = _run_recency_probe(
            by_name[key], mode, v_exp=v_exp, mode_scale=mode_scale)
        out[key] = got
    return out


def evaluate(label, mode, v_exp=None, mode_scale=None):
    a = battery_a(mode, v_exp, mode_scale)
    c = battery_c(mode, v_exp, mode_scale)
    # Expected: entrenched professional reopens, very-stable preference does not.
    diagnostic = (
        c["noisy-then-shift professional"] == "U"
        and c["quiet-then-shift professional"] == "U"
    )
    control_held = c["noisy-then-shift preference"] == "R"
    return {
        "label": label,
        "a": a,
        "c": c,
        "diagnostic": diagnostic,
        "control_held": control_held,
        "keep": a["real"] == 1.0 and a["causal"] and diagnostic and control_held,
    }


def main():
    print("=" * 84)
    print("ALLOSTATIC ABLATION — V_d exponent in E_t  x  s(m) scaling on theta_t")
    print("=" * 84)

    results = [evaluate("homeostatic (reference law)", "homeostatic")]
    for mode_scale in MODE_SCALES:
        for v_exp in V_EXPS:
            label = f"allostatic  p={v_exp:<4}  s(m)={'on ' if mode_scale else 'off'}"
            results.append(
                evaluate(label, "allostatic", v_exp=v_exp, mode_scale=mode_scale))

    header = (
        f"  {'config':<34}{'A real':>8}{'A flat':>8}{'A swap':>8}{'causal':>8}"
        f"{'prof/noisy':>12}{'pref/noisy':>12}{'prof/quiet':>12}{'keep':>7}"
    )
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        a, c = r["a"], r["c"]
        print(
            f"  {r['label']:<34}{a['real']:>7.0%}{a['flat']:>8.0%}{a['swap']:>8.0%}"
            f"{str(a['causal']):>8}"
            f"{c['noisy-then-shift professional']:>12}"
            f"{c['noisy-then-shift preference']:>12}"
            f"{c['quiet-then-shift professional']:>12}"
            f"{('yes' if r['keep'] else 'no'):>7}"
        )

    print("\n  A columns = Battery A accuracy (20 probes) under real / flat / swap")
    print("  priors. 'causal' = real > flat and real >= swap. Recency columns are")
    print("  U (audited, memory updated) or R (retained).")
    print("  Wanted: prof/noisy=U, prof/quiet=U, pref/noisy=R (very-stable holds).")

    # ── what the sweep implies ────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("READING")
    print("=" * 84)

    keepers = [r for r in results if r["keep"]]
    scale_on = {r["label"]: r for r in results if "s(m)=on" in r["label"]}
    scale_off = {r["label"]: r for r in results if "s(m)=off" in r["label"]}

    print(f"  configs meeting every bar: {len(keepers)}")
    for r in keepers:
        print(f"    {r['label']}")

    # Does s(m) change any outcome at matched p?
    differs = []
    for v_exp in V_EXPS:
        on = next(r for r in scale_on.values() if f"p={v_exp:<4}" in r["label"])
        off = next(r for r in scale_off.values() if f"p={v_exp:<4}" in r["label"])
        if on["c"] != off["c"] or on["a"]["real"] != off["a"]["real"]:
            differs.append(v_exp)

    print()
    if differs:
        print(f"  s(m) changes an outcome at p in {differs} — the surprise term is")
        print("  doing work the exponent alone does not.")
    else:
        print("  s(m) changes NO outcome at any p on this suite. On these probes the")
        print("  recency-shift win comes from removing the double V_d penalty, not")
        print("  from mode-switching. s(m) is unfalsified here, not validated — it")
        print("  needs a probe where surprise accumulates without an explicit,")
        print("  high-M statement to ride the theta cap.")

    # Largest p that still recovers the diagnostic — how much V_d can stay in E_t.
    ok_ps = [
        v for v in V_EXPS
        if any(
            r["keep"] and f"p={v:<4}" in r["label"]
            for r in results
        )
    ]
    if ok_ps:
        print(f"\n  V_d exponent still passing everything: up to p={max(ok_ps)}.")
        print("  Higher p keeps more domain sensitivity in the score, which should")
        print("  degrade more gracefully when the domain label is wrong — the next")
        print("  thing to test (classification corpus, ~84% heuristic accuracy).")

    surprise_sensitivity()


# ── s(m) tuning sensitivity ───────────────────────────────────────────────────
# Battery C could not exercise s(m) at all (explicit M=0.90 rides the θ-cap).
# Battery D can: weak_inference only, so nothing but s(m) can reopen the bar.
# There, s(m) drives θ from 0.500 down to 0.2941 against E_t = 0.2940 — it
# converges to within 2e-4 ABOVE the trigger and never crosses. So the question
# is not "does s(m) work" but "are S_MIN and the half-life calibrated".
#
# This sweep answers that honestly by reporting the FULL cost of each setting,
# not just whether the target probe fires: Battery A under all three priors
# (the causal control), Battery C, and all four Battery D probes including the
# three negative controls.

S_MINS = [0.05, 0.10, 0.15, 0.20, 0.25]
HALFLIVES = [14.0, 30.0]

BURN_TARGET = "weak burn, daily"


@contextlib.contextmanager
def surprise_tuning(s_min, halflife):
    old = (sc.S_MIN, sc.SURPRISE_HALFLIFE_DAYS)
    sc.S_MIN, sc.SURPRISE_HALFLIFE_DAYS = s_min, halflife
    try:
        yield
    finally:
        sc.S_MIN, sc.SURPRISE_HALFLIFE_DAYS = old


def surprise_sensitivity():
    print("\n" + "=" * 84)
    print("s(m) TUNING SENSITIVITY — does the surprise term work at any calibration?")
    print("=" * 84)

    controls = [p for p in SLOW_BURN_PROBES if p["name"] != BURN_TARGET]
    target = next(p for p in SLOW_BURN_PROBES if p["name"] == BURN_TARGET)

    header = (
        f"  {'S_MIN':>6}{'half-life':>11}{'A real':>8}{'A flat':>8}{'A swap':>8}"
        f"{'C ok':>7}{'weak burn':>11}{'controls':>10}{'verdict':>10}"
    )
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))

    for halflife in HALFLIVES:
        for s_min in S_MINS:
            with surprise_tuning(s_min, halflife):
                a = battery_a("allostatic")
                c = battery_c("allostatic")
                c_ok = (
                    c["noisy-then-shift professional"] == "U"
                    and c["quiet-then-shift professional"] == "U"
                    and c["noisy-then-shift preference"] == "R"
                )
                fired = _run_slow_burn(target, "allostatic")
                held = all(
                    _run_slow_burn(p, "allostatic") is None for p in controls)

            good = (
                a["real"] == 1.0 and a["causal"] and c_ok
                and fired is not None and held)
            print(
                f"  {s_min:>6.2f}{halflife:>11.0f}{a['real']:>8.0%}{a['flat']:>8.0%}"
                f"{a['swap']:>8.0%}{str(c_ok):>7}"
                f"{('never' if fired is None else f'turn {fired}'):>11}"
                f"{('held' if held else 'BROKE'):>10}"
                f"{('ok' if good else '-'):>10}"
            )

    print("\n  'weak burn' = turn the accumulated-weak-evidence job change is caught")
    print("  (never = missed). 'controls' = the very-stable, confirmed-in-between,")
    print("  and spread-over-months probes all correctly stayed shut.")
    print(f"  Shipped setting is S_MIN={sc.S_MIN}, half-life={sc.SURPRISE_HALFLIFE_DAYS:.0f}d.")


if __name__ == "__main__":
    main()
