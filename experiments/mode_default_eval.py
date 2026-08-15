"""
Default-mode decision: homeostatic vs composite (allostatic as reference)
=========================================================================

Batteries A–E measure the overwrite law given a perfect router.
Battery F measures remember() on the shipped keyword path, where a false
merge can make a more plastic law more expensive.

Promote composite if it wins Battery C, matches homeostatic on E (false
updates), and does not raise F's irreversible count.

Run:
    .venv/bin/python experiments/mode_default_eval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from end_to_end_eval import run_config                       # noqa: E402
from label_noise_eval import aggregate, build_confusion      # noqa: E402
from voltmem_eval import (                                   # noqa: E402
    RECENCY_SHIFT_PROBES,
    SLOW_BURN_PROBES,
    _run_recency_probe,
    _run_slow_burn,
    run_escalation,
)

MODES = ("homeostatic", "composite", "allostatic")


def battery_a():
    rows = {}
    for mode in MODES:
        c, n, detail = run_escalation("real", escalation_mode=mode)
        failed = [r for r in detail if not r[3]]
        rows[mode] = {"correct": c, "n": n, "failed": failed}
    return rows


def battery_c():
    rows = []
    for probe in RECENCY_SHIFT_PROBES:
        got = {}
        for mode in MODES:
            ur, action = _run_recency_probe(probe, mode)
            got[mode] = (ur, action)
        want_h = probe["want_homeostatic"]
        want_a = probe["want_allostatic"]
        rows.append({
            "name": probe["name"],
            "want_homeostatic": want_h,
            "want_allostatic": want_a,
            "got": got,
            "ok": {
                "homeostatic": got["homeostatic"][0] == want_h,
                "composite": got["composite"][0] == want_a,
                "allostatic": got["allostatic"][0] == want_a,
            },
        })
    return rows


def battery_d():
    rows = []
    for probe in SLOW_BURN_PROBES:
        got = {}
        for mode in MODES:
            turn = _run_slow_burn(probe, mode)
            got[mode] = "never" if turn is None else f"turn {turn}"
        rows.append({"name": probe["name"], "got": got})
    return rows


def battery_e():
    conf, err_rate, _ = build_confusion()
    rows = {}
    for rate_label, rate in (("real", None), ("50%", 0.50)):
        rows[rate_label] = {
            mode: aggregate(mode, rate, "structured", conf, err_rate)
            for mode in MODES
        }
    return rows


def battery_f():
    rows = {}
    for labels in ("oracle", "shipped"):
        rows[labels] = {
            mode: run_config(None, oracle=(labels == "oracle"),
                             escalation_mode=mode)
            for mode in MODES
        }
    return rows


def recommend(a, c, e, f):
    c_win = all(row["ok"]["composite"] for row in c)
    e_real = e["real"]
    e_hold = (
        e_real["composite"]["false_update"]
        <= e_real["homeostatic"]["false_update"] + 1e-9
        and e_real["composite"]["acc"]
        >= e_real["homeostatic"]["acc"] - 1e-9
    )
    a_ok = a["composite"]["correct"] == a["composite"]["n"]
    f_ship = f["shipped"]
    f_ok = (
        f_ship["composite"]["irreversible"]
        <= f_ship["homeostatic"]["irreversible"]
    )
    promote = c_win and e_hold and a_ok and f_ok
    return {
        "promote": promote,
        "c_win": c_win,
        "e_hold": e_hold,
        "a_ok": a_ok,
        "f_ok": f_ok,
    }


def _fmt_fail(failed):
    if not failed:
        return "—"
    return ", ".join(f"{r[0]} want={r[1]} got={r[2]}" for r in failed)


def main():
    print("=" * 84)
    print("DEFAULT MODE — homeostatic vs composite  (allostatic = reference)")
    print("=" * 84)

    print("\nBattery A — selective updating, real priors, labels handed in")
    a = battery_a()
    for mode in MODES:
        r = a[mode]
        print(f"  {mode:<12} {r['correct']}/{r['n']}  "
              f"fails={_fmt_fail(r['failed'])}")

    print("\nBattery C — recency-shift (composite should match allostatic wants)")
    c = battery_c()
    for row in c:
        bits = "  ".join(
            f"{mode[0:4]}={row['got'][mode][0]}"
            + ("" if row["ok"][mode] else " XX")
            for mode in MODES
        )
        print(f"  {row['name']:<32} home_want={row['want_homeostatic']}  "
              f"allo_want={row['want_allostatic']}  {bits}")

    print("\nBattery D — live weak stream (should stay shut in every mode)")
    d = battery_d()
    for row in d:
        bits = "  ".join(f"{mode[0:4]}={row['got'][mode]}" for mode in MODES)
        print(f"  {row['name']:<28} {bits}")

    print("\nBattery E — label noise, structured (false update = irreversible)")
    e = battery_e()
    for rate_label in ("real", "50%"):
        print(f"  {rate_label} classifier error:")
        for mode in MODES:
            r = e[rate_label][mode]
            print(f"    {mode:<12} acc={r['acc']:.1%}  "
                  f"FU={r['false_update']:.2f}  MU={r['missed_update']:.2f}")

    print("\nBattery F — remember() keyword path (shipped matcher, no verifier)")
    f = battery_f()
    for labels in ("oracle", "shipped"):
        print(f"  {labels} labels:")
        for mode in MODES:
            r = f[labels][mode]
            print(f"    {mode:<12} routed={r['routed']:.1%}  "
                  f"overall={r['total_correct']:.1%}  "
                  f"irrev={r['irreversible']}  recov={r['recoverable']}  "
                  f"FU={r['false_update']}  merge={r['false_merge']}")

    rec = recommend(a, c, e, f)
    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    print(f"  A no regression:          {'yes' if rec['a_ok'] else 'NO'}")
    print(f"  C composite wins career:  {'yes' if rec['c_win'] else 'NO'}")
    print(f"  E insurance holds:        {'yes' if rec['e_hold'] else 'NO'}")
    print(f"  F irrev does not rise:    {'yes' if rec['f_ok'] else 'NO'}")
    if rec["promote"]:
        print("\n  => recommend composite as the default.")
        print("     It takes the career-change win without extra false updates")
        print("     on mixed labels, and the shipped matcher does not pay more")
        print("     irreversible cost.")
    else:
        print("\n  => keep homeostatic as the default.")
        if not rec["c_win"]:
            print("     Composite missed a recency-shift case it was meant to take.")
        if not rec["e_hold"]:
            print("     Composite lost Battery E insurance (more false updates).")
        if not rec["f_ok"]:
            print("     Composite raised irreversible errors on remember().")
            home_i = f["shipped"]["homeostatic"]["irreversible"]
            comp_i = f["shipped"]["composite"]["irreversible"]
            print(f"     shipped irrev homeostatic={home_i} composite={comp_i}")
        if not rec["a_ok"]:
            print("     Composite regresses Battery A.")
    return rec


if __name__ == "__main__":
    main()
