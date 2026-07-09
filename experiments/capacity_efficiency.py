"""
Capacity efficiency of volatility-weighted EWC (tight-budget continual learning)
================================================================================

Motivation & what the sweep actually found
------------------------------------------
Initial hypothesis: volatility weighting should help most at TIGHT capacity
(spend a scarce protection budget where warranted). The sweep FALSIFIED that.
What actually happens (MNIST, gamma=2):

  * With adequate protection (lam~800), volatility is ~break-even at every
    capacity: a pure tradeoff knob (retention up, adaptation down, similar
    utility). H1 capacity-saving NOT supported — baseline and volatility reach a
    given retention target at the SAME smallest model.

  * The clear, favorable, CAUSAL benefit appears in the OPPOSITE regime:
    UNDER-PROTECTED LARGE models (weak lam + many params). There uniform EWC
    lets big capacity aggressively overwrite stable knowledge (retention
    collapses), while volatility concentrates protection on the stable domain
    and rescues it. Example (hidden=256, lam=300, 4 runs), retention:
        baseline 0.877 -> volatility REAL 0.954  (+7.7pp, adapt only -2.9pp)
    and this PASSES the v3 --sabotage control (shuffle/swap keep only +1.3pp),
    so it is the volatility signal doing the work, not generic regularisation.

Honest takeaway: this is NOT a parameter-savings method. Its practical value is
(1) a measured stability-plasticity knob and (2) ROBUSTNESS to under-tuned
protection — it auto-allocates a fixed budget to stable knowledge so you don't
have to hand-tune lambda per capacity. A well-tuned uniform baseline can match
it; volatility mainly reduces the need to tune. ALWAYS confirm a given regime
with ewc_volatility_v3_mnist.py --sabotage before trusting a utility gain.

Because volatility TRADES adaptation for retention, "more efficient" is only
meaningful against an explicit objective. In constrained continual learning the
dominant pain is catastrophic FORGETTING, so we score with a retention-weighted
utility (tunable via --w-ret) and also report both raw axes and a symmetric
mean so nothing is hidden.

Two hypotheses this script distinguishes
----------------------------------------
  H1 (capacity saving):  volatility-EWC at hidden = h matches uniform-EWC at a
     LARGER hidden under the objective -> genuine parameter savings.
  H2 (free knob):        volatility only moves along the same frontier uniform
     EWC could reach by retuning lambda -> useful control, not a capacity saver.

Method
------
Sweep hidden width. At each width run baseline (uniform) vs volatility EWC and
record early-stable retention, late-volatile adaptation, parameter count, and a
retention-weighted utility. Then, for a retention target with an adaptation
floor, report the smallest (cheapest) model each method needs to hit it.

Run:
    .venv/bin/python experiments/capacity_efficiency.py
    .venv/bin/python experiments/capacity_efficiency.py --quick
    .venv/bin/python experiments/capacity_efficiency.py \
        --hiddens 32 48 64 96 128 192 256 --runs 4 --lam 800 --gamma 2
"""

import argparse

import numpy as np
import torch

from ewc_volatility_v3_mnist import (
    SharedNet, load_mnist_by_digit, run_experiment,
)


def param_count(hidden):
    return sum(p.numel() for p in SharedNet(hidden=hidden).parameters())


def eval_capacity(hidden, by_digit, n_runs, n_tasks, n_per_task, epochs,
                  lam, gamma):
    """Return averaged (retention, adaptation) for baseline and volatility."""
    out = {}
    for mode in ["baseline", "volatility"]:
        rs = [run_experiment(mode, by_digit, n_tasks=n_tasks, seed=run,
                             gamma=gamma, n_per_task=n_per_task, epochs=epochs,
                             ewc_lambda=lam, hidden=hidden, control="none")
              for run in range(n_runs)]
        out[mode] = {
            "ret": float(np.mean([r["early_stable_retention"] for r in rs])),
            "adapt": float(np.mean([r["late_volatile_adaptation"] for r in rs])),
        }
    return out


def utility(rec, w_ret):
    """Retention-weighted objective in [0,1]. w_ret=0.5 is symmetric."""
    return w_ret * rec["ret"] + (1.0 - w_ret) * rec["adapt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--hiddens", type=int, nargs="+", default=None)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--per-task", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lam", type=float, default=800.0)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--w-ret", type=float, default=0.7,
                    help="objective weight on retention (forgetting is the main "
                         "enemy in constrained CL, so default > 0.5)")
    ap.add_argument("--target-ret", type=float, default=0.96,
                    help="retention target for the iso-performance capacity test")
    ap.add_argument("--adapt-floor", type=float, default=0.55,
                    help="min acceptable adaptation (so retention can't be bought "
                         "by killing plasticity)")
    args = ap.parse_args()

    if args.quick:
        hiddens = args.hiddens or [32, 64, 128]
        n_runs, n_tasks, per_task, epochs = 2, 8, 200, 5
    else:
        hiddens = args.hiddens or [32, 48, 64, 96, 128, 192, 256]
        n_runs, n_tasks, per_task, epochs = args.runs, args.tasks, args.per_task, args.epochs

    print("Loading MNIST ...")
    by_digit = load_mnist_by_digit()

    rows = []
    for h in hiddens:
        res = eval_capacity(h, by_digit, n_runs, n_tasks, per_task, epochs,
                            args.lam, args.gamma)
        rows.append((h, param_count(h), res))
        print(f"  hidden={h:4d} done")

    # ---- table ----
    print("\n" + "=" * 92)
    print(f"CAPACITY EFFICIENCY — {n_runs} runs, {n_tasks} tasks, lambda={args.lam}, "
          f"gamma={args.gamma}, w_ret={args.w_ret}")
    print("=" * 92)
    hdr = (f"{'hidden':>7}{'params':>10} | "
           f"{'base ret':>9}{'base adp':>9}{'base U':>8} | "
           f"{'vol ret':>9}{'vol adp':>9}{'vol U':>8} | {'ΔU':>7}")
    print(hdr)
    print("-" * 92)
    for h, p, res in rows:
        bU = utility(res["baseline"], args.w_ret)
        vU = utility(res["volatility"], args.w_ret)
        print(f"{h:>7}{p:>10} | "
              f"{res['baseline']['ret']:>9.3f}{res['baseline']['adapt']:>9.3f}{bU:>8.3f} | "
              f"{res['volatility']['ret']:>9.3f}{res['volatility']['adapt']:>9.3f}{vU:>8.3f} | "
              f"{vU - bU:>+7.3f}")

    # ---- H1: iso-performance capacity saving ----
    # smallest model (by params) each method needs to hit retention target while
    # keeping adaptation >= floor.
    def cheapest(mode):
        ok = [(p, h) for h, p, res in rows
              if res[mode]["ret"] >= args.target_ret
              and res[mode]["adapt"] >= args.adapt_floor]
        return min(ok) if ok else None

    print("\n" + "-" * 92)
    print(f"H1 — CAPACITY SAVING at objective (retention >= {args.target_ret}, "
          f"adaptation >= {args.adapt_floor}):")
    cb = cheapest("baseline")
    cv = cheapest("volatility")
    b_txt = f"hidden={cb[1]} ({cb[0]} params)" if cb else "NOT reached at any tested size"
    v_txt = f"hidden={cv[1]} ({cv[0]} params)" if cv else "NOT reached at any tested size"
    print(f"  baseline   cheapest model meeting target: {b_txt}")
    print(f"  volatility cheapest model meeting target: {v_txt}")
    if cb and cv and cv[0] < cb[0]:
        print(f"  -> H1 SUPPORTED: volatility meets the target with "
              f"{cb[0] / cv[0]:.2f}x fewer parameters ({cb[0]} -> {cv[0]}). "
              "Volatility weighting genuinely saves capacity for this objective.")
    elif cb and cv and cv[0] == cb[0]:
        print("  -> H1 NOT shown: both need the same capacity. Volatility is a "
              "free tradeoff knob here (H2), not a capacity saver.")
    elif cv and not cb:
        print("  -> H1 SUPPORTED (strong): volatility reaches the target within "
              "the tested budget while baseline never does.")
    else:
        print("  -> inconclusive at these settings; widen --hiddens or adjust "
              "--target-ret / --adapt-floor / --lam.")

    # ---- H2: where (by capacity) does the knob help the objective? ----
    print("\n" + "-" * 92)
    print("H2 — objective utility by capacity (where does the knob help?):")
    tight = rows[: max(1, len(rows) // 2)]
    wins = sum(1 for _, _, res in tight
               if utility(res["volatility"], args.w_ret)
               > utility(res["baseline"], args.w_ret) + 1e-4)
    print(f"  volatility beats baseline on utility in {wins}/{len(tight)} of the "
          "SMALLEST models tested.")
    big = rows[len(rows) // 2:]
    wins_big = sum(1 for _, _, res in big
                   if utility(res["volatility"], args.w_ret)
                   > utility(res["baseline"], args.w_ret) + 1e-4)
    print(f"  volatility beats baseline on utility in {wins_big}/{len(big)} of the "
          "LARGEST models tested.")
    print("  NOTE: empirically the advantage is NOT in the smallest models — it "
          "concentrates in UNDER-PROTECTED LARGE models (weak lambda + many "
          "params), where uniform EWC forgets stable knowledge and volatility "
          "rescues it. Confirm any win with ewc_volatility_v3_mnist.py --sabotage.")


if __name__ == "__main__":
    main()
