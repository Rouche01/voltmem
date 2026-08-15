"""
Battery E — label-noise robustness: homeostatic vs allostatic under a wrong domain
=============================================================================

Every VoltMem escalation decision is conditioned on the domain label, and in
production that label comes from a classifier that is ~84% accurate. Battery A
hands each probe its true domain, so it measures the control law given a
perfect classifier. This battery measures it given the real one.

Why this is the deciding test for allostatic mode:

  homeostatic: E_t = residual * V_d      theta_t = theta_0 / V_d
               -> escalates when residual * V_d^2 > theta_0   (QUADRATIC in V_d)
  allostatic:  E_t = residual            theta_t = theta_0 / V_d * s(m)
               -> escalates when residual * V_d   > theta_0   (LINEAR in V_d)

So on paper allostatic should be LESS sensitive to a wrong V_d, not more. But
allostatic is also more plastic at mid-range V_d (it drops the V_d discount on
the score), so it may convert label errors into a different, worse error type.
Those two effects pull in opposite directions and the net is an empirical
question. Battery A's flat-prior number (65% vs homeostatic's 75%) conflates them:
"flat" collapses every prior to 0.5, which is a systematic bias shift, not the
random, structured error a classifier actually makes.

Error structure is taken from the real classifier, not invented. The confusion
matrix over tests/fixtures/classification_corpus.json is genuinely hostile:

    core_preference  (V=0.08) -> current_task (V=0.90)   x2
    personality_trait(V=0.05) -> location     (V=0.60)   x1
    biographical     (V=0.10) -> emotional_context(V=0.80) x1
    transient_fact   (V=0.95) -> stated_preference(V=0.45) x10

Mean |V_gold - V_pred| when it misfires is 0.355, max 0.82 — a mislabel is not
a small perturbation, it often inverts the stability class entirely.

Two error types are counted separately, because they do not cost the same:

  false update  — expected RETAIN, got UPDATE. A true, stable fact was
                  overwritten. Irreversible: the old content is superseded.
  missed update — expected UPDATE, got RETAIN. A stale fact survived.
                  Recoverable: the next confirmation or mismatch can fix it.

A mode that trades missed updates for false updates is worse even at equal
accuracy. That asymmetry, not the headline number, should drive the decision.

Ground truth (U/R) always follows the TRUE domain. A classifier mistake does
not change whether the memory ought to be updated.

Run:
    .venv/bin/python experiments/label_noise_eval.py
"""

import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voltmem import MemoryLayer                                # noqa: E402
from voltmem.classification_eval import evaluate_classifier    # noqa: E402
from voltmem.classifiers import resolve_classifier             # noqa: E402
from voltmem.domains import DOMAIN_VOLATILITY                  # noqa: E402
from voltmem_eval import (                                     # noqa: E402
    ESCALATION_PROBES,
    _got_ur,
    volatility_profile,
)

SEEDS = 80
RATES = [0.0, 0.10, 0.20, 0.30, 0.50]
STRUCTURES = ("structured", "uniform")
MODES = ("homeostatic", "allostatic")

ALL_DOMAINS = sorted(DOMAIN_VOLATILITY)


def build_confusion():
    """Real confusion distribution + per-domain error rate from the classifier."""
    clf = resolve_classifier(None)
    result = evaluate_classifier(lambda t: clf.classify_domain(t))
    conf, err_rate = {}, {}
    for gold, counter in result.confusion.items():
        total = sum(counter.values())
        wrong = [
            (pred, cnt) for pred, cnt in counter.items()
            if pred != gold and pred in DOMAIN_VOLATILITY
        ]
        err_rate[gold] = (sum(c for _, c in wrong) / total) if total else 0.0
        if wrong:
            conf[gold] = wrong
    return conf, err_rate, result


def sample_wrong(gold, rng, structure, conf):
    """A plausible wrong label. None means 'this domain is never confused'."""
    if structure == "structured":
        opts = conf.get(gold)
        if not opts:
            return None
        labels = [lab for lab, _ in opts]
        weights = [w for _, w in opts]
        return rng.choices(labels, weights=weights, k=1)[0]
    return rng.choice([d for d in ALL_DOMAINS if d != gold])


def run_trial(mode, rate, structure, conf, err_rate, rng):
    """One pass over Battery A's one-shot probes with labels perturbed.

    The same wrong label is used for the write and the observe: a fact that
    reads as a mood when stored reads as a mood when contradicted. (Labels
    disagreeing between write and observe is a different failure — the
    observation never finds the stored item and silently inserts a duplicate —
    which hits both modes identically and is reported separately.)
    """
    correct = false_update = missed_update = mislabeled = 0
    for (domain, base, obs, mm, src, expected, _note) in ESCALATION_PROBES:
        p = err_rate.get(domain, 0.0) if rate is None else rate
        used = domain
        if p > 0.0 and rng.random() < p:
            wrong = sample_wrong(domain, rng, structure, conf)
            if wrong is not None:
                used = wrong
                mislabeled += 1
        with MemoryLayer(":memory:", escalation_mode=mode) as mem:
            mem.write(base, domain=used)
            res = mem.observe(
                obs, domain=used, mismatch_magnitude=mm, source=src)
        got = _got_ur(res.action)
        if got == expected:
            correct += 1
        elif expected == "R":
            false_update += 1
        else:
            missed_update += 1
    return correct, false_update, missed_update, mislabeled


def aggregate(mode, rate, structure, conf, err_rate, seeds=SEEDS):
    n_probes = len(ESCALATION_PROBES)
    deterministic = (rate == 0.0)
    runs = 1 if deterministic else seeds
    accs, fus, mus, mls = [], [], [], []
    for seed in range(runs):
        rng = random.Random(seed * 7919 + 13)
        correct, fu, mu, ml = run_trial(
            mode, rate, structure, conf, err_rate, rng)
        accs.append(correct / n_probes)
        fus.append(fu)
        mus.append(mu)
        mls.append(ml)
    return {
        "acc": statistics.mean(accs),
        "sd": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
        "false_update": statistics.mean(fus),
        "missed_update": statistics.mean(mus),
        "mislabeled": statistics.mean(mls),
    }


def duplicate_insert_rate(conf, err_rate, seeds=SEEDS):
    """How often disagreeing write/observe labels silently insert a duplicate.

    Mode-independent — escalation is never reached — but it bounds how much of
    the classifier's error budget the control law can even be blamed for.
    """
    hits = trials = 0
    for seed in range(seeds):
        rng = random.Random(seed * 104729 + 7)
        for (domain, base, obs, mm, src, _expected, _note) in ESCALATION_PROBES:
            p = err_rate.get(domain, 0.0)
            w_label = o_label = domain
            if p > 0 and rng.random() < p:
                w = sample_wrong(domain, rng, "structured", conf)
                if w:
                    w_label = w
            if p > 0 and rng.random() < p:
                w = sample_wrong(domain, rng, "structured", conf)
                if w:
                    o_label = w
            trials += 1
            if w_label != o_label:
                with MemoryLayer(":memory:") as mem:
                    mem.write(base, domain=w_label)
                    res = mem.observe(obs, domain=o_label,
                                      mismatch_magnitude=mm, source=src)
                if res.action == "inserted":
                    hits += 1
    return hits / trials if trials else 0.0


def main():
    conf, err_rate, cls = build_confusion()

    print("=" * 88)
    print("BATTERY E — LABEL NOISE: does allostatic survive a wrong domain label?")
    print("=" * 88)
    print(f"  classifier: {cls.correct}/{cls.n} = {cls.accuracy:.1%} on the "
          "labeled corpus")
    print(f"  domains it ever mislabels: {len(conf)} of {len(ALL_DOMAINS)}")
    print(f"  one-shot probes per trial: {len(ESCALATION_PROBES)}   "
          f"seeds per point: {SEEDS}")

    print("\n  Empirical per-domain error rates used by the 'real' row:")
    for domain in sorted(err_rate):
        if err_rate[domain] > 0:
            print(f"    {domain:<22} {err_rate[domain]:>5.1%}")

    with volatility_profile("real"):
        for structure in STRUCTURES:
            label = (
                "STRUCTURED — wrong labels drawn from the real confusion matrix"
                if structure == "structured"
                else "UNIFORM — wrong label uniform over all other domains "
                     "(unstructured stress)"
            )
            print("\n" + "-" * 88)
            print(f"  {label}")
            print("-" * 88)
            rows = []
            for rate in RATES:
                cur = aggregate("homeostatic", rate, structure, conf, err_rate)
                allo = aggregate("allostatic", rate, structure, conf, err_rate)
                comp = aggregate("composite", rate, structure, conf, err_rate)
                rows.append((f"{rate:.0%}", cur, allo, comp))
            if structure == "structured":
                cur = aggregate("homeostatic", None, structure, conf, err_rate)
                allo = aggregate("allostatic", None, structure, conf, err_rate)
                comp = aggregate("composite", None, structure, conf, err_rate)
                rows.append(("real", cur, allo, comp))

            header = (
                f"  {'noise':>8}{'mislab':>7} |"
                f"{'home acc':>9}{'fu':>6} |"
                f"{'allo acc':>10}{'fu':>6} |"
                f"{'comp acc':>10}{'fu':>6}"
            )
            print(header)
            print("  " + "-" * (len(header) - 2))
            for name, cur, allo, comp in rows:
                print(
                    f"  {name:>8}{cur['mislabeled']:>7.1f} |"
                    f"{cur['acc']:>8.1%}{cur['false_update']:>6.2f} |"
                    f"{allo['acc']:>9.1%}{allo['false_update']:>6.2f} |"
                    f"{comp['acc']:>9.1%}{comp['false_update']:>6.2f}"
                )
            if structure == "structured":
                structured_rows = rows

        print("\n  'mislab' = probes given a wrong label per trial (of "
              f"{len(ESCALATION_PROBES)}).")
        print("  'false upd' = stable fact wrongly overwritten (irreversible).")
        print("  'missed'    = real change wrongly ignored (recoverable).")

        dup = duplicate_insert_rate(conf, err_rate)

    # ── verdict ───────────────────────────────────────────────────────────────
    real_cur = next(c for n, c, _a, _p in structured_rows if n == "real")
    real_allo = next(a for n, _c, a, _p in structured_rows if n == "real")
    real_comp = next(p for n, _c, _a, p in structured_rows if n == "real")
    worst_cur = next(c for n, c, _a, _p in structured_rows if n == "50%")
    worst_allo = next(a for n, _c, a, _p in structured_rows if n == "50%")
    worst_comp = next(p for n, _c, _a, p in structured_rows if n == "50%")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    print(f"  At the classifier's real error rate:")
    print(f"    homeostatic {real_cur['acc']:.1%}  "
          f"false updates {real_cur['false_update']:.2f}/trial")
    print(f"    allostatic  {real_allo['acc']:.1%}  "
          f"false updates {real_allo['false_update']:.2f}/trial")
    print(f"    composite   {real_comp['acc']:.1%}  "
          f"false updates {real_comp['false_update']:.2f}/trial")
    print(f"  At a forced 50% mislabel rate (structured):")
    print(f"    homeostatic {worst_cur['acc']:.1%}  "
          f"false updates {worst_cur['false_update']:.2f}/trial")
    print(f"    allostatic  {worst_allo['acc']:.1%}  "
          f"false updates {worst_allo['false_update']:.2f}/trial")
    print(f"    composite   {worst_comp['acc']:.1%}  "
          f"false updates {worst_comp['false_update']:.2f}/trial")

    acc_ok = real_allo["acc"] >= real_cur["acc"] - 0.02
    harm_ok = real_allo["false_update"] <= real_cur["false_update"] + 1e-9
    gate_ok = real_comp["false_update"] <= real_cur["false_update"] + 1e-9
    print()
    if acc_ok and harm_ok:
        print("  => allostatic is NOT more fragile to label noise. It holds accuracy")
        print("     and does not convert label errors into extra false updates.")
        print("     The flat-prior gap in Battery A was a bias artifact, not")
        print("     label sensitivity. Safe to consider as the default.")
    elif harm_ok:
        print("  => allostatic loses some accuracy but does NOT increase the")
        print("     expensive error type. Acceptable if the missed updates are")
        print("     recovered by later evidence — check the cumulative probes.")
    else:
        print("  => allostatic converts label errors into MORE false updates —")
        print("     stable facts destroyed because the domain was misread.")
        print("     Do not promote to default.")
    if gate_ok:
        print("  => composite keeps homeostatic's false-update rate on Battery E")
        print("     (fresh items stay homeostatic unless the statement is explicit).")
    else:
        print("  => composite still overwrites more than homeostatic — the gate is")
        print("     not earning Battery E insurance. Do not ship it.")

    print(f"\n  Separately, mode-independent: when the write and observe labels")
    print(f"  disagree, the observation cannot find the stored item and inserts a")
    print(f"  silent duplicate instead. That happens on {dup:.1%} of probes at the")
    print("  real error rate, and no escalation law can fix it — it is a routing")
    print("  failure upstream of the decision.")


if __name__ == "__main__":
    main()
