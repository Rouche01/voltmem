"""
LLM-agent memory benchmark: volatility-aware updating vs naive memory policies
=============================================================================

The other experiments test the continual-learning math (EWC) and the library's
unit behaviour. This one asks the product question directly:

    Over a long, noisy user history, which memory policy answers
    "what is the user's CURRENT X?" correctly most often?

Why this is a real test and not a demo
--------------------------------------
A memory layer for an LLM agent faces two failure modes at once:

  * STALE  — it keeps an old value after the truth has changed (bad on volatile
    facts like current task / mood / project), and
  * CORRUPTED — it overwrites a durable fact because of a confident-sounding but
    wrong observation (bad on stable facts like home city / communication style).

Naive policies are forced onto ONE side of that tradeoff. VoltMem's claim is that
scaling the update threshold by per-domain volatility escapes it: protect stable
facts hard, let volatile facts move. We test that against strong baselines and,
crucially, against negative controls (flat + swapped volatility) so a win can be
attributed to the volatility signal rather than to the machinery.

The competitors
---------------
  * voltmem_real   — VoltMem with the true domain volatility priors
  * voltmem_flat   — VoltMem, all domains equal volatility (ablation)
  * voltmem_swap   — VoltMem, volatility inverted (NEGATIVE CONTROL)
  * always_write   — adopt every contradicting observation (trust everything)
  * never_write    — write once, never update (trust nothing)
  * reliability    — adopt iff source reliability >= 0.7 (strong non-volatility
                     heuristic: the honest competitor — uses trust but not volatility)

The generative model is deliberately built so that neither reliability nor
recency alone can win: it includes confident-but-false blips on stable facts
(which a reliability rule wrongly adopts) and weak-but-true updates on highly
volatile facts (which a reliability rule wrongly ignores). Volatility is the
signal that separates the two.

Scoring: for each attribute, after each session, is the system's current belief
equal to ground truth? We report accuracy overall and split by stable/volatile,
averaged over many runs. A causal result requires real > flat > swap.

Run:
    .venv/bin/python experiments/llm_memory_bench.py
    .venv/bin/python experiments/llm_memory_bench.py --sessions 30 --runs 25
"""

import argparse
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import voltmem.domains as vdomains          # noqa: E402
from voltmem import MemoryLayer             # noqa: E402
from voltmem.domains import SOURCE_RELIABILITY  # noqa: E402

# ── the simulated user's attributes ─────────────────────────────────────────
# Each attribute has a canonical volatility (its TRUE rate of change). Ground
# truth drifts at a rate derived from this; it is fixed regardless of what
# volatility profile a given system believes.

@dataclass
class Attr:
    name:   str
    domain: str      # VoltMem domain (drives the volatility prior)
    vol:    float    # canonical (true) volatility
    n_vals: int = 6  # size of the value pool


ATTRS = [
    # stable — should resist confident-but-false blips
    Attr("home_city",            "biographical",         0.10),
    Attr("communication_style",  "personality_trait",    0.05),
    Attr("dietary_preference",   "core_preference",      0.08),
    # medium
    Attr("job_title",            "professional_context", 0.30),
    Attr("close_collaborator",   "relationship",         0.35),
    # volatile — should track fast, even from weak signals
    Attr("current_project",      "current_project",      0.55),
    Attr("current_mood",         "emotional_context",    0.80),
    Attr("current_task",         "current_task",         0.90),
]

STABLE = {a.name for a in ATTRS if a.vol <= 0.20}
VOLATILE = {a.name for a in ATTRS if a.vol >= 0.50}

CONTRADICT = 0.85   # mismatch magnitude for a contradicting observation
CONFIRM = 0.05      # mismatch magnitude for a confirming observation


# ── volatility profiles (real / flat / swap) ────────────────────────────────
def profile_volatility(profile: str) -> dict[str, float]:
    real = {a.domain: a.vol for a in ATTRS}
    if profile == "real":
        return dict(real)
    if profile == "flat":
        return {d: 0.40 for d in real}
    if profile == "swap":
        # reflect within the used range so stable<->volatile swap
        return {d: round(1.0 - v, 3) for d, v in real.items()}
    raise ValueError(profile)


@contextmanager
def volatility_profile(profile: str):
    """Temporarily patch DOMAIN_VOLATILITY for the attributes we use."""
    original = dict(vdomains.DOMAIN_VOLATILITY)
    try:
        vdomains.DOMAIN_VOLATILITY.update(profile_volatility(profile))
        yield
    finally:
        vdomains.DOMAIN_VOLATILITY.clear()
        vdomains.DOMAIN_VOLATILITY.update(original)


# ── observation model ───────────────────────────────────────────────────────
@dataclass
class Obs:
    attr:     str
    value:    int
    source:   str
    mismatch: float
    is_true:  bool          # does this observation match the (new) ground truth?


@dataclass
class World:
    """Ground truth + the noisy observation stream a system sees each session."""
    rng: random.Random
    truth: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for a in ATTRS:
            self.truth[a.name] = self.rng.randrange(a.n_vals)

    def _new_value(self, a: Attr) -> int:
        cur = self.truth[a.name]
        choices = [v for v in range(a.n_vals) if v != cur]
        return self.rng.choice(choices)

    def _update_source(self, a: Attr) -> str:
        # very volatile facts are often only weakly inferred from behaviour;
        # a reliability rule will WRONGLY ignore these true updates.
        if a.vol >= 0.80:
            return self.rng.choices(
                ["weak_inference", "strong_inference", "explicit_statement"],
                weights=[0.55, 0.30, 0.15])[0]
        if a.vol >= 0.50:
            return self.rng.choices(
                ["strong_inference", "explicit_statement", "weak_inference"],
                weights=[0.45, 0.35, 0.20])[0]
        # stable / medium true changes are usually stated explicitly
        return self.rng.choices(
            ["explicit_statement", "strong_inference"], weights=[0.7, 0.3])[0]

    def _blip_source(self, a: Attr) -> str:
        # blips on STABLE facts are often confident-sounding (out-of-character
        # one-offs, reliable-looking hearsay); a reliability rule WRONGLY adopts.
        if a.name in STABLE:
            return self.rng.choices(
                ["explicit_statement", "strong_inference", "weak_inference"],
                weights=[0.5, 0.2, 0.3])[0]
        return self.rng.choices(
            ["weak_inference", "strong_inference"], weights=[0.7, 0.3])[0]

    def step(self) -> list[Obs]:
        """Advance ground truth one session and emit observations."""
        obs: list[Obs] = []
        for a in ATTRS:
            change_p = min(max(a.vol * 0.6, 0.02), 0.6)
            blip_p = 0.18

            if self.rng.random() < change_p:
                # genuine change → truthful (possibly weak) update observation
                self.truth[a.name] = self._new_value(a)
                obs.append(Obs(a.name, self.truth[a.name],
                               self._update_source(a), CONTRADICT, True))
            elif self.rng.random() < blip_p:
                # no real change, but a contradicting FALSE observation arrives
                false_val = self._new_value(a)
                obs.append(Obs(a.name, false_val, self._blip_source(a),
                               CONTRADICT, False))
            else:
                # a confirming observation of the current truth
                obs.append(Obs(a.name, self.truth[a.name],
                               "explicit_statement", CONFIRM, True))
        self.rng.shuffle(obs)
        return obs


# ── memory systems (each maps observations -> a current belief per attribute) ─
class BaseSystem:
    name = "base"

    def __init__(self, init_truth: dict[str, int]):
        self.belief = dict(init_truth)

    def ingest(self, o: Obs):
        raise NotImplementedError

    def answer(self, attr: str) -> int:
        return self.belief[attr]

    def close(self):
        pass


class AlwaysWrite(BaseSystem):
    name = "always_write"

    def ingest(self, o: Obs):
        if o.mismatch >= 0.15:
            self.belief[o.attr] = o.value


class NeverWrite(BaseSystem):
    name = "never_write"

    def ingest(self, o: Obs):
        pass  # write-once: keep the initial value forever


class ReliabilityThreshold(BaseSystem):
    name = "reliability"

    def ingest(self, o: Obs):
        if o.mismatch >= 0.15 and SOURCE_RELIABILITY.get(o.source, 0.5) >= 0.7:
            self.belief[o.attr] = o.value


class VoltMemSystem(BaseSystem):
    """VoltMem under a given volatility profile. Belief = the currently active
    memory item per attribute. Volatility is frozen at the profile prior (EMA
    reset each ingest) so real/flat/swap is a clean, controlled treatment."""

    def __init__(self, init_truth: dict[str, int], profile: str):
        self.name = f"voltmem_{profile}"
        self.profile = profile
        self.mem = MemoryLayer(":memory:")
        self._attr_domain = {a.name: a.domain for a in ATTRS}
        with volatility_profile(profile):
            for a in ATTRS:
                self.mem.write(self._fmt(a.name, init_truth[a.name]),
                               domain=a.domain, source="explicit_statement",
                               tags=[a.name])

    @staticmethod
    def _fmt(attr: str, value: int) -> str:
        return f"{attr}={value}"

    def _freeze_volatility(self, domain: str):
        for it in self.mem._store.all_active(domain=domain):
            if it.volatility_ema >= 0:
                it.volatility_ema = -1.0
                self.mem._store.update(it)

    def ingest(self, o: Obs):
        domain = self._attr_domain[o.attr]
        with volatility_profile(self.profile):
            self.mem.observe(self._fmt(o.attr, o.value), domain=domain,
                             mismatch_magnitude=o.mismatch, source=o.source,
                             tags=[o.attr])
            self._freeze_volatility(domain)

    def answer(self, attr: str) -> int:
        domain = self._attr_domain[attr]
        active = [it for it in self.mem._store.all_active(domain=domain)
                  if attr in it.tags]
        if not active:
            return -1
        cur = max(active, key=lambda x: x.last_confirmed_at)
        try:
            return int(cur.content.split("=")[1])
        except (IndexError, ValueError):
            return -1

    def close(self):
        self.mem.close()


SYSTEMS = ["voltmem_real", "voltmem_flat", "voltmem_swap",
           "always_write", "never_write", "reliability"]


def make_system(key: str, init_truth: dict[str, int]) -> BaseSystem:
    if key == "voltmem_real":
        return VoltMemSystem(init_truth, "real")
    if key == "voltmem_flat":
        return VoltMemSystem(init_truth, "flat")
    if key == "voltmem_swap":
        return VoltMemSystem(init_truth, "swap")
    if key == "always_write":
        return AlwaysWrite(init_truth)
    if key == "never_write":
        return NeverWrite(init_truth)
    if key == "reliability":
        return ReliabilityThreshold(init_truth)
    raise ValueError(key)


# ── one run ─────────────────────────────────────────────────────────────────
def run_once(seed: int, sessions: int):
    rng = random.Random(seed)
    world = World(rng)
    systems = {k: make_system(k, world.truth) for k in SYSTEMS}

    # correct counts: [system][class] -> (correct, total)
    stats = {k: {"stable": [0, 0], "volatile": [0, 0], "all": [0, 0]}
             for k in SYSTEMS}

    for _ in range(sessions):
        for o in world.step():
            for s in systems.values():
                s.ingest(o)
        # score every attribute against current ground truth this session
        for a in ATTRS:
            cls = "stable" if a.name in STABLE else (
                "volatile" if a.name in VOLATILE else "medium")
            for k, s in systems.items():
                correct = int(s.answer(a.name) == world.truth[a.name])
                stats[k]["all"][0] += correct
                stats[k]["all"][1] += 1
                if cls in ("stable", "volatile"):
                    stats[k][cls][0] += correct
                    stats[k][cls][1] += 1

    for s in systems.values():
        s.close()
    return stats


def run(sessions: int, runs: int, seed0: int = 0):
    agg = {k: {"stable": [0, 0], "volatile": [0, 0], "all": [0, 0]}
           for k in SYSTEMS}
    for r in range(runs):
        st = run_once(seed0 + r, sessions)
        for k in SYSTEMS:
            for cls in ("stable", "volatile", "all"):
                agg[k][cls][0] += st[k][cls][0]
                agg[k][cls][1] += st[k][cls][1]

    def acc(k, cls):
        c, t = agg[k][cls]
        return c / t if t else 0.0

    print("=" * 74)
    print("LLM-AGENT MEMORY BENCHMARK — current-truth accuracy")
    print("=" * 74)
    print(f"  {sessions} sessions x {runs} runs; {len(ATTRS)} attributes "
          f"({len(STABLE)} stable, {len(VOLATILE)} volatile, "
          f"{len(ATTRS) - len(STABLE) - len(VOLATILE)} medium)\n")
    print(f"  {'system':<16}{'overall':>10}{'stable':>10}{'volatile':>10}"
          f"{'balanced':>10}")
    print("  " + "-" * 56)
    rows = {}
    for k in SYSTEMS:
        o, s, v = acc(k, "all"), acc(k, "stable"), acc(k, "volatile")
        bal = 2 * s * v / (s + v) if (s + v) else 0.0   # harmonic mean
        rows[k] = (o, s, v, bal)
        star = "  <-- real priors" if k == "voltmem_real" else ""
        print(f"  {k:<16}{o:>10.3f}{s:>10.3f}{v:>10.3f}{bal:>10.3f}{star}")

    # ── verdict ──────────────────────────────────────────────────────────────
    print("\n" + "-" * 74)
    print("VERDICT:")
    real = rows["voltmem_real"]
    flat = rows["voltmem_flat"]
    swap = rows["voltmem_swap"]
    best_baseline = max(("always_write", "never_write", "reliability"),
                        key=lambda k: rows[k][3])   # by balanced
    bb = rows[best_baseline]

    causal = real[3] > flat[3] > swap[3]
    beats_baselines = real[3] >= bb[3]

    print(f"  balanced (harmonic mean of stable & volatile):")
    print(f"    voltmem_real={real[3]:.3f}  flat={flat[3]:.3f}  "
          f"swap={swap[3]:.3f}")
    print(f"    best naive baseline = {best_baseline} ({bb[3]:.3f})")
    print(f"    always_write bal={rows['always_write'][3]:.3f}, "
          f"never_write bal={rows['never_write'][3]:.3f}, "
          f"reliability bal={rows['reliability'][3]:.3f}")

    if causal and beats_baselines:
        print(f"\n  PASS. VoltMem with real priors gives the best BALANCED memory "
              f"correctness ({real[3]:.3f}) — it neither goes stale on volatile "
              "facts (like never_write) nor gets corrupted on stable facts (like "
              "always_write), and it beats a pure reliability heuristic. The "
              "monotonic real > flat > swap ordering is the causal evidence: the "
              "volatility signal is doing the work, not the machinery.")
    elif causal:
        print(f"\n  PARTIAL. Causal ordering holds (real>flat>swap) but a naive "
              f"baseline ({best_baseline}) matches/*beats* the balanced score. "
              "Volatility steers behaviour but the win over heuristics is not "
              "clean in this regime — report honestly and inspect per-class.")
    else:
        print("\n  NEEDS REVIEW. real>flat>swap did not hold; effect not cleanly "
              "attributable to volatility here. Inspect the observation model.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=24)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.sessions, args.runs, args.seed)


if __name__ == "__main__":
    main()
