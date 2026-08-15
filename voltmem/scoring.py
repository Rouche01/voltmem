"""
Scoring — the mathematical core derived from our conversation.

Escalation score (homeostatic):
    E_t = [M_t * R_t / C^alpha] * V_d * G_t_factor
    theta_t = theta_0 * (1 / V_d) * L_t

Escalation score (allostatic):
    E_t = [M_t * R_t / C^alpha] * G_t_factor
    theta_t = theta_0 * (1 / V_d) * L_t * s(m)

Default is ``composite``: allostatic only for an explicit high-M correction
or an unexpected residual; otherwise homeostatic.

Decision:
    E_t > theta_t  →  Audit  →  then retrieve from updated calibration
    E_t <= theta_t →  Retrieve directly from existing calibration

Staleness score (for retrieval ranking):
    staleness = 1 - exp(-V_d * age_days)
    Ranges 0 (fresh) → 1 (fully stale)
    Used to down-rank old volatile memories in retrieval.
"""

import math
import time
from .domains import MemoryItem, DOMAIN_VOLATILITY, SOURCE_RELIABILITY


# ── tunable global parameters ─────────────────────────────────────────────────

ALPHA = 0.6          # entrenchment exponent — controls how strongly
                     # repetition suppresses escalation (must stay < 1
                     # to avoid making old items unauditable)

THETA_0 = 0.15       # base escalation threshold

GAMMA = 2.0          # volatility exponent in protection weight:
                     # w_d = 1 / V_d^gamma

BETA = 0.5           # EMA decay for per-item volatility update

VOL_DRIFT_MAX = 0.2  # per-item EMA may drift at most this far from domain prior

# Strong explicit corrections must be able to update medium-stable domains.
# Low V_d penalises E_t and raises theta_t, so M_t=1.0 alone can still fail
# (e.g. professional_context V_d=0.3 → E≈0.28 < θ=0.5). For the medium-stable
# band only, cap θ_t relative to V_d so drift via auto_discover stays safe.
# Very-stable domains (V_d < EXPLICIT_MIN_VD) never get the cap — cumulative
# mismatch is the fallback. Volatile domains (V_d > EXPLICIT_MAX_VD) rely on
# raw E_t > theta_t.
EXPLICIT_OVERRIDE_M = 0.85
EXPLICIT_MIN_VD = 0.15       # below: very stable — no θ-cap
EXPLICIT_MAX_VD = 0.55       # above: volatile enough — raw math suffices
EXPLICIT_E_RATIO = 0.85      # cap θ at V_d * ratio; need E_t > cap (E ≈ M·R·V·G)

# After this many logged (below-threshold) mismatches, the next conflicting
# observation escalates — cumulative evidence of a real change.
CUMULATIVE_MISMATCH_ESCALATE = 3

# Allostatic mode (opt-in): V_d sets the bar, residual drives the score,
# recent surprise s(m) reopens or settles the threshold.
ESCALATION_MODES = ("homeostatic", "allostatic", "composite")
ESCALATION_MODE_ALIASES = {"current": "homeostatic"}
S_MIN = 0.25         # floor on s(m): high sustained surprise → 25% of prior θ
SURPRISE_BETA = 0.5  # EMA decay for per-item surprise_ema (same family as BETA)
# Quiet time halves recent surprise → the bar re-settles. Calibrated on Battery D
# (experiments/allostatic_ablation.py): at 14d the surprise EMA's fixed point under
# a stream of weak evidence converges to within 2e-4 ABOVE the trigger and never
# crosses, so accumulated-weak-evidence changes were missed. 30d clears it at turn
# 10 with every negative control still holding, and Battery A/B/C unchanged.
SURPRISE_HALFLIFE_DAYS = 30.0
# Anticipation for residual surprise r_t = |M_t − Ê_t| / σ_t.
# Confirms sit near 0.05; V_d widens σ so a volatile channel's ordinary noise
# is expected rather than surprising. Z_SCALE maps 3σ onto r_t = 1.
MISMATCH_PRIOR = 0.05
SIGMA_FLOOR = 0.08
SIGMA_VD_KAPPA = 0.35
RESIDUAL_Z_SCALE = 3.0
# Sleeptime / consolidate: time-decayed evidence mass vs a V_d-scaled bar.
# Online r_t asks "was this step unexpected?"; this asks "did belief that the
# stored fact still holds actually move?" Same half-life as surprise decay so
# daily piles accumulate and monthly ones do not. V_d is only in the bar —
# stable facts need a heavier pile, matching Battery D's preference control.
BELIEF_SHIFT_K = 0.35
BELIEF_SHIFT_MIN_VD = 0.05
# Composite gate: r_t at/above this (≈ 1.5σ on the 3σ scale) is unexpected,
# but only after the item has a learned Ê. A fresh item has no anticipation,
# so the first weak blip must not open allostatic — that is Battery E.
RESIDUAL_GATE = 0.50

# V_d exponent in E_t. The current law charges V_d twice (numerator of E_t and
# denominator of theta_t); allostatic charges it once, on the threshold only.
# Exposed as a knob so the two can be compared without a second control law.
V_EXP_HOMEOSTATIC = 1.0
V_EXP_CURRENT = V_EXP_HOMEOSTATIC  # alias; prefer V_EXP_HOMEOSTATIC
V_EXP_ALLOSTATIC = 0.0

STALENESS_HALFLIFE = {   # days at which an item reaches 50% staleness
    # derived from V_d: halflife ≈ ln(2) / V_d (in days)
    # pre-computed for reference; staleness() uses V_d directly
}

# Problem 3 — dampen freshness when top-candidate similarity is flat
# (under-specified queries). mix=1 → today's full V_d penalty; mix→MIX_MIN
# on a plateau so volatility cannot invert near-ties.
SIM_SPREAD_FLAT = 0.05   # max−min sim at/below this → mix = MIX_MIN
SIM_SPREAD_FULL = 0.15   # at/above this → mix = 1.0 (full freshness)
MIX_MIN = 0.25           # floor freshness weight on a plateau


# ── core equations ────────────────────────────────────────────────────────────

def protection_weight(item: MemoryItem) -> float:
    """
    w_d = 1 / V_d^gamma

    High volatility → small weight → memory gets weak EWC-style protection
    (easy to overwrite or let decay). Low volatility → high weight → protected.
    Clamped to [0.05, 20.0] for numerical stability.
    """
    v = max(item.effective_volatility, 1e-6)
    return float(min(max(1.0 / (v ** GAMMA), 0.05), 20.0))


def _g_factor(goal_delta: float) -> float:
    # sigmoid-ish, centred at 0, range [0.1, 2.0]
    return 0.1 + 1.9 / (1.0 + math.exp(-3.0 * goal_delta))


def normalize_escalation_mode(mode: str) -> str:
    mode = ESCALATION_MODE_ALIASES.get(mode, mode)
    if mode not in ESCALATION_MODES:
        raise ValueError(
            f"escalation_mode must be one of {ESCALATION_MODES}, got {mode!r}")
    return mode


def expected_mismatch(item: MemoryItem) -> float:
    """Ê_t: mismatch size this item currently treats as normal."""
    ema = float(getattr(item, "mismatch_ema", -1.0) or -1.0)
    if ema < 0.0:
        return MISMATCH_PRIOR
    return float(max(0.0, min(1.0, ema)))


def mismatch_sigma(item: MemoryItem) -> float:
    """σ_t: expected variability of mismatch, widened by the domain prior V_d.

    Empirical scatter is the item's learned noise. V_d is anticipated
    uncertainty for this *kind* of fact — Sterling's moved setpoint — so the
    same raw M is less surprising on a volatile channel.
    """
    V_d = DOMAIN_VOLATILITY.get(
        item.domain, float(getattr(item, "effective_volatility", 0.5) or 0.5))
    var = float(getattr(item, "mismatch_var", -1.0) or -1.0)
    empirical = math.sqrt(max(var, 0.0)) if var >= 0.0 else 0.0
    return float(max(SIGMA_FLOOR, empirical) + SIGMA_VD_KAPPA * max(V_d, 0.0))


def residual_surprise(item: MemoryItem, observed_mismatch: float) -> float:
    """Unexpected residual r_t in [0, 1].

    Distance from the *predicted* mismatch, not from the stored sentence.
    A constant weak stream becomes predicted (r_t → 0); a quiet-then-shift
    spike is many σ out even at the same M_t.
    """
    M_t = float(max(0.0, min(1.0, observed_mismatch)))
    z = abs(M_t - expected_mismatch(item)) / mismatch_sigma(item)
    return float(min(1.0, z / RESIDUAL_Z_SCALE))


def resolve_escalation_law(
    item: MemoryItem,
    mismatch_magnitude: float,
    source: str = "explicit_statement",
    mode: str = "homeostatic",
) -> str:
    """Which control law this observation should use.

    ``homeostatic`` and ``allostatic`` pass through. ``composite`` is homeostatic
    unless the world is *telling you* it moved: an explicit high-M correction,
    or an unexpected residual against a learned mismatch size. Cumulative
    weak piles are sleeptime's job, not this gate.
    """
    mode = normalize_escalation_mode(mode)
    if mode != "composite":
        return mode
    M_t = float(max(0.0, min(1.0, mismatch_magnitude)))
    if M_t >= EXPLICIT_OVERRIDE_M and source == "explicit_statement":
        return "allostatic"
    learned = float(getattr(item, "mismatch_ema", -1.0) or -1.0) >= 0.0
    if learned and residual_surprise(item, M_t) >= RESIDUAL_GATE:
        return "allostatic"
    return "homeostatic"


def belief_shift_mass(
    evidence: list[dict],
    *,
    now: float | None = None,
    cutoff: float = 0.0,
) -> float:
    """Decayed evidence that the stored fact is no longer true.

    Each logged mismatch votes ``M_t * R_t``, halved every
    ``SURPRISE_HALFLIFE_DAYS``. Rows at or before ``cutoff`` (typically the
    last confirm) are ignored so interleaved confirmations reset the pile.
    """
    if now is None:
        now = time.time()
    total = 0.0
    for row in evidence:
        created = float(row.get("created_at") or 0.0)
        if created < cutoff - 1.0:
            continue
        M_t = float(max(0.0, min(1.0, float(row.get("mismatch_magnitude") or 0.0))))
        R_t = SOURCE_RELIABILITY.get(str(row.get("source") or ""), 0.5)
        age_days = max(0.0, (now - created) / 86400.0)
        total += M_t * R_t * (0.5 ** (age_days / SURPRISE_HALFLIFE_DAYS))
    return float(total)


def belief_shift_bar(domain: str) -> float:
    """Mass required to treat the pile as a real change. Higher for stable domains."""
    V_d = DOMAIN_VOLATILITY.get(domain, 0.5)
    return float(BELIEF_SHIFT_K / max(V_d, BELIEF_SHIFT_MIN_VD))


def belief_has_shifted(
    item: MemoryItem,
    evidence: list[dict],
    *,
    now: float | None = None,
    min_evidence: int = 3,
) -> tuple[bool, float, float]:
    """Sleeptime gate: (moved, mass, bar).

    Does not read ``mismatch_count``. A lifetime tally cannot habituation-decay.
    """
    if now is None:
        now = time.time()
    created = float(getattr(item, "created_at", 0.0) or 0.0)
    confirmed = float(getattr(item, "last_confirmed_at", 0.0) or 0.0)
    # Write() stamps last_confirmed_at = created_at. A later confirm resets the pile.
    cutoff = confirmed if confirmed > created + 0.5 else created
    recent = [
        row for row in evidence
        if float(row.get("created_at") or 0.0) >= cutoff - 1.0
    ]
    mass = belief_shift_mass(recent, now=now, cutoff=cutoff)
    bar = belief_shift_bar(item.domain)
    moved = len(recent) >= min_evidence and mass >= bar
    return moved, float(mass), float(bar)


def update_mismatch_expectation(
    item: MemoryItem,
    observed_mismatch: float,
    source: str = "explicit_statement",
) -> tuple[float, float]:
    """Fold M_t into (Ê, variance). Call AFTER residual_surprise for this step."""
    M_t = float(max(0.0, min(1.0, observed_mismatch)))
    reliability = SOURCE_RELIABILITY.get(source, 0.5)
    reliability = min(max(reliability, 0.0), 1.0)
    alpha = (1.0 - SURPRISE_BETA) * reliability
    e = expected_mismatch(item)
    e_new = (1.0 - alpha) * e + alpha * M_t
    var = float(getattr(item, "mismatch_var", -1.0) or -1.0)
    if var < 0.0:
        sig = mismatch_sigma(item)
        var = sig * sig
    delta = M_t - e
    var_new = (1.0 - alpha) * var + alpha * (delta * delta)
    return float(max(0.0, min(1.0, e_new))), float(max(var_new, 0.0))


def recent_surprise(item: MemoryItem, now: float | None = None) -> float:
    """Per-item surprise in [0, 1], decayed by time since it was last updated.

    Confirms pull the EMA down; quiet time decays whatever is left. Both
    routes must exist or the signal can only ever grow.
    """
    s = max(0.0, min(1.0, float(getattr(item, "surprise_ema", 0.0) or 0.0)))
    if s <= 0.0:
        return 0.0
    at = float(getattr(item, "surprise_at", 0.0) or 0.0)
    if at <= 0.0:
        return s
    if now is None:
        now = time.time()
    age_days = max(0.0, (now - at) / 86400.0)
    return float(s * 0.5 ** (age_days / SURPRISE_HALFLIFE_DAYS))


def surprise_mode_scale(item: MemoryItem, now: float | None = None) -> float:
    """s(m) in [S_MIN, 1]: 1 = settled / homeostatic, S_MIN = reopened.

    Deliberately ignores ``mismatch_count``: that counter only ever increases
    for a living item, so folding it in would hold the bar permanently open on
    long-lived memories — reopening without any route back to settled.
    """
    return float(1.0 - (1.0 - S_MIN) * recent_surprise(item, now))


def escalation_score(
    item: MemoryItem,
    mismatch_magnitude: float,          # M_t: [0,1] how strongly new info contradicts
    source: str = "explicit_statement", # R_t source of new signal
    goal_delta: float = 0.0,            # G_t: >0 if audit helps goal
    load: float = 1.0,                  # L_t: cognitive/compute load scalar
    mode: str = "homeostatic",
    v_exp: float | None = None,
    mode_scale: bool | None = None,
    now: float | None = None,
) -> tuple[float, float]:
    """
    Returns (E_t, theta_t).

    homeostatic:
        E_t = [M_t * R_t / C^alpha] * V_d * G_factor
        theta_t = theta_0 * (1 / V_d) * L_t
    allostatic:
        E_t = [M_t * R_t / C^alpha] * G_factor
        theta_t = theta_0 * (1 / V_prior) * L_t * s(m)
    composite:
        allostatic if explicit high-M or (learned Ê and r_t >= RESIDUAL_GATE);
        otherwise homeostatic. Weak piles stay on homeostatic (sleeptime).

    ``v_exp`` and ``mode_scale`` override the mode's defaults so the two
    ingredients (dropping V_d from E_t, and scaling theta by recent surprise)
    can be ablated independently. See experiments/allostatic_ablation.py.

    G_factor: maps goal_delta to a multiplier. Negative goal_delta
    (audit would hurt goal) suppresses escalation; positive amplifies.
    """
    mode = normalize_escalation_mode(mode)
    if v_exp is None and mode_scale is None:
        mode = resolve_escalation_law(
            item, mismatch_magnitude, source, mode)
    if v_exp is None:
        v_exp = V_EXP_HOMEOSTATIC if mode == "homeostatic" else V_EXP_ALLOSTATIC
    if mode_scale is None:
        mode_scale = (mode == "allostatic")

    R_t = SOURCE_RELIABILITY.get(source, 0.5)
    C   = max(item.repetition_count, 1)
    V_d = item.effective_volatility
    M_t = float(max(0.0, min(1.0, mismatch_magnitude)))
    G_factor = _g_factor(goal_delta)

    E_t = (M_t * R_t / (C ** ALPHA)) * G_factor * (max(V_d, 1e-6) ** v_exp)

    # Allostatic reads the bar off the domain prior, not the drifted per-item
    # EMA — otherwise recent mismatch walks a very-stable item into the θ-cap band.
    V_bar = (
        DOMAIN_VOLATILITY.get(item.domain, V_d) if mode == "allostatic" else V_d)
    theta_t = THETA_0 * (1.0 / max(V_bar, 1e-6)) * load
    if mode_scale:
        theta_t *= surprise_mode_scale(item, now)

    return float(E_t), float(theta_t)


def explicit_theta_cap(V_d: float) -> float | None:
    """
    Relative θ cap for medium-stable explicit overrides.

    Returns None when the band policy does not apply (very stable or already
    volatile). Otherwise V_d * EXPLICIT_E_RATIO — scales with auto_discover drift.
    """
    if V_d < EXPLICIT_MIN_VD or V_d > EXPLICIT_MAX_VD:
        return None
    return float(V_d * EXPLICIT_E_RATIO)


def escalation_decision(
    item: MemoryItem,
    mismatch_magnitude: float,
    source: str = "explicit_statement",
    goal_delta: float = 0.0,
    load: float = 1.0,
    mode: str = "homeostatic",
    v_exp: float | None = None,
    mode_scale: bool | None = None,
    now: float | None = None,
) -> tuple[bool, float, float]:
    """
    Returns (escalate, E_t, theta_effective).

    Uses E_t > theta_t, plus two adjustments that keep medium-stable domains
    correctable without weakening weak-evidence / very-stable retention:

    1. High-M_t explicit statement in the medium-stable V_d band — cap θ_t at
       V_d * EXPLICIT_E_RATIO (scales with drift; very-stable domains excluded).
    2. Cumulative mismatches — after CUMULATIVE_MISMATCH_ESCALATE logged
       conflicts, further mismatch evidence forces escalation.
    """
    M_t = float(max(0.0, min(1.0, mismatch_magnitude)))
    if v_exp is None and mode_scale is None:
        mode = resolve_escalation_law(item, M_t, source, mode)
    E_t, theta_t = escalation_score(
        item, mismatch_magnitude, source, goal_delta, load,
        mode=mode, v_exp=v_exp, mode_scale=mode_scale, now=now)

    V_d = item.effective_volatility
    if mode == "allostatic":
        V_d = DOMAIN_VOLATILITY.get(item.domain, V_d)
    if (
        M_t >= EXPLICIT_OVERRIDE_M
        and source == "explicit_statement"
        and goal_delta >= 0.0
    ):
        cap = explicit_theta_cap(V_d)
        if cap is not None:
            theta_t = min(theta_t, cap)

    escalate = E_t > theta_t
    # Weak inferences must not grind down a stable fact via repetition alone.
    R_t = SOURCE_RELIABILITY.get(source, 0.5)
    if (
        item.mismatch_count >= CUMULATIVE_MISMATCH_ESCALATE
        and M_t >= 0.5
        and R_t >= SOURCE_RELIABILITY["strong_inference"]
    ):
        escalate = True

    return escalate, float(E_t), float(theta_t)


def should_escalate(
    item: MemoryItem,
    mismatch_magnitude: float,
    source: str = "explicit_statement",
    goal_delta: float = 0.0,
    load: float = 1.0,
    mode: str = "homeostatic",
    v_exp: float | None = None,
    mode_scale: bool | None = None,
    now: float | None = None,
) -> bool:
    """True when the observation should audit/update the stored item."""
    return escalation_decision(
        item, mismatch_magnitude, source, goal_delta, load,
        mode=mode, v_exp=v_exp, mode_scale=mode_scale, now=now)[0]


def staleness(item: MemoryItem, now: float | None = None) -> float:
    """
    staleness = 1 - exp(-V_d * age_in_days)

    0 = perfectly fresh, 1 = fully stale.
    Used to penalise volatile memories that haven't been confirmed recently.
    """
    if now is None:
        now = time.time()
    age_secs = max(0.0, now - item.last_confirmed_at)
    age_days = age_secs / 86400.0
    V_d = item.effective_volatility
    return float(1.0 - math.exp(-V_d * age_days))


def similarity_spread(sims: list[float] | tuple[float, ...]) -> float:
    """max(sim) − min(sim) over candidates; 0.0 if fewer than two scores."""
    if len(sims) < 2:
        return 0.0
    return float(max(sims) - min(sims))


def freshness_mix(
    spread: float,
    *,
    flat: float = SIM_SPREAD_FLAT,
    full: float = SIM_SPREAD_FULL,
    mix_min: float = MIX_MIN,
) -> float:
    """
    How strongly to apply V_d·staleness given top-candidate similarity spread.

    Returns 1.0 when spread is informative (specific query), mix_min when
    similarity is flat (under-specified / plateau), and linearly interpolates
    between. Fewer than two candidates should pass spread=0 but callers ought
    to treat that as mix=1.0 (no plateau signal) — see MemoryLayer.retrieve.
    """
    if full <= flat:
        return 1.0
    if spread >= full:
        return 1.0
    if spread <= flat:
        return float(mix_min)
    t = (spread - flat) / (full - flat)
    return float(mix_min + (1.0 - mix_min) * t)


def retrieval_score(
    item: MemoryItem,
    semantic_similarity: float,         # [0,1] from embedding or keyword match
    now: float | None = None,
    mix: float = 1.0,
) -> float:
    """
    Combined retrieval score balancing semantic relevance and freshness.

    score = semantic_similarity * (1 - mix * V_d * staleness)

    V_d scales how much age hurts (volatile → more). ``mix`` (default 1)
    further scales that penalty; values < 1 dampen freshness when candidate
    similarities are flat (Problem 3 — under-specified queries).
    """
    stale = staleness(item, now)
    weight = item.effective_volatility  # volatile → staleness matters more
    m = float(max(0.0, min(1.0, mix)))
    return float(semantic_similarity * (1.0 - m * weight * stale))


def update_volatility_ema(
    item: MemoryItem,
    observed_mismatch: float,
    source: str = "explicit_statement",
) -> float:
    """
    Update the per-item empirical volatility estimate via EMA.
    Called *before* any write/audit (pre-update, to avoid circularity).

    The learning rate toward the new observation is scaled by the source's
    reliability. A base step of (1 - BETA) is taken for a fully reliable source
    (reliability >= 1.0); a low-trust source (e.g. weak_inference, R=0.4) takes a
    proportionally smaller step, so the volatility estimate is not yanked around
    by noisy signals. This is what stops a genuinely stable domain from drifting
    "volatile" after a handful of weak, contradictory observations.

        alpha   = (1 - BETA) * clamp(reliability, 0, 1)
        updated = (1 - alpha) * current + alpha * observed_mismatch

    For a fully reliable source this reduces exactly to the original
    EMA (alpha = 1 - BETA), so reliable updates behave as before.

    Returns the updated EMA value.
    """
    prior = DOMAIN_VOLATILITY.get(item.domain, 0.5)
    current = item.volatility_ema if item.volatility_ema >= 0 else prior
    reliability = SOURCE_RELIABILITY.get(source, 0.5)
    reliability = min(max(reliability, 0.0), 1.0)
    alpha = (1.0 - BETA) * reliability
    updated = (1.0 - alpha) * current + alpha * observed_mismatch
    lo = max(0.05, prior - VOL_DRIFT_MAX)
    hi = min(0.95, prior + VOL_DRIFT_MAX)
    return float(min(max(updated, lo), hi))


def update_surprise_ema(
    item: MemoryItem,
    observed_mismatch: float,
    source: str = "explicit_statement",
    now: float | None = None,
) -> float:
    """
    Medium-timescale tracker of *unexpected* residual for allostatic s(m).

    Folds residual_surprise(item, M_t) — distance from predicted mismatch —
    not raw M_t. A predicted weak stream therefore does not keep the bar
    open; an out-of-regime spike does. Confirms (M near Ê) pull toward 0.
    Time decay still applies via recent_surprise before the update.
    """
    current = recent_surprise(item, now)
    reliability = SOURCE_RELIABILITY.get(source, 0.5)
    reliability = min(max(reliability, 0.0), 1.0)
    alpha = (1.0 - SURPRISE_BETA) * reliability
    observed = residual_surprise(item, observed_mismatch)
    updated = (1.0 - alpha) * current + alpha * observed
    return float(min(max(updated, 0.0), 1.0))
