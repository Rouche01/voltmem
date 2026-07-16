"""
Scoring — the mathematical core derived from our conversation.

Escalation score:
    E_t = [M_t * R_t / C^alpha] * V_d * G_t_factor

Dynamic threshold:
    theta_t = theta_0 * (1 / V_d) * L_t

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

STALENESS_HALFLIFE = {   # days at which an item reaches 50% staleness
    # derived from V_d: halflife ≈ ln(2) / V_d (in days)
    # pre-computed for reference; staleness() uses V_d directly
}


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


def escalation_score(
    item: MemoryItem,
    mismatch_magnitude: float,          # M_t: [0,1] how strongly new info contradicts
    source: str = "explicit_statement", # R_t source of new signal
    goal_delta: float = 0.0,            # G_t: >0 if audit helps goal
    load: float = 1.0,                  # L_t: cognitive/compute load scalar
) -> tuple[float, float]:
    """
    Returns (E_t, theta_t).

    E_t = [M_t * R_t / C^alpha] * V_d * G_factor
    theta_t = theta_0 * (1 / V_d) * L_t

    G_factor: maps goal_delta to a multiplier. Negative goal_delta
    (audit would hurt goal) suppresses escalation; positive amplifies.
    """
    R_t = SOURCE_RELIABILITY.get(source, 0.5)
    C   = max(item.repetition_count, 1)
    V_d = item.effective_volatility
    M_t = float(max(0.0, min(1.0, mismatch_magnitude)))

    # G_factor: sigmoid-ish, centred at 0, range [0.1, 2.0]
    # goal_delta in [-1, 1]; positive = escalation amplified
    G_factor = 0.1 + 1.9 / (1 + math.exp(-3.0 * goal_delta))

    E_t = (M_t * R_t / (C ** ALPHA)) * V_d * G_factor
    theta_t = THETA_0 * (1.0 / max(V_d, 1e-6)) * load

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
    E_t, theta_t = escalation_score(
        item, mismatch_magnitude, source, goal_delta, load)

    V_d = item.effective_volatility
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
) -> bool:
    """True when the observation should audit/update the stored item."""
    return escalation_decision(
        item, mismatch_magnitude, source, goal_delta, load)[0]


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


def retrieval_score(
    item: MemoryItem,
    semantic_similarity: float,         # [0,1] from embedding or keyword match
    now: float | None = None,
) -> float:
    """
    Combined retrieval score balancing semantic relevance and freshness.

    score = semantic_similarity * (1 - staleness_weight * staleness)

    staleness_weight scales with volatility so that stable memories are
    barely penalised for age, while volatile memories decay fast.
    """
    stale  = staleness(item, now)
    weight = item.effective_volatility  # volatile → staleness matters more
    return float(semantic_similarity * (1.0 - weight * stale))


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
