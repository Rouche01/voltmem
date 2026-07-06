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


def should_escalate(
    item: MemoryItem,
    mismatch_magnitude: float,
    source: str = "explicit_statement",
    goal_delta: float = 0.0,
    load: float = 1.0,
) -> bool:
    E_t, theta_t = escalation_score(
        item, mismatch_magnitude, source, goal_delta, load)
    return E_t > theta_t


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


def update_volatility_ema(item: MemoryItem, observed_mismatch: float) -> float:
    """
    Update the per-item empirical volatility estimate via EMA.
    Called *before* any write/audit (pre-update, to avoid circularity).

    Returns the updated EMA value.
    """
    current = item.effective_volatility
    updated = BETA * current + (1 - BETA) * observed_mismatch
    return float(updated)
