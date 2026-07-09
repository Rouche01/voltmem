"""
Domain volatility priors.

V_d is the expected rate of change for a given memory domain.
Higher = more volatile = weaker protection = lower mismatch threshold
to trigger an audit/update.

Scale: 0.0 (never changes) → 1.0 (changes very fast).
These are defaults; callers can override per-item or register custom domains.
"""

from dataclasses import dataclass, field
from typing import Optional

# ── built-in domain priors ────────────────────────────────────────────────────

DOMAIN_VOLATILITY: dict[str, float] = {
    # Slow-changing — protect hard
    "personality_trait":    0.05,   # how the user fundamentally operates
    "core_preference":      0.08,   # deep aesthetic / communication preferences
    "biographical":         0.10,   # birthplace, native language, background
    "long_term_goal":       0.15,   # career direction, life goals

    # Medium — moderate protection
    "professional_context": 0.30,   # job, company, role (changes every few years)
    "skill":                0.25,   # competencies the user has
    "relationship":         0.35,   # people they work with / are close to

    # Fast-changing — hold loosely
    "current_project":      0.55,   # what they're working on right now
    "stated_preference":    0.45,   # preferences stated in this period
    "opinion":              0.50,   # views that may shift
    "location":             0.60,   # where they are / living situation
    "emotional_context":    0.80,   # current mood, stress level
    "current_task":         0.90,   # immediate to-do
    "transient_fact":       0.95,   # anything clearly moment-specific
}

# Source reliability weights (R_t in the escalation equation)
# Higher = more trustworthy signal for both write and mismatch detection
SOURCE_RELIABILITY: dict[str, float] = {
    "explicit_statement":   1.0,    # user directly stated it
    "repeated_confirmation":1.2,    # confirmed across multiple turns
    "strong_inference":     0.7,    # clearly implied
    "weak_inference":       0.4,    # loosely inferred from behaviour
    "system_generated":     0.3,    # injected by the wrapping system
}


@dataclass
class MemoryItem:
    """A single unit of persistent memory."""
    id:               str
    content:          str                    # the actual stored fact/preference
    domain:           str                    # one of DOMAIN_VOLATILITY keys
    source:           str                    # one of SOURCE_RELIABILITY keys
    namespace:        str   = "default"      # tenant/user isolation key

    # Equation terms — updated over time
    repetition_count: int   = 1             # C: how many times confirmed
    volatility_ema:   float = -1.0          # V_d EMA; -1 means use domain prior
    mismatch_count:   int   = 0             # cumulative mismatch events
    goal_delta:       float = 0.0           # G_t: positive = helps goal, neg = hurts

    # Timestamps (unix seconds)
    created_at:       float = 0.0
    last_confirmed_at:float = 0.0
    last_audited_at:  float = 0.0

    # Metadata
    tags:             list[str] = field(default_factory=list)
    superseded_by:    Optional[str] = None  # id of item that replaced this one

    @property
    def is_active(self) -> bool:
        return self.superseded_by is None

    @property
    def effective_volatility(self) -> float:
        if self.volatility_ema >= 0:
            return self.volatility_ema
        return DOMAIN_VOLATILITY.get(self.domain, 0.5)
