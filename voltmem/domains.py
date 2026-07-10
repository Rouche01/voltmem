"""
Domain volatility priors.

V_d is the expected rate of change for a given memory domain.
Higher = more volatile = weaker protection = lower mismatch threshold
to trigger an audit/update.

Scale: 0.0 (never changes) → 1.0 (changes very fast).
These are defaults; callers can override per-item or register custom domains.
"""

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

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

# Domains that usually hold a single "current truth" slot (mood, city, task).
# remember() uses a lower in-slot linking threshold for these.
SLOT_DOMAINS: frozenset[str] = frozenset({
    "emotional_context",
    "location",
    "current_task",
    "transient_fact",
})

# Related domains searched together when linking a new statement to existing
# memories (e.g. paraphrased prefs split across classifiers).
DOMAIN_SIBLINGS: dict[str, frozenset[str]] = {
    "core_preference": frozenset({"core_preference", "stated_preference"}),
    "stated_preference": frozenset({"core_preference", "stated_preference"}),
}

# Minimum semantic overlap to link a lone item in a volatile slot when the
# volatility-scaled threshold is not quite met.
SLOT_LINK_FLOOR: float = 0.30

# Source reliability weights (R_t in the escalation equation)
# Higher = more trustworthy signal for both write and mismatch detection
SOURCE_RELIABILITY: dict[str, float] = {
    "explicit_statement":   1.0,    # user directly stated it
    "repeated_confirmation":1.2,    # confirmed across multiple turns
    "strong_inference":     0.7,    # clearly implied
    "weak_inference":       0.4,    # loosely inferred from behaviour
    "system_generated":     0.3,    # injected by the wrapping system
}


class DomainRegistry:
    """Register custom domains with volatility priors and slot/linking rules.

    Pass to ``create_memory(domains=...)``. Patches module-level domain maps for
    the lifetime of the returned ``Memory`` (restored on ``close()``).
    """

    def __init__(self) -> None:
        self._volatility: dict[str, float] = {}
        self._slot_domains: set[str] = set()
        self._siblings: dict[str, frozenset[str]] = {}

    def register(
        self,
        name: str,
        volatility: float,
        *,
        slot: bool = False,
        siblings: Iterable[str] | None = None,
    ) -> "DomainRegistry":
        """Add or override a domain. Volatility in [0, 1] — higher = faster-changing."""
        if not 0.0 <= volatility <= 1.0:
            raise ValueError(f"volatility must be in [0, 1], got {volatility}")
        self._volatility[name] = volatility
        if slot:
            self._slot_domains.add(name)
        if siblings is not None:
            self._siblings[name] = frozenset({name, *siblings})
        return self

    def known_domains(self) -> frozenset[str]:
        return frozenset({*DOMAIN_VOLATILITY.keys(), *self._volatility.keys()})

    def volatility(self, domain: str) -> float:
        if domain in self._volatility:
            return self._volatility[domain]
        return DOMAIN_VOLATILITY.get(domain, 0.5)

    def install(self) -> Callable[[], None]:
        """Merge registered domains into module-level maps. Returns a restore fn."""
        import voltmem.domains as dom

        saved_volatility = dict(dom.DOMAIN_VOLATILITY)
        saved_slots = dom.SLOT_DOMAINS
        saved_siblings = dict(dom.DOMAIN_SIBLINGS)

        dom.DOMAIN_VOLATILITY.update(self._volatility)
        dom.SLOT_DOMAINS = frozenset(set(dom.SLOT_DOMAINS) | self._slot_domains)
        for name, group in self._siblings.items():
            dom.DOMAIN_SIBLINGS[name] = group

        def restore() -> None:
            dom.DOMAIN_VOLATILITY.clear()
            dom.DOMAIN_VOLATILITY.update(saved_volatility)
            dom.SLOT_DOMAINS = saved_slots
            dom.DOMAIN_SIBLINGS.clear()
            dom.DOMAIN_SIBLINGS.update(saved_siblings)

        return restore


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
