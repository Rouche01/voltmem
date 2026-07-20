"""
Domain observation tracking — prior calibration telemetry + optional auto-discovery.

Always records confirm / mismatch / audit / insert counts per (namespace, domain)
for ``MemoryLayer.domain_stats()``. When ``auto_discover=True``, also blends
empirical volatility estimates with hand-tuned priors at scoring time.
"""

from __future__ import annotations

from dataclasses import dataclass

from .domains import DOMAIN_VOLATILITY

MIN_OBSERVATIONS = 3
EMPIRICAL_WEIGHT = 0.3
RAMP_OBSERVATIONS = 10


@dataclass
class DomainStats:
    """Aggregated observation stats for one domain within a tenant namespace."""

    domain: str
    n_confirms: int = 0
    n_mismatches: int = 0
    n_supersedes: int = 0
    n_inserts: int = 0
    mismatch_sum: float = 0.0

    @property
    def total(self) -> int:
        """Escalation-relevant events (excludes cold inserts)."""
        return self.n_confirms + self.n_mismatches + self.n_supersedes

    @property
    def empirical_volatility(self) -> float:
        """Higher update + mismatch rates → higher learned volatility."""
        n = self.total
        if n == 0:
            return 0.5
        update_rate = self.n_supersedes / n
        mismatch_events = self.n_mismatches + self.n_supersedes
        mismatch_avg = (
            self.mismatch_sum / mismatch_events if mismatch_events else 0.0
        )
        raw = 0.3 * update_rate + 0.7 * mismatch_avg
        if self.n_confirms > self.n_supersedes and mismatch_avg < 0.2:
            raw *= 0.5
        return float(min(0.95, max(0.05, raw)))

    def record(self, action: str, mismatch: float = 0.0) -> None:
        if action == "confirmed":
            self.n_confirms += 1
        elif action == "logged_mismatch":
            self.n_mismatches += 1
            self.mismatch_sum += max(0.0, mismatch)
        elif action == "audited":
            self.n_supersedes += 1
            self.mismatch_sum += max(0.0, mismatch)
        elif action == "inserted":
            self.n_inserts += 1

    def to_row(self, namespace: str) -> dict:
        return {
            "namespace": namespace,
            "domain": self.domain,
            "n_confirms": self.n_confirms,
            "n_mismatches": self.n_mismatches,
            "n_supersedes": self.n_supersedes,
            "n_inserts": self.n_inserts,
            "mismatch_sum": self.mismatch_sum,
        }

    @classmethod
    def from_row(cls, row: dict) -> "DomainStats":
        return cls(
            domain=row["domain"],
            n_confirms=int(row["n_confirms"]),
            n_mismatches=int(row["n_mismatches"]),
            n_supersedes=int(row["n_supersedes"]),
            n_inserts=int(row.get("n_inserts") or 0),
            mismatch_sum=float(row["mismatch_sum"]),
        )

    def as_telemetry(self, prior: float) -> dict:
        """Public calibration row: counts + rates for stubborn vs twitchy priors."""
        decisions = self.total
        return {
            "prior": prior,
            "inserted": self.n_inserts,
            "confirmed": self.n_confirms,
            "logged_mismatch": self.n_mismatches,
            "audited": self.n_supersedes,
            "decisions": decisions,
            "audit_rate": (
                self.n_supersedes / decisions if decisions else 0.0
            ),
            "mismatch_rate": (
                self.n_mismatches / decisions if decisions else 0.0
            ),
            "confirm_rate": (
                self.n_confirms / decisions if decisions else 0.0
            ),
            "empirical_volatility": round(self.empirical_volatility, 4),
        }


def blend_volatility(prior: float, empirical: float, n_observations: int) -> float:
    """Prior-anchored blend; ramps empirical weight as evidence accumulates."""
    if n_observations < MIN_OBSERVATIONS:
        return prior
    w = min(EMPIRICAL_WEIGHT, EMPIRICAL_WEIGHT * (n_observations / RAMP_OBSERVATIONS))
    return float((1.0 - w) * prior + w * empirical)


class VolatilityTracker:
    """Per-tenant domain observation tracker backed by MemoryStore."""

    def __init__(self, store) -> None:
        self._store = store

    def record(
        self,
        namespace: str,
        domain: str,
        action: str,
        mismatch: float = 0.0,
    ) -> DomainStats:
        stats = self._store.get_domain_stats(namespace, domain)
        if stats is None:
            stats = DomainStats(domain=domain)
        stats.record(action, mismatch)
        self._store.upsert_domain_stats(namespace, stats)
        return stats

    def get_stats(self, namespace: str, domain: str) -> DomainStats | None:
        return self._store.get_domain_stats(namespace, domain)

    def all_stats(self, namespace: str) -> dict[str, DomainStats]:
        return self._store.all_domain_stats(namespace)

    def empirical_volatility(self, namespace: str, domain: str) -> float | None:
        stats = self.get_stats(namespace, domain)
        if stats is None or stats.total < MIN_OBSERVATIONS:
            return None
        return stats.empirical_volatility

    def resolve_volatility(
        self,
        namespace: str,
        domain: str,
        item_vol_ema: float,
    ) -> float:
        """Effective volatility for scoring: per-item EMA overrides domain blend."""
        if item_vol_ema >= 0:
            return item_vol_ema
        prior = DOMAIN_VOLATILITY.get(domain, 0.5)
        stats = self.get_stats(namespace, domain)
        if stats is None:
            return prior
        empirical = stats.empirical_volatility
        return blend_volatility(prior, empirical, stats.total)

    def summary(self, namespace: str) -> dict[str, dict]:
        """Human-readable domain discovery state for introspection."""
        out: dict[str, dict] = {}
        for domain, stats in self.all_stats(namespace).items():
            prior = DOMAIN_VOLATILITY.get(domain, 0.5)
            resolved = self.resolve_volatility(namespace, domain, -1.0)
            out[domain] = {
                "prior": prior,
                "empirical": round(stats.empirical_volatility, 4),
                "resolved": round(resolved, 4),
                "n_confirms": stats.n_confirms,
                "n_mismatches": stats.n_mismatches,
                "n_supersedes": stats.n_supersedes,
                "n_inserts": stats.n_inserts,
                "total": stats.total,
            }
        return out

    def telemetry(self, namespace: str) -> dict[str, dict]:
        """Per-domain calibration table for prior / audit-threshold health."""
        out: dict[str, dict] = {}
        for domain, stats in self.all_stats(namespace).items():
            prior = DOMAIN_VOLATILITY.get(domain, 0.5)
            out[domain] = stats.as_telemetry(prior)
        return out

    def clear_namespace(self, namespace: str) -> None:
        self._store.delete_domain_stats_namespace(namespace)
