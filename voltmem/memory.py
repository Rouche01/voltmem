"""
MemoryLayer — the primary public interface.

Usage:
    from voltmem import MemoryLayer

    mem = MemoryLayer("my_app.db")

    # Write a new memory
    mem.write(
        content="User prefers concise, direct responses",
        domain="core_preference",
        source="explicit_statement",
    )

    # Retrieve memories relevant to a query
    items = mem.retrieve("communication style")

    # Present a new observation that may contradict existing memory
    result = mem.observe(
        content="User asked for more detail and explanation today",
        domain="core_preference",
        mismatch_magnitude=0.6,
        source="weak_inference",
    )
    # result.action tells you what happened: "confirmed", "audited", "inserted"
"""

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .domains import MemoryItem, DOMAIN_VOLATILITY, SOURCE_RELIABILITY
from .store import MemoryStore
from .scoring import (
    should_escalate,
    staleness,
    retrieval_score,
    update_volatility_ema,
    protection_weight,
    escalation_score,
)


# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class WriteResult:
    action:  str            # "inserted" | "confirmed" | "audited" | "superseded"
    item:    MemoryItem
    detail:  str = ""


@dataclass
class RetrieveResult:
    items:   list[MemoryItem]
    scores:  list[float]    # retrieval_score per item, same order


# ── main class ────────────────────────────────────────────────────────────────

class MemoryLayer:
    """
    Volatility-adjusted persistent memory layer.

    Plug into any system by passing text observations through .observe()
    and fetching relevant context via .retrieve().

    Parameters
    ----------
    db_path : str or Path
        SQLite database path. Use ":memory:" for an in-process ephemeral store.
    load : float
        Cognitive/compute load scalar (L_t). Raise this when the system is
        under time pressure to suppress low-confidence audits.
    goal_delta_default : float
        Default G_t value used when the caller doesn't supply one.
        Positive = system is in a goal-directed mode (amplifies escalation
        for contradicting items). 0 = neutral.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        load: float = 1.0,
        goal_delta_default: float = 0.0,
    ):
        self._store = MemoryStore(db_path)
        self.load = load
        self.goal_delta_default = goal_delta_default

    # ── primary write path ────────────────────────────────────────────────────

    def write(
        self,
        content: str,
        domain: str,
        source: str = "explicit_statement",
        tags: list[str] | None = None,
        goal_delta: float | None = None,
    ) -> WriteResult:
        """
        Write a new memory item unconditionally.
        Use this for bootstrapping known facts.
        For observations that may conflict with existing memory, use .observe().
        """
        item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            domain=domain,
            source=source,
            tags=tags or [],
            goal_delta=goal_delta if goal_delta is not None else self.goal_delta_default,
            created_at=time.time(),
            last_confirmed_at=time.time(),
        )
        self._store.insert(item)
        return WriteResult(action="inserted", item=item)

    # ── primary observe path ──────────────────────────────────────────────────

    def observe(
        self,
        content: str,
        domain: str,
        mismatch_magnitude: float = 0.0,
        source: str = "explicit_statement",
        tags: list[str] | None = None,
        goal_delta: float | None = None,
        load: float | None = None,
    ) -> WriteResult:
        """
        Present a new observation to the memory layer.

        The layer will:
        1. Look for existing active items in the same domain with similar content.
        2. Compute the escalation score E_t against any conflicting item.
        3. If E_t > theta_t  → AUDIT: update the existing item or supersede it.
        4. If E_t <= theta_t → CONFIRM: increment repetition count, no content change.
        5. If no existing item found → INSERT as a new memory.

        Parameters
        ----------
        content : str
            The new observed fact / preference / context.
        domain : str
            Memory domain (see domains.DOMAIN_VOLATILITY for options).
        mismatch_magnitude : float [0,1]
            How strongly this observation contradicts what's stored.
            0 = confirming, 1 = directly contradicting.
        source : str
            Reliability of the source (see domains.SOURCE_RELIABILITY).
        goal_delta : float [-1, 1]
            Whether auditing/updating this memory helps (+) or hurts (-) the
            current goal. 0 = neutral.
        load : float
            Override instance-level load for this call.
        """
        gd = goal_delta if goal_delta is not None else self.goal_delta_default
        ld = load if load is not None else self.load

        # ── pre-update: measure mismatch BEFORE any write ─────────────────────
        existing = self._find_domain_items(domain)

        if not existing:
            return self.write(content, domain, source, tags, gd)

        # find the most relevant existing item (simple: most recently confirmed)
        candidate = max(existing, key=lambda x: x.last_confirmed_at)

        # ── update volatility EMA on candidate (pre-update, avoids circularity)
        if mismatch_magnitude > 0:
            candidate.volatility_ema = update_volatility_ema(
                candidate, mismatch_magnitude)

        # ── escalation decision ───────────────────────────────────────────────
        escalate = should_escalate(
            item=candidate,
            mismatch_magnitude=mismatch_magnitude,
            source=source,
            goal_delta=gd,
            load=ld,
        )

        now = time.time()

        if mismatch_magnitude < 0.15:
            # low mismatch: this is a confirmation, not a conflict
            candidate.repetition_count += 1
            candidate.last_confirmed_at = now
            candidate.volatility_ema = update_volatility_ema(candidate, mismatch_magnitude)
            self._store.update(candidate)
            return WriteResult(
                action="confirmed",
                item=candidate,
                detail=f"Repetition count now {candidate.repetition_count}",
            )

        if not escalate:
            # mismatch present but below threshold — log it, don't update content
            candidate.mismatch_count += 1
            candidate.volatility_ema = update_volatility_ema(candidate, mismatch_magnitude)
            self._store.update(candidate)
            E_t, theta_t = escalation_score(candidate, mismatch_magnitude, source, gd, ld)
            return WriteResult(
                action="logged_mismatch",
                item=candidate,
                detail=(f"E_t={E_t:.3f} <= theta_t={theta_t:.3f}; "
                        f"mismatch logged but content retained. "
                        f"Cumulative mismatches: {candidate.mismatch_count}"),
            )

        # escalated: audit and supersede
        candidate.superseded_by = "pending"
        candidate.last_audited_at = now
        self._store.update(candidate)

        new_item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            domain=domain,
            source=source,
            tags=tags or candidate.tags,
            repetition_count=1,
            volatility_ema=candidate.volatility_ema,  # carry forward EMA
            goal_delta=gd,
            created_at=now,
            last_confirmed_at=now,
        )
        self._store.insert(new_item)

        # link old → new
        candidate.superseded_by = new_item.id
        self._store.update(candidate)

        E_t, theta_t = escalation_score(candidate, mismatch_magnitude, source, gd, ld)
        return WriteResult(
            action="audited",
            item=new_item,
            detail=(f"E_t={E_t:.3f} > theta_t={theta_t:.3f}; "
                    f"old item {candidate.id[:8]} superseded."),
        )

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        domain: str | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> RetrieveResult:
        """
        Retrieve memories relevant to a query, ranked by a combination of
        semantic similarity (keyword-based by default) and freshness.

        For production: swap _similarity() with an embedding-based scorer.

        Parameters
        ----------
        query : str
            The current context/question to match against.
        domain : str | None
            Restrict to a single domain if provided.
        top_k : int
            Maximum number of items to return.
        min_score : float
            Minimum retrieval score to include an item.
        """
        candidates = (
            self._store.all_active(domain=domain)
            if domain
            else self._store.all_active()
        )

        scored = []
        now = time.time()
        for item in candidates:
            sim = self._similarity(query, item.content)
            score = retrieval_score(item, sim, now)
            if score >= min_score:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        return RetrieveResult(
            items=[item for _, item in top],
            scores=[score for score, _ in top],
        )

    # ── introspection ─────────────────────────────────────────────────────────

    def inspect(self, item_id: str) -> dict:
        """Return a human-readable breakdown of an item's scoring state."""
        item = self._store.get(item_id)
        if not item:
            return {"error": f"Item {item_id} not found"}
        now = time.time()
        stale = staleness(item, now)
        prot = protection_weight(item)
        return {
            "id": item.id,
            "content": item.content,
            "domain": item.domain,
            "source": item.source,
            "repetition_count": item.repetition_count,
            "effective_volatility": item.effective_volatility,
            "protection_weight": prot,
            "staleness": round(stale, 4),
            "mismatch_count": item.mismatch_count,
            "active": item.is_active,
            "age_days": round((now - item.created_at) / 86400, 2),
            "days_since_confirmed": round((now - item.last_confirmed_at) / 86400, 2),
        }

    def summary(self) -> dict:
        """High-level summary of the memory store."""
        all_items = self._store.all_active()
        by_domain: dict[str, int] = {}
        for item in all_items:
            by_domain[item.domain] = by_domain.get(item.domain, 0) + 1
        return {
            "total_active_memories": len(all_items),
            "by_domain": by_domain,
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _find_domain_items(self, domain: str) -> list[MemoryItem]:
        return self._store.all_active(domain=domain)

    @staticmethod
    def _similarity(query: str, content: str) -> float:
        """
        Minimal keyword overlap similarity.
        Replace with cosine similarity over embeddings for production.
        """
        q_words = set(query.lower().split())
        c_words = set(content.lower().split())
        if not q_words or not c_words:
            return 0.0
        overlap = q_words & c_words
        return len(overlap) / max(len(q_words), len(c_words))

    def close(self):
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
