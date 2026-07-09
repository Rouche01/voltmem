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
from typing import Callable, Optional

from .domains import MemoryItem, DOMAIN_VOLATILITY, SOURCE_RELIABILITY
from .extract import HeuristicExtractor
from .store import MemoryStore
from .scoring import (
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
    similarity_fn : callable (query: str, content: str) -> float in [0, 1]
        Optional semantic-similarity function used at retrieval time. Defaults to
        the built-in keyword-overlap scorer. Pass an embedding-based scorer
        (see voltmem.embeddings.EmbeddingSimilarity) for production-quality
        semantic retrieval. VoltMem's volatility/freshness weighting is applied
        on top of whatever similarity function you provide.
    namespace: str = "default"
        Tenant/user key. All reads and writes on this layer are scoped to this
        namespace so one database can serve many users. Use for_user() to get a
        lightweight view for another tenant without opening a second connection.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        load: float = 1.0,
        goal_delta_default: float = 0.0,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        extractor: Optional[object] = None,
        relate_threshold: float = 0.55,
        namespace: str = "default",
    ):
        self._store = MemoryStore(db_path)
        self.load = load
        self.goal_delta_default = goal_delta_default
        self.namespace = namespace
        self._similarity_fn = similarity_fn or self._similarity
        # extractor powers the batteries-included remember(): infers domain and
        # contradiction so callers don't hand-supply them. Default is the
        # dependency-free heuristic; pass LLMExtractor() for higher quality.
        self._extractor = extractor or HeuristicExtractor(
            relate_similarity=relate_threshold)
        self.relate_threshold = relate_threshold

    # ── multi-tenant ──────────────────────────────────────────────────────────

    def for_user(self, namespace: str) -> "MemoryLayer":
        """Return a lightweight view of this layer scoped to `namespace`.

        The view shares the same underlying store/connection, similarity function
        and extractor — only reads and writes are isolated per tenant:

            mem = MemoryLayer("app.db")
            alice = mem.for_user("alice")
            bob   = mem.for_user("bob")
            alice.remember("I live in Berlin")   # invisible to bob

        Note: the store connection is shared, so closing any view (or the parent)
        closes it for all. Construct separate MemoryLayer objects if you need
        independent lifecycles.
        """
        view = object.__new__(MemoryLayer)
        view._store = self._store
        view.load = self.load
        view.goal_delta_default = self.goal_delta_default
        view.namespace = namespace
        view._similarity_fn = self._similarity_fn
        view._extractor = self._extractor
        view.relate_threshold = self.relate_threshold
        return view

    # ── primary write path ────────────────────────────────────────────────────

    def write(
        self,
        content: str,
        domain: str,
        source: str = "explicit_statement",
        tags: list[str] | None = None,
        goal_delta: float | None = None,
        at_time: float | None = None,
    ) -> WriteResult:
        """
        Write a new memory item unconditionally.
        Use this for bootstrapping known facts.
        For observations that may conflict with existing memory, use .observe().
        """
        now = at_time if at_time is not None else time.time()
        item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            domain=domain,
            source=source,
            namespace=self.namespace,
            tags=tags or [],
            goal_delta=goal_delta if goal_delta is not None else self.goal_delta_default,
            created_at=now,
            last_confirmed_at=now,
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
        at_time: float | None = None,
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
            return self.write(content, domain, source, tags, gd, at_time=at_time)

        # find the existing item this observation is actually about. With a
        # single item in the domain this is trivial (backward compatible); with
        # several, pick the best semantic match so distinct facts in the same
        # domain don't collide (e.g. two different core_preferences).
        candidate = self._select_candidate(content, existing)

        # ── escalation decision uses the volatility we knew BEFORE this
        #    observation. We judge the observation against the item's established
        #    behaviour, THEN fold it into the EMA. Updating the EMA first would
        #    let a single surprising signal inflate measured volatility and lower
        #    its own threshold — a self-fulfilling loop in which one confident
        #    blip overwrites an otherwise-stable fact.
        E_t, theta_t = escalation_score(
            candidate, mismatch_magnitude, source, gd, ld)
        escalate = E_t > theta_t

        # ── now fold this observation into the volatility EMA (reliability-
        #    weighted, single update). Future decisions benefit from the learned
        #    volatility; the current decision does not move its own goalposts.
        candidate.volatility_ema = update_volatility_ema(
            candidate, mismatch_magnitude, source)

        now = at_time if at_time is not None else time.time()

        if mismatch_magnitude < 0.15:
            # low mismatch: this is a confirmation, not a conflict
            candidate.repetition_count += 1
            candidate.last_confirmed_at = now
            self._store.update(candidate)
            return WriteResult(
                action="confirmed",
                item=candidate,
                detail=f"Repetition count now {candidate.repetition_count}",
            )

        if not escalate:
            # mismatch present but below threshold — log it, don't update content
            candidate.mismatch_count += 1
            self._store.update(candidate)
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
            namespace=self.namespace,
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

        return WriteResult(
            action="audited",
            item=new_item,
            detail=(f"E_t={E_t:.3f} > theta_t={theta_t:.3f}; "
                    f"old item {candidate.id[:8]} superseded."),
        )

    # ── batteries-included API ────────────────────────────────────────────────

    def remember(
        self,
        text: str,
        source: str = "explicit_statement",
        domain: str | None = None,
        tags: list[str] | None = None,
        at_time: float | None = None,
    ) -> WriteResult:
        """
        The one-call write path: hand it a raw statement and it figures out the
        rest. It finds whether the statement is about something already known
        (semantic match across all memories); if so it lets the volatility engine
        decide whether to update or keep the existing memory; if not, it
        classifies the domain and stores a new memory.

        No manual domain / mismatch_magnitude required — those are inferred by the
        configured extractor (heuristic by default; pass an LLMExtractor for
        higher quality). Provide `domain` to skip classification for new facts.

        Examples
        --------
        mem.remember("I live in Berlin")
        mem.remember("Actually I moved to Paris")   # updates the location memory
        mem.remember("I prefer concise answers")
        """
        match, sim = self._best_match_global(text, self.relate_threshold)
        if match is not None:
            mismatch = self._extractor.mismatch(text, match.content, sim)
            return self.observe(
                content=text,
                domain=match.domain,
                mismatch_magnitude=mismatch,
                source=source,
                tags=tags,
                at_time=at_time,
            )
        dom = domain or self._extractor.classify_domain(text)
        return self.write(text, domain=dom, source=source, tags=tags, at_time=at_time)

    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        now: float | None = None,
        use_staleness: bool = True,
    ) -> list[str]:
        """
        The one-call read path: return the most relevant, still-fresh memories as
        plain strings, ready to drop into a prompt. Ranking combines semantic
        similarity with volatility-aware freshness (stale volatile memories are
        down-ranked). Use retrieve() if you need items + scores.

        Pass `now` (unix seconds) to score staleness as of a specific date — useful
        when replaying historical benchmarks (e.g. LongMemEval question_date).
        """
        result = self.retrieve(query, top_k=top_k, min_score=min_score, now=now,
                               use_staleness=use_staleness)
        return [item.content for item in result.items]

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        domain: str | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
        now: float | None = None,
        use_staleness: bool = True,
    ) -> RetrieveResult:
        """
        Retrieve memories relevant to a query, ranked by a combination of
        semantic similarity (keyword-based by default) and freshness.

        For production, pass similarity_fn=EmbeddingSimilarity(...) to the
        MemoryLayer constructor for embedding-based semantic retrieval.

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
        now : float | None
            Evaluate staleness as of this unix timestamp (default: now).
        use_staleness : bool
            If False, rank by semantic similarity only (no volatility/freshness
            penalty). Useful as a baseline in retrieval benchmarks.
        """
        candidates = (
            self._active(domain=domain)
            if domain
            else self._active()
        )

        scored = []
        eval_now = now if now is not None else time.time()
        for item in candidates:
            sim = self._similarity_fn(query, item.content)
            if use_staleness:
                score = retrieval_score(item, sim, eval_now)
            else:
                score = float(sim)
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
        if item.namespace != self.namespace:
            return {"error": f"Item {item_id} not found"}
        now = time.time()
        stale = staleness(item, now)
        prot = protection_weight(item)
        return {
            "id": item.id,
            "content": item.content,
            "domain": item.domain,
            "namespace": item.namespace,
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
        """High-level summary of the memory store for this namespace."""
        all_items = self._active()
        by_domain: dict[str, int] = {}
        for item in all_items:
            by_domain[item.domain] = by_domain.get(item.domain, 0) + 1
        return {
            "namespace": self.namespace,
            "total_active_memories": len(all_items),
            "by_domain": by_domain,
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _active(self, domain: str | None = None) -> list[MemoryItem]:
        """Active memories scoped to this layer's namespace."""
        return self._store.all_active(namespace=self.namespace, domain=domain)

    def _find_domain_items(self, domain: str) -> list[MemoryItem]:
        return self._active(domain=domain)

    def _select_candidate(
        self, content: str, items: list[MemoryItem]
    ) -> MemoryItem:
        """Pick which existing item an observation refers to.

        One item → return it (backward compatible). Several → highest semantic
        similarity to `content`, tie-broken by recency, so distinct facts sharing
        a domain are not conflated.
        """
        if len(items) == 1:
            return items[0]
        return max(
            items,
            key=lambda it: (self._similarity_fn(content, it.content),
                            it.last_confirmed_at),
        )

    def _best_match_global(
        self, content: str, min_similarity: float
    ) -> tuple[Optional[MemoryItem], float]:
        """Best semantic match across ALL active memories (any domain).

        Used by the batteries-included remember() to decide whether an incoming
        statement is about something already known. Returns (item, similarity)
        or (None, best_similarity_seen) if nothing clears the threshold.
        """
        best_item: Optional[MemoryItem] = None
        best_sim = 0.0
        for it in self._active():
            sim = self._similarity_fn(content, it.content)
            if sim > best_sim:
                best_sim, best_item = sim, it
        if best_item is not None and best_sim >= min_similarity:
            return best_item, best_sim
        return None, best_sim

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

    def clear(self) -> None:
        """Delete all memories for this layer's namespace."""
        self._store.delete_namespace(self.namespace)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
