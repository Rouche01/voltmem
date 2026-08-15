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

import copy
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from .domains import (
    MemoryItem,
    DOMAIN_VOLATILITY,
    DOMAIN_SIBLINGS,
    SLOT_LINK_FLOOR,
)
from . import domains as _domains
from .vector_index import VectorIndex, create_vector_index
from .extract import HeuristicExtractor
from .store import MemoryStore
from .discovery import VolatilityTracker
from .verify import resolve_link_verifier
from .structure import (
    HeuristicStructuredExtractor,
    join_structured,
    facts_from_dicts,
    facts_to_dicts,
    normalize_attribute,
    normalize_subject,
)
from .scoring import (
    staleness,
    retrieval_score,
    update_volatility_ema,
    update_surprise_ema,
    update_mismatch_expectation,
    protection_weight,
    escalation_decision,
    similarity_spread,
    freshness_mix,
    normalize_escalation_mode,
    recent_surprise,
    surprise_mode_scale,
    expected_mismatch,
    mismatch_sigma,
)


def memory_item_to_payload(item: MemoryItem) -> dict:
    """JSON-friendly snapshot for maintenance purge rollback."""
    return {
        "id": item.id,
        "content": item.content,
        "domain": item.domain,
        "source": item.source,
        "namespace": item.namespace,
        "event_id": item.event_id,
        "modality": item.modality,
        "expires_at": item.expires_at,
        "repetition_count": item.repetition_count,
        "volatility_ema": item.volatility_ema,
        "surprise_ema": item.surprise_ema,
        "surprise_at": item.surprise_at,
        "mismatch_ema": item.mismatch_ema,
        "mismatch_var": item.mismatch_var,
        "mismatch_count": item.mismatch_count,
        "goal_delta": item.goal_delta,
        "created_at": item.created_at,
        "last_confirmed_at": item.last_confirmed_at,
        "last_audited_at": item.last_audited_at,
        "tags": list(item.tags),
        "facts": list(item.facts or []),
        "superseded_by": item.superseded_by,
    }


def _memory_item_from_payload(payload: dict) -> MemoryItem:
    return MemoryItem(
        id=str(payload["id"]),
        content=str(payload["content"]),
        domain=str(payload["domain"]),
        source=str(payload.get("source") or "system_generated"),
        namespace=str(payload.get("namespace") or "default"),
        event_id=payload.get("event_id"),
        modality=payload.get("modality"),
        expires_at=payload.get("expires_at"),
        repetition_count=int(payload.get("repetition_count") or 1),
        volatility_ema=float(payload.get("volatility_ema") if payload.get("volatility_ema") is not None else -1.0),
        surprise_ema=float(payload.get("surprise_ema") or 0.0),
        surprise_at=float(payload.get("surprise_at") or 0.0),
        mismatch_ema=float(payload.get("mismatch_ema") if payload.get("mismatch_ema") is not None else -1.0),
        mismatch_var=float(payload.get("mismatch_var") if payload.get("mismatch_var") is not None else -1.0),
        mismatch_count=int(payload.get("mismatch_count") or 0),
        goal_delta=float(payload.get("goal_delta") or 0.0),
        created_at=float(payload.get("created_at") or 0.0),
        last_confirmed_at=float(payload.get("last_confirmed_at") or 0.0),
        last_audited_at=float(payload.get("last_audited_at") or 0.0),
        tags=list(payload.get("tags") or []),
        facts=list(payload.get("facts") or []),
        superseded_by=payload.get("superseded_by"),
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
    auto_discover : bool
        When True, blend empirical per-domain volatility into scoring.
    escalation_mode : str
        ``"composite"`` (default) uses allostatic only for an explicit high-M
        correction or an unexpected residual against a learned Ê; otherwise
        homeostatic. ``"homeostatic"`` (``"current"`` is an alias) keeps
        E_t · V_d / θ(V_d). ``"allostatic"`` drops V_d from E_t and scales
        θ by recent surprise s(m).
    escalation_v_exp : float, optional
        Override the V_d exponent in E_t (1.0 = homeostatic, 0.0 = allostatic).
    escalation_mode_scale : bool, optional
        Override whether θ is scaled by s(m). Together with ``escalation_v_exp``
        this allows ablating the two allostatic ingredients separately.
    link_verifier : LinkVerifier, callable, ``"auto"``, or None
        Precision half of linking. ``"auto"`` (default) attaches the local LLM
        verifier when an embedder is present so sleeptime ``reconcile_twins``
        can judge twins. Keyword-only layers stay on the threshold ladder.
        Pass ``None`` to force the ladder and skip sleeptime verification.
    verify_on_write : bool or None
        When to ask the verifier. Default ``None`` means **sleeptime** for
        ``"auto"`` (millisecond ``remember()``, 14B in ``reconcile_twins``)
        and **write** when a verifier is passed in. Set ``True`` to ask on
        grey ``remember()`` calls (no heuristic cards, neighbour above the
        recall bar). A heuristic refusal never goes to the model.
    link_recall_bar : float
        Similarity floor for two-stage recall (default 0.20). Safe this low
        only because the verifier has the final say.
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
        vector_index: Union[VectorIndex, str, None] = "auto",
        embed_fn: Optional[Callable[[str], list[float]]] = None,
        candidate_multiplier: int = 5,
        auto_discover: bool = False,
        # Default: composite (mode_default_eval.py, 2026-08-15).
        escalation_mode: str = "composite",
        escalation_v_exp: float | None = None,
        escalation_mode_scale: bool | None = None,
        link_verifier: object | None = "auto",
        verify_on_write: bool | None = None,
        link_recall_bar: float = 0.20,
        link_recall_top_k: int = 3,
        ollama_url: str = "http://localhost:11434",
        llm_model: str = "qwen2.5-coder:14b",
    ):
        self._store = MemoryStore(db_path)
        self.load = load
        self.goal_delta_default = goal_delta_default
        self.namespace = namespace
        self.auto_discover = auto_discover
        self.escalation_mode = normalize_escalation_mode(escalation_mode)
        self.escalation_v_exp = escalation_v_exp
        self.escalation_mode_scale = escalation_mode_scale
        # Always track write-path actions for prior calibration telemetry;
        # auto_discover only controls whether empirical V_d is blended at score time.
        self._tracker = VolatilityTracker(self._store)
        self._similarity_fn = similarity_fn or self._similarity
        if embed_fn is None:
            embed_fn = getattr(self._similarity_fn, "embed", None)
        self._embed_fn = embed_fn
        self.candidate_multiplier = max(1, candidate_multiplier)
        if isinstance(vector_index, str):
            self._vector_index = create_vector_index(
                vector_index,
                db_path,
                has_embedder=self._embed_fn is not None,
            )
        else:
            self._vector_index = vector_index
        # extractor powers the batteries-included remember(): infers domain and
        # contradiction so callers don't hand-supply them. Default is the
        # dependency-free heuristic; pass LLMExtractor() for higher quality.
        self._extractor = extractor or HeuristicExtractor(
            relate_similarity=relate_threshold)
        self.relate_threshold = relate_threshold
        # Two-stage linking. Keyword-only layers stay on the threshold ladder.
        # An embedder auto-attaches the local verifier for sleeptime twin
        # reconciliation. Live ``remember()`` does not ask it unless
        # verify_on_write is on (or a verifier was passed in).
        self._link_verifier = resolve_link_verifier(
            link_verifier,
            has_embedder=self._embed_fn is not None,
            ollama_url=ollama_url,
            llm_model=llm_model,
        )
        self.link_recall_bar = link_recall_bar
        self.link_recall_top_k = max(1, link_recall_top_k)
        # Cheap write-path join for known frames (city, birth year, skill, job).
        # An explicit verifier still runs as given; heuristic only intercepts
        # the auto/ladder paths so tests that inject a verifier keep seeing it.
        self._heuristic_extractor = HeuristicStructuredExtractor()
        self._use_heuristic_link = link_verifier in ("auto", None, False)
        if verify_on_write is None:
            self.verify_on_write = link_verifier not in ("auto", None, False)
        else:
            self.verify_on_write = bool(verify_on_write)

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
        view.auto_discover = self.auto_discover
        view.escalation_mode = self.escalation_mode
        view.escalation_v_exp = self.escalation_v_exp
        view.escalation_mode_scale = self.escalation_mode_scale
        view._tracker = self._tracker
        view._similarity_fn = self._similarity_fn
        view._extractor = self._extractor
        view.relate_threshold = self.relate_threshold
        view._vector_index = self._vector_index
        view._embed_fn = self._embed_fn
        view.candidate_multiplier = self.candidate_multiplier
        view._link_verifier = self._link_verifier
        view.verify_on_write = self.verify_on_write
        view.link_recall_bar = self.link_recall_bar
        view.link_recall_top_k = self.link_recall_top_k
        view._heuristic_extractor = self._heuristic_extractor
        view._use_heuristic_link = self._use_heuristic_link
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
        event_id: str | None = None,
        modality: str | None = None,
        expires_at: float | None = None,
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
            event_id=event_id,
            modality=modality,
            expires_at=expires_at,
            created_at=now,
            last_confirmed_at=now,
        )
        self._stamp_facts(item)
        self._store.insert(item)
        self._index_upsert(item)
        self._record_domain_observation("inserted", domain)
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
        event_id: str | None = None,
        modality: str | None = None,
        expires_at: float | None = None,
        force_update: bool = False,
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
        event_id : str | None
            Shared key when this observation is one facet of a multi-facet event.
        modality : str | None
            Kind of content (text / image / audio / sensor / structured).
        expires_at : float | None
            Unix timestamp when this fact becomes invalid (optional TTL).
        force_update : bool
            When True, bypass escalation math and always audit/supersede.
            Intended for maintenance-driven reorganizations that have already
            done their own threshold checking. Use with care.
        """
        gd = goal_delta if goal_delta is not None else self.goal_delta_default
        ld = load if load is not None else self.load

        # ── pre-update: measure mismatch BEFORE any write ─────────────────────
        existing = self._find_domain_items(domain)

        if not existing:
            return self.write(content, domain, source, tags, gd, at_time=at_time,
                             event_id=event_id, modality=modality, expires_at=expires_at)

        # find the existing item this observation is actually about. With a
        # single item in the domain this is trivial (backward compatible); with
        # several, pick the best semantic match so distinct facts in the same
        # domain don't collide (e.g. two different core_preferences).
        candidate = self._select_candidate(content, existing)
        scoring_item = self._resolve_item_for_scoring(candidate)

        # ── escalation decision uses the volatility we knew BEFORE this
        #    observation. We judge the observation against the item's established
        #    behaviour, THEN fold it into the EMA. Updating the EMA first would
        #    let a single surprising signal inflate measured volatility and lower
        #    its own threshold — a self-fulfilling loop in which one confident
        #    blip overwrites an otherwise-stable fact.
        # Cap θ / cumulative overrides live in escalation_decision — do not
        # compare raw E_t > theta_t here or medium-stable domains stay stuck.
        now = at_time if at_time is not None else time.time()

        if force_update:
            escalate, E_t, theta_t = True, 1.0, 0.0
        else:
            escalate, E_t, theta_t = escalation_decision(
                scoring_item, mismatch_magnitude, source, gd, ld,
                mode=self.escalation_mode,
                v_exp=self.escalation_v_exp,
                mode_scale=self.escalation_mode_scale,
                now=now)

        # ── now fold this observation into the volatility EMA (reliability-
        #    weighted, single update). Future decisions benefit from the learned
        #    volatility; the current decision does not move its own goalposts.
        # Judge against the anticipation we had BEFORE this observation, then
        # fold M into Ê/σ and the unexpected residual into surprise_ema.
        # Updating anticipation first would let a spike explain itself away.
        candidate.surprise_ema = update_surprise_ema(
            candidate, mismatch_magnitude, source, now=now)
        candidate.surprise_at = now
        candidate.mismatch_ema, candidate.mismatch_var = update_mismatch_expectation(
            candidate, mismatch_magnitude, source)
        candidate.volatility_ema = update_volatility_ema(
            candidate, mismatch_magnitude, source)

        if not force_update and mismatch_magnitude < 0.15:
            # low mismatch: this is a confirmation, not a conflict
            candidate.repetition_count += 1
            candidate.last_confirmed_at = now
            self._store.update(candidate)
            self._record_domain_observation("confirmed", domain, mismatch_magnitude)
            return WriteResult(
                action="confirmed",
                item=candidate,
                detail=f"Repetition count now {candidate.repetition_count}",
            )

        if not escalate:
            # mismatch present but below threshold — log it, don't update content
            candidate.mismatch_count += 1
            self._store.update(candidate)
            self._store.append_mismatch_evidence(
                candidate.id,
                self.namespace,
                content,
                mismatch_magnitude=mismatch_magnitude,
                source=source,
                created_at=now,
            )
            self._record_domain_observation(
                "logged_mismatch", domain, mismatch_magnitude)
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
            surprise_ema=0.0,  # re-settle after absorbing the change
            mismatch_ema=-1.0,  # new fact starts with the confirm prior
            mismatch_var=-1.0,
            goal_delta=gd,
            event_id=event_id if event_id is not None else candidate.event_id,
            modality=modality if modality is not None else candidate.modality,
            expires_at=expires_at,
            created_at=now,
            last_confirmed_at=now,
        )
        self._stamp_facts(new_item)
        self._store.insert(new_item)

        # link old → new
        candidate.superseded_by = new_item.id
        self._store.update(candidate)
        self._index_delete(candidate.id)
        self._index_upsert(new_item)
        self._record_domain_observation("audited", domain, mismatch_magnitude)

        return WriteResult(
            action="audited",
            item=new_item,
            detail=(f"E_t={E_t:.3f} vs theta_t={theta_t:.3f}; "
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
        event_id: str | None = None,
        modality: str | None = None,
        expires_at: float | None = None,
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
        if self._use_heuristic_link:
            h_item, h_sim, h_kind = self._heuristic_match(text)
            if h_kind == "hit":
                return self.observe(
                    content=text,
                    domain=h_item.domain,
                    mismatch_magnitude=self._extractor.mismatch(
                        text, h_item.content, h_sim),
                    source=source,
                    tags=tags,
                    at_time=at_time,
                    event_id=event_id,
                    modality=modality,
                    expires_at=expires_at,
                )
            if h_kind == "keep_both":
                return self._remember_insert(
                    text, source, domain, tags, at_time,
                    event_id, modality, expires_at)
            # no_cards: grey / insert — never send a heuristic refusal to 14B.

        if self.verify_on_write and self._link_verifier is not None:
            verified, vsim, kind = self._verified_match(text)
            if kind == "hit":
                return self.observe(
                    content=text,
                    domain=verified.domain,
                    mismatch_magnitude=self._extractor.mismatch(
                        text, verified.content, vsim),
                    source=source,
                    tags=tags,
                    at_time=at_time,
                    event_id=event_id,
                    modality=modality,
                    expires_at=expires_at,
                )
            if kind in ("keep_both", "no_recall"):
                return self._remember_insert(
                    text, source, domain, tags, at_time,
                    event_id, modality, expires_at)
            # infra_fail: the ladder can still link a pair a dead model missed.

        if self._link_verifier is not None and not self.verify_on_write:
            # Sleeptime mode: grey frames insert. reconcile_twins asks later.
            return self._remember_insert(
                text, source, domain, tags, at_time,
                event_id, modality, expires_at)

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
                event_id=event_id,
                modality=modality,
                expires_at=expires_at,
            )
        dom = domain or self._extractor.classify_domain(text)
        slot_match, slot_sim = self._best_match_in_slot(text, dom)
        if slot_match is not None:
            mismatch = self._extractor.mismatch(
                text, slot_match.content, slot_sim)
            return self.observe(
                content=text,
                domain=slot_match.domain,
                mismatch_magnitude=mismatch,
                source=source,
                tags=tags,
                at_time=at_time,
                event_id=event_id,
                modality=modality,
                expires_at=expires_at,
            )
        return self.write(
            text,
            domain=dom,
            source=source,
            tags=tags,
            at_time=at_time,
            event_id=event_id,
            modality=modality,
            expires_at=expires_at,
        )

    def _remember_insert(
        self,
        text: str,
        source: str,
        domain: str | None,
        tags: list[str] | None,
        at_time: float | None,
        event_id: str | None,
        modality: str | None,
        expires_at: float | None,
    ) -> WriteResult:
        return self.write(
            text,
            domain=domain or self._extractor.classify_domain(text),
            source=source,
            tags=tags,
            at_time=at_time,
            event_id=event_id,
            modality=modality,
            expires_at=expires_at,
        )

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

    def add_event(
        self,
        event_id: str,
        facets: list[dict],
        source: str = "explicit_statement",
        tags: list[str] | None = None,
        goal_delta: float | None = None,
        at_time: float | None = None,
    ) -> list[WriteResult]:
        """
        Store a multi-facet observation as N linked memory items.

        Each facet is a dict with at minimum ``content`` and ``domain`` keys.
        Optional keys per facet: ``modality``, ``ttl_seconds``, ``expires_at``.

        Example::

            mem.add_event("tick-50ms-001", facets=[
                {"content": "corridor map patch A12", "domain": "spatial_map", "modality": "structured"},
                {"content": "battery 37%", "domain": "power_state", "modality": "sensor"},
                {"content": "go to charging dock", "domain": "current_task", "modality": "text"},
            ])

        Each facet is written unconditionally (no escalation). For facets that
        supersede existing memory, call ``observe()`` individually.
        """
        results: list[WriteResult] = []
        now = at_time if at_time is not None else time.time()
        for facet in facets:
            expires = facet.get("expires_at")
            ttl = facet.get("ttl_seconds")
            if expires is None and ttl is not None:
                expires = now + ttl
            result = self.write(
                content=facet["content"],
                domain=facet["domain"],
                source=source,
                tags=tags,
                goal_delta=goal_delta,
                at_time=at_time,
                event_id=event_id,
                modality=facet.get("modality"),
                expires_at=expires,
            )
            results.append(result)
        return results

    def retrieve_by_event(self, event_id: str) -> list[MemoryItem]:
        """Return all items (active and superseded) for a given event, ordered by creation time."""
        return self._store.get_by_event(self.namespace, event_id)

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
        candidates = self._retrieve_candidates(query, domain, top_k)

        scored = []
        eval_now = now if now is not None else time.time()
        # Plateau detection on the top similarity pool (not the full corpus).
        pool_n = max(top_k * self.candidate_multiplier, top_k, 2)
        by_sim = sorted(candidates, key=lambda x: x[1], reverse=True)[:pool_n]
        probe_sims = [s for _, s in by_sim]
        if use_staleness and len(probe_sims) >= 2:
            mix = freshness_mix(similarity_spread(probe_sims))
        else:
            mix = 1.0

        for item, sim in candidates:
            scoring_item = self._resolve_item_for_scoring(item)
            if scoring_item.is_expired:
                continue
            if use_staleness:
                score = retrieval_score(scoring_item, sim, eval_now, mix=mix)
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

    def remove(self, item_id: str) -> bool:
        """Delete one memory and drop it from the vector index."""
        if not self._store.delete(item_id, self.namespace):
            return False
        self._index_delete(item_id)
        return True

    def purge_expired(self, *, now: float | None = None) -> int:
        """Hard-delete expired rows and drop them from the vector index.

        Returns the number of rows deleted. Prefer this over
        ``MemoryStore.purge_expired`` so embeddings cannot orphan.
        """
        ids = self._store.purge_expired(self.namespace, now=now)
        for item_id in ids:
            self._index_delete(item_id)
        return len(ids)

    def start_maintenance_run(self, *, dry_run: bool = False) -> str:
        """Open a ledgered maintenance run; returns ``run_id``."""
        return self._store.start_maintenance_run(
            self.namespace, dry_run=dry_run)

    def finish_maintenance_run(self, run_id: str) -> None:
        self._store.finish_maintenance_run(run_id)

    def rollback_maintenance(self, run_id: str) -> dict:
        """Undo a maintenance run recorded in the ledger.

        Supports:
          * ``supersede`` — reactivate ``old_id``, retire ``new_id`` (when still valid)
          * ``retire`` — reactivate ``old_id`` without touching the keeper
          * ``purge`` — re-insert the snapshotted item

        Returns a summary dict. Raises ``KeyError`` if the run is missing or
        wrong namespace; ``ValueError`` if already rolled back.
        """
        run = self._store.get_maintenance_run(run_id)
        if run is None:
            raise KeyError(f"Unknown maintenance run: {run_id!r}")
        if run["namespace"] != self.namespace:
            raise KeyError(f"Unknown maintenance run: {run_id!r}")
        if run.get("rolled_back_at"):
            raise ValueError(f"Maintenance run {run_id} already rolled back")
        if run.get("dry_run"):
            raise ValueError(f"Maintenance run {run_id} was a dry_run (nothing to undo)")

        restored = 0
        retired = 0
        skipped: list[dict] = []

        for action in reversed(self._store.list_maintenance_actions(run_id)):
            atype = action["action_type"]
            if atype == "supersede":
                old_id = action.get("old_id")
                new_id = action.get("new_id")
                if not old_id or not new_id:
                    skipped.append({**action, "reason": "missing ids"})
                    continue
                old = self._store.get(old_id)
                new = self._store.get(new_id)
                if old is None or new is None:
                    skipped.append({**action, "reason": "item missing"})
                    continue
                if old.namespace != self.namespace or new.namespace != self.namespace:
                    skipped.append({**action, "reason": "namespace mismatch"})
                    continue
                # Only safe if new is still the active tip and old points at new
                if new.superseded_by is not None:
                    skipped.append({**action, "reason": "new item already superseded"})
                    continue
                if old.superseded_by != new_id:
                    skipped.append({**action, "reason": "old item no longer points at new"})
                    continue
                new.superseded_by = old_id
                old.superseded_by = None
                self._store.update(new)
                self._store.update(old)
                self._index_delete(new_id)
                self._index_upsert(old)
                restored += 1
                retired += 1
            elif atype == "retire":
                old_id = action.get("old_id")
                keeper_id = action.get("new_id")
                if not old_id or not keeper_id:
                    skipped.append({**action, "reason": "missing ids"})
                    continue
                old = self._store.get(old_id)
                if old is None:
                    skipped.append({**action, "reason": "item missing"})
                    continue
                if old.namespace != self.namespace:
                    skipped.append({**action, "reason": "namespace mismatch"})
                    continue
                if old.superseded_by != keeper_id:
                    skipped.append({**action, "reason": "old item no longer points at keeper"})
                    continue
                old.superseded_by = None
                self._store.update(old)
                self._index_upsert(old)
                restored += 1
            elif atype == "purge":
                payload = action.get("payload") or {}
                item_id = action.get("old_id") or payload.get("id")
                if not item_id or not payload:
                    skipped.append({**action, "reason": "missing purge snapshot"})
                    continue
                existing = self._store.get(item_id)
                if existing is not None:
                    skipped.append({**action, "reason": "item already exists"})
                    continue
                item = _memory_item_from_payload(payload)
                if item.namespace != self.namespace:
                    skipped.append({**action, "reason": "namespace mismatch"})
                    continue
                self._store.insert(item)
                self._index_upsert(item)
                restored += 1
            else:
                skipped.append({**action, "reason": f"unsupported action_type {atype}"})

        self._store.mark_maintenance_rolled_back(run_id)
        return {
            "run_id": run_id,
            "restored": restored,
            "retired": retired,
            "skipped": skipped,
        }

    # ── introspection ─────────────────────────────────────────────────────────

    def inspect(self, item_id: str) -> dict:
        """Return a human-readable breakdown of an item's scoring state."""
        item = self._store.get(item_id)
        if not item:
            return {"error": f"Item {item_id} not found"}
        if item.namespace != self.namespace:
            return {"error": f"Item {item_id} not found"}
        scoring_item = self._resolve_item_for_scoring(item)
        now = time.time()
        stale = staleness(scoring_item, now)
        prot = protection_weight(scoring_item)
        out = {
            "id": item.id,
            "content": item.content,
            "domain": item.domain,
            "namespace": item.namespace,
            "source": item.source,
            "repetition_count": item.repetition_count,
            "effective_volatility": scoring_item.effective_volatility,
            "protection_weight": prot,
            "staleness": round(stale, 4),
            "mismatch_count": item.mismatch_count,
            "surprise_ema": round(item.surprise_ema, 4),
            "expected_mismatch": round(expected_mismatch(item), 4),
            "mismatch_sigma": round(mismatch_sigma(item), 4),
            "recent_surprise": round(recent_surprise(item, now), 4),
            "mode_scale": round(surprise_mode_scale(item, now), 4),
            "escalation_mode": self.escalation_mode,
            "link_verifier": self._link_verifier is not None,
            "verify_on_write": self.verify_on_write,
            "active": item.is_active,
            "age_days": round((now - item.created_at) / 86400, 2),
            "days_since_confirmed": round((now - item.last_confirmed_at) / 86400, 2),
            "facts": list(item.facts or []),
        }
        stats = self._tracker.get_stats(self.namespace, item.domain)
        if stats is not None:
            out["domain_stats"] = {
                "empirical_volatility": round(stats.empirical_volatility, 4),
                "n_confirms": stats.n_confirms,
                "n_mismatches": stats.n_mismatches,
                "n_supersedes": stats.n_supersedes,
                "n_inserts": stats.n_inserts,
            }
        return out

    def domain_stats(self) -> dict[str, dict]:
        """
        Prior-calibration telemetry: per-domain write-path action counts and rates.

        Returns a dict suitable for histograms / ops dashboards, e.g.::

            {
              "location": {
                "prior": 0.6,
                "inserted": 3,
                "confirmed": 10,
                "logged_mismatch": 2,
                "audited": 4,
                "audit_rate": 0.25,
                ...
              },
            }

        Always available (does not require ``auto_discover=True``). Use audit_rate
        and mismatch_rate to spot stubborn vs twitchy domain priors.
        """
        return self._tracker.telemetry(self.namespace)

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
            "auto_discover": self.auto_discover,
            "domain_discovery": self._tracker.summary(self.namespace),
            "domain_stats": self.domain_stats(),
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _resolve_item_for_scoring(self, item: MemoryItem) -> MemoryItem:
        """Apply learned domain volatility when auto_discover is enabled."""
        if not self.auto_discover:
            return item
        resolved = self._tracker.resolve_volatility(
            self.namespace, item.domain, item.volatility_ema)
        if item.volatility_ema >= 0:
            current = item.volatility_ema
        else:
            current = DOMAIN_VOLATILITY.get(item.domain, 0.5)
        if abs(resolved - current) < 1e-9:
            return item
        resolved_item = copy.copy(item)
        resolved_item.volatility_ema = resolved
        return resolved_item

    def _record_domain_observation(
        self, action: str, domain: str, mismatch: float = 0.0
    ) -> None:
        self._tracker.record(self.namespace, domain, action, mismatch)

    def _retrieve_candidates(
        self, query: str, domain: str | None, top_k: int
    ) -> list[tuple[MemoryItem, float]]:
        """ANN pre-filter when a vector index is configured; else full scan."""
        if self._vector_index is not None and self._embed_fn is not None:
            pool = max(top_k * self.candidate_multiplier, top_k)
            hits = self._vector_index.search(
                self._embed_fn(query),
                self.namespace,
                pool,
                domain=domain,
            )
            out: list[tuple[MemoryItem, float]] = []
            for item_id, sim in hits:
                item = self._store.get(item_id)
                if (
                    item
                    and item.is_active
                    and item.namespace == self.namespace
                    and (domain is None or item.domain == domain)
                ):
                    out.append((item, sim))
            if out:
                return out

        items = self._active(domain=domain)
        return [
            (it, self._similarity_fn(query, it.content))
            for it in items
        ]

    def _index_upsert(self, item: MemoryItem) -> None:
        if self._vector_index is None or self._embed_fn is None:
            return
        if not item.is_active:
            return
        self._vector_index.upsert(
            item.id,
            item.namespace,
            item.domain,
            self._embed_fn(item.content),
            event_id=item.event_id,
        )

    def _index_delete(self, item_id: str) -> None:
        if self._vector_index is None:
            return
        self._vector_index.delete(item_id, self.namespace)

    def _active(self, domain: str | None = None, event_id: str | None = None) -> list[MemoryItem]:
        """Active memories scoped to this layer's namespace."""
        return self._store.all_active(
            namespace=self.namespace, domain=domain, event_id=event_id)

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

    def _linking_domains(self, domain: str) -> frozenset[str]:
        return DOMAIN_SIBLINGS.get(domain, frozenset({domain}))

    def _slot_relate_threshold(self, domain: str) -> float:
        """Volatility-scaled in-domain link bar (lower than global relate)."""
        V_d = DOMAIN_VOLATILITY.get(domain, 0.5)
        return max(SLOT_LINK_FLOOR, self.relate_threshold - 0.12 - 0.10 * V_d)

    def _best_match_in_slot(
        self, content: str, domain: str
    ) -> tuple[Optional[MemoryItem], float]:
        """Link within a domain slot when global relate misses a paraphrase.

        Volatile singleton slots (mood, location) accept weaker overlap; domains
        with several coexisting facts require a clearer best match.
        """
        group = self._linking_domains(domain)
        items = [it for it in self._active() if it.domain in group]
        if not items:
            return None, 0.0

        candidate = self._select_candidate(content, items)
        sim = self._similarity_fn(content, candidate.content)
        threshold = self._slot_relate_threshold(domain)

        if len(items) > 1 and domain not in _domains.SLOT_DOMAINS:
            threshold = max(threshold, self.relate_threshold - 0.05)
            ranked = sorted(
                (self._similarity_fn(content, it.content) for it in items),
                reverse=True,
            )
            if len(ranked) >= 2 and ranked[0] - ranked[1] < 0.08:
                return None, sim

        if sim >= threshold:
            return candidate, sim

        if domain in _domains.SLOT_DOMAINS and len(items) == 1 and sim >= SLOT_LINK_FLOOR:
            return candidate, sim

        return None, sim

    def _verified_match(
        self, content: str
    ) -> tuple[Optional[MemoryItem], float, str]:
        """Two-stage link: recall cheaply and widely, then let the verifier judge.

        Stage 1 takes the top-k candidates above ``link_recall_bar`` — a bar low
        enough that recall is near-perfect (27/28 on held-out must-link pairs at
        0.20 with embeddings), which is only safe because stage 2 has the final
        say. Stage 2 asks the verifier in descending similarity order and takes
        the first agreement, so the most likely candidate is charged for first.

        Returns ``(item, similarity, kind)`` where kind is:

        * ``hit`` — verifier said UPDATE
        * ``keep_both`` — at least one live verdict was KEEP_BOTH (do not
          fall through to the ladder; that is how false merges happen)
        * ``no_recall`` — nothing cleared the recall bar
        * ``infra_fail`` — every ask errored; fall through to the ladder
        """
        scored = sorted(
            ((self._similarity_fn(content, it.content), it)
             for it in self._active()),
            key=lambda pair: pair[0],
            reverse=True,
        )
        if not scored:
            return None, 0.0, "no_recall"
        asked = 0
        failed = 0
        for sim, item in scored[:self.link_recall_top_k]:
            if sim < self.link_recall_bar:
                break
            asked += 1
            before = int(getattr(self._link_verifier, "failures", 0) or 0)
            if self._link_verifier.verify(content, item.content, item.domain):
                return item, sim, "hit"
            after = int(getattr(self._link_verifier, "failures", 0) or 0)
            if after > before:
                failed += 1
        if asked == 0:
            return None, scored[0][0], "no_recall"
        if failed == asked:
            return None, scored[0][0], "infra_fail"
        return None, scored[0][0], "keep_both"

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

    def _stamp_facts(self, item: MemoryItem, text: str | None = None) -> None:
        facts = self._heuristic_extractor.extract(
            text if text is not None else item.content)
        item.facts = facts_to_dicts(facts)

    def _facts_for(self, item: MemoryItem) -> list:
        """Persisted cards, with a one-shot backfill for pre-SAV rows."""
        if item.facts:
            return facts_from_dicts(item.facts)
        got = self._heuristic_extractor.extract(item.content)
        if got:
            item.facts = facts_to_dicts(got)
            self._store.update(item)
        return got

    def _heuristic_match(
        self, content: str
    ) -> tuple[Optional[MemoryItem], float, str]:
        """Join known-frame cards among same-subject items.

        Cosine is not the recall gate here — subject overlap is (the Graphiti
        cut). Attribute match still happens inside ``join_structured``.

        ``hit`` — conservative join said UPDATE.
        ``keep_both`` — the new statement had cards and no candidate joined.
        ``no_cards`` — unrecognised frame; caller falls through.
        """
        new_facts = self._heuristic_extractor.extract(content)
        if not new_facts:
            return None, 0.0, "no_cards"
        new_subjects = {normalize_subject(f.subject) for f in new_facts}
        new_pairs = {
            (normalize_subject(f.subject), normalize_attribute(f.attribute))
            for f in new_facts
        }
        candidates: list[tuple[MemoryItem, list, int]] = []
        for item in self._active():
            stored_facts = self._facts_for(item)
            if not stored_facts:
                continue
            stored_subjects = {
                normalize_subject(f.subject) for f in stored_facts
            }
            if not (stored_subjects & new_subjects):
                continue
            shared_attr = any(
                (normalize_subject(f.subject),
                 normalize_attribute(f.attribute)) in new_pairs
                for f in stored_facts
            )
            candidates.append((item, stored_facts, 0 if shared_attr else 1))
        candidates.sort(
            key=lambda row: (row[2], -row[0].last_confirmed_at))
        for item, stored_facts, _rank in candidates:
            if join_structured(stored_facts, new_facts, content, item.content):
                sim = self._similarity_fn(content, item.content)
                return item, sim, "hit"
        return None, 0.0, "keep_both"

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
        if self._vector_index is not None:
            self._vector_index.close()
        self._store.close()

    def clear(self) -> None:
        """Delete all memories for this layer's namespace."""
        self._store.delete_namespace(self.namespace)
        if self._vector_index is not None:
            self._vector_index.delete_namespace(self.namespace)
        if self._tracker is not None:
            self._tracker.clear_namespace(self.namespace)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
