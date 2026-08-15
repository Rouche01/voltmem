"""
VoltMem — Volatility-Adjusted Persistent Memory Layer
======================================================

A pluggable memory layer for LLM applications and any system that needs
persistent context with principled staleness handling.

Core idea: not all memories age at the same rate. Stable knowledge
(personality traits, core preferences) should be protected strongly
against overwriting. Volatile knowledge (current tasks, emotional context,
location) should be held loosely and updated readily.

Quick start:
    from voltmem import MemoryLayer

    mem = MemoryLayer("my_app.db")

    mem.write("User prefers direct communication", domain="core_preference")
    mem.write("User is currently job hunting", domain="current_project")

    # New observation — may or may not update existing memory
    result = mem.observe(
        content="User mentioned they accepted a job offer",
        domain="current_project",
        mismatch_magnitude=0.8,
        source="explicit_statement",
    )
    print(result.action)  # "audited" — volatile domain, high mismatch → updated

    # Retrieve relevant memories for a query
    results = mem.retrieve("career and work context")
    for item, score in zip(results.items, results.scores):
        print(f"[{score:.2f}] {item.content}")
"""

from .memory import MemoryLayer, WriteResult, RetrieveResult
from .client import Memory, create_memory
from .domains import MemoryItem, DOMAIN_VOLATILITY, SOURCE_RELIABILITY, DomainRegistry
from .classifiers import (
    Classifier,
    HeuristicClassifier,
    LLMClassifier,
    KeywordClassifier,
    ChainedClassifier,
    CallableClassifier,
    resolve_classifier,
)
from .vector_index import (
    VectorIndex,
    BruteForceVectorIndex,
    SqliteVectorIndex,
    create_vector_index,
)
from .embeddings import EmbeddingSimilarity
from .extract import HeuristicExtractor, LLMExtractor, HeuristicFactExtractor, LLMFactExtractor
from .structure import (
    StructuredFact,
    join_structured,
    HeuristicStructuredExtractor,
    LLMStructuredExtractor,
    StructuredJoinVerifier,
)
from .verify import (
    LinkVerifier,
    LLMLinkVerifier,
    CrossEncoderVerifier,
    VERIFY_PROMPT,
    VERIFY_SYSTEM,
)
from .summarize import MemorySummarizer, HeuristicSummarizer, LLMSummarizer
from .discovery import DomainStats, VolatilityTracker, blend_volatility
from .maintenance import (
    MaintenanceWindow,
    MaintenanceContext,
    expire_cleanup,
    reclassify_ambiguous,
    pattern_audit,
    consolidate,
    reconcile_twins,
)
from .scoring import (
    escalation_score,
    retrieval_score,
    staleness,
    protection_weight,
    update_volatility_ema,
    update_surprise_ema,
    update_mismatch_expectation,
    residual_surprise,
    resolve_escalation_law,
    RESIDUAL_GATE,
    belief_has_shifted,
    belief_shift_mass,
    belief_shift_bar,
    BELIEF_SHIFT_K,
    expected_mismatch,
    mismatch_sigma,
    recent_surprise,
    surprise_mode_scale,
    similarity_spread,
    freshness_mix,
    VOL_DRIFT_MAX,
    SIM_SPREAD_FLAT,
    SIM_SPREAD_FULL,
    MIX_MIN,
    EXPLICIT_OVERRIDE_M,
    EXPLICIT_MIN_VD,
    EXPLICIT_MAX_VD,
    EXPLICIT_E_RATIO,
    CUMULATIVE_MISMATCH_ESCALATE,
    S_MIN,
    ESCALATION_MODES,
    SURPRISE_HALFLIFE_DAYS,
    V_EXP_HOMEOSTATIC,
    V_EXP_CURRENT,
    V_EXP_ALLOSTATIC,
)

__all__ = [
    "Memory",
    "create_memory",
    "MemoryLayer",
    "WriteResult",
    "RetrieveResult",
    "MemoryItem",
    "DOMAIN_VOLATILITY",
    "SOURCE_RELIABILITY",
    "DomainRegistry",
    "Classifier",
    "HeuristicClassifier",
    "LLMClassifier",
    "KeywordClassifier",
    "ChainedClassifier",
    "CallableClassifier",
    "resolve_classifier",
    "EmbeddingSimilarity",
    "VectorIndex",
    "BruteForceVectorIndex",
    "SqliteVectorIndex",
    "create_vector_index",
    "HeuristicExtractor",
    "LLMExtractor",
    "HeuristicFactExtractor",
    "LLMFactExtractor",
    "StructuredFact",
    "join_structured",
    "HeuristicStructuredExtractor",
    "LLMStructuredExtractor",
    "StructuredJoinVerifier",
    "LinkVerifier",
    "LLMLinkVerifier",
    "CrossEncoderVerifier",
    "VERIFY_PROMPT",
    "VERIFY_SYSTEM",
    "MemorySummarizer",
    "HeuristicSummarizer",
    "LLMSummarizer",
    "DomainStats",
    "VolatilityTracker",
    "blend_volatility",
    "escalation_score",
    "retrieval_score",
    "staleness",
    "protection_weight",
    "update_volatility_ema",
    "update_surprise_ema",
    "update_mismatch_expectation",
    "residual_surprise",
    "resolve_escalation_law",
    "RESIDUAL_GATE",
    "belief_has_shifted",
    "belief_shift_mass",
    "belief_shift_bar",
    "BELIEF_SHIFT_K",
    "expected_mismatch",
    "mismatch_sigma",
    "recent_surprise",
    "surprise_mode_scale",
    "similarity_spread",
    "freshness_mix",
    "VOL_DRIFT_MAX",
    "SIM_SPREAD_FLAT",
    "SIM_SPREAD_FULL",
    "MIX_MIN",
    "EXPLICIT_OVERRIDE_M",
    "EXPLICIT_MIN_VD",
    "EXPLICIT_MAX_VD",
    "EXPLICIT_E_RATIO",
    "CUMULATIVE_MISMATCH_ESCALATE",
    "S_MIN",
    "ESCALATION_MODES",
    "SURPRISE_HALFLIFE_DAYS",
    "V_EXP_HOMEOSTATIC",
    "V_EXP_CURRENT",
    "V_EXP_ALLOSTATIC",
    "MaintenanceWindow",
    "MaintenanceContext",
    "expire_cleanup",
    "reclassify_ambiguous",
    "pattern_audit",
    "consolidate",
    "reconcile_twins",
]

__version__ = "0.3.1"
