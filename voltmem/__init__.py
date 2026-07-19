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
from .discovery import DomainStats, VolatilityTracker, blend_volatility
from .scoring import (
    escalation_score,
    retrieval_score,
    staleness,
    protection_weight,
    update_volatility_ema,
    similarity_spread,
    freshness_mix,
    VOL_DRIFT_MAX,
    SIM_SPREAD_FLAT,
    SIM_SPREAD_FULL,
    MIX_MIN,
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
    "DomainStats",
    "VolatilityTracker",
    "blend_volatility",
    "escalation_score",
    "retrieval_score",
    "staleness",
    "protection_weight",
    "update_volatility_ema",
    "similarity_spread",
    "freshness_mix",
    "VOL_DRIFT_MAX",
    "SIM_SPREAD_FLAT",
    "SIM_SPREAD_FULL",
    "MIX_MIN",
]

__version__ = "0.2.0"
