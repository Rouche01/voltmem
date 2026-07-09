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
from .domains import MemoryItem, DOMAIN_VOLATILITY, SOURCE_RELIABILITY
from .embeddings import EmbeddingSimilarity
from .extract import HeuristicExtractor, LLMExtractor
from .scoring import (
    escalation_score,
    retrieval_score,
    staleness,
    protection_weight,
)

__all__ = [
    "MemoryLayer",
    "WriteResult",
    "RetrieveResult",
    "MemoryItem",
    "DOMAIN_VOLATILITY",
    "SOURCE_RELIABILITY",
    "EmbeddingSimilarity",
    "HeuristicExtractor",
    "LLMExtractor",
    "escalation_score",
    "retrieval_score",
    "staleness",
    "protection_weight",
]

__version__ = "0.1.0"
