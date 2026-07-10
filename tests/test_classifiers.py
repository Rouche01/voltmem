"""Tests for pluggable classifiers and DomainRegistry."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voltmem import (  # noqa: E402
    ChainedClassifier,
    DomainRegistry,
    HeuristicClassifier,
    KeywordClassifier,
    create_memory,
)
from voltmem import domains as dom  # noqa: E402


def test_keyword_classifier_matches_custom_domain():
    domains = DomainRegistry()
    domains.register("style_preference", 0.08)
    restore = domains.install()
    try:
        clf = KeywordClassifier({
            "style_preference": ["darker colors", "minimal style"],
        })
        assert clf.classify_domain("I prefer darker colors") == "style_preference"
        assert clf.match_domain("hello there") is None
    finally:
        restore()


def test_chained_classifier_falls_through_to_heuristic():
    domains = DomainRegistry()
    domains.register("style_constraint", 0.25)
    restore = domains.install()
    try:
        clf = ChainedClassifier([
            KeywordClassifier({"style_constraint": ["no wool", "tight budget"]}),
            HeuristicClassifier(),
        ])
        assert clf.classify_domain("no wool please") == "style_constraint"
        assert clf.classify_domain("I live in Berlin") == "location"
    finally:
        restore()


def test_callable_classifier_dict():
    with create_memory(
        ":memory:",
        user_id="u1",
        embeddings=False,
        classifier={
            "classify": lambda t: "custom_domain"
            if "stylens" in t.lower() else "stated_preference",
            "mismatch": lambda n, e, s: 0.05 if s > 0.8 else 0.7,
        },
        domains=DomainRegistry().register("custom_domain", 0.4),
    ) as mem:
        row = mem.add("stylens user prefers navy")
        assert row["domain"] == "custom_domain"


def test_create_memory_classifier_string_llm_alias():
    # Should not raise — builds LLM classifier (may fall back on classify)
    with create_memory(
        ":memory:",
        classifier="llm",
        embeddings=False,
    ) as mem:
        assert mem.layer._extractor is not None


def test_domain_registry_restored_on_close():
    before = dict(dom.DOMAIN_VOLATILITY)
    reg = DomainRegistry().register("ephemeral_domain", 0.33, slot=True)
    with create_memory(":memory:", domains=reg, embeddings=False) as mem:
        assert "ephemeral_domain" in dom.DOMAIN_VOLATILITY
        assert "ephemeral_domain" in dom.SLOT_DOMAINS
        mem.add("ephemeral fact", extract=False)
    assert dom.DOMAIN_VOLATILITY == before
    assert "ephemeral_domain" not in dom.SLOT_DOMAINS


def test_domain_registry_custom_domain_classified():
    domains = DomainRegistry()
    domains.register("style_preference", 0.08)
    classifier = KeywordClassifier({
        "style_preference": ["darker colors", "minimal style"],
    })
    with create_memory(
        ":memory:",
        domains=domains,
        classifier=classifier,
        embeddings=False,
    ) as mem:
        row = mem.add("I love minimal style and darker colors")
        assert row["domain"] == "style_preference"
        assert "style_preference" in domains.known_domains()


def test_legacy_extractor_kwarg_still_works():
    custom = HeuristicClassifier(relate_similarity=0.6)
    with create_memory(":memory:", extractor=custom, embeddings=False) as mem:
        assert mem.layer._extractor is custom


if __name__ == "__main__":
    tests = [
        test_keyword_classifier_matches_custom_domain,
        test_chained_classifier_falls_through_to_heuristic,
        test_callable_classifier_dict,
        test_create_memory_classifier_string_llm_alias,
        test_domain_registry_restored_on_close,
        test_domain_registry_custom_domain_classified,
        test_legacy_extractor_kwarg_still_works,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
