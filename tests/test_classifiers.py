"""Tests for pluggable classifiers, DomainRegistry, and classification corpus."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voltmem import (  # noqa: E402
    ChainedClassifier,
    DOMAIN_VOLATILITY,
    DomainRegistry,
    HeuristicClassifier,
    KeywordClassifier,
    create_memory,
)
from voltmem import domains as dom  # noqa: E402
from voltmem.classification_eval import (  # noqa: E402
    evaluate_classifier,
    load_corpus,
)

# Floors from experiments/classification_baseline.py (HeuristicClassifier ~84%).
# Regression guard — raise after intentional keyword improvements.
HEURISTIC_MIN_ACCURACY = 0.80
HEURISTIC_MIN_ACCURACY_EXCL_TRANSIENT = 0.88


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


# ── SCHEDULE 1.4 — classification eval corpus ─────────────────────────────────

def test_corpus_schema_and_coverage():
    items = load_corpus()
    assert len(items) >= 200
    domains = {i.domain for i in items}
    missing = set(DOMAIN_VOLATILITY) - domains
    assert not missing, f"corpus missing domains: {sorted(missing)}"
    assert any(i.is_collision for i in items), "expected collision:* tagged items"


def test_heuristic_baseline_accuracy_floor():
    clf = HeuristicClassifier()
    result = evaluate_classifier(clf.classify_domain)
    assert result.accuracy >= HEURISTIC_MIN_ACCURACY, (
        f"heuristic accuracy {result.accuracy:.1%} < floor {HEURISTIC_MIN_ACCURACY:.0%}"
    )
    # transient_fact has no keyword map — exclude from the “keyword domains” floor
    excl = [
        i for i in load_corpus() if i.domain != "transient_fact"
    ]
    excl_result = evaluate_classifier(clf.classify_domain, excl)
    assert excl_result.accuracy >= HEURISTIC_MIN_ACCURACY_EXCL_TRANSIENT, (
        f"excl-transient accuracy {excl_result.accuracy:.1%} "
        f"< floor {HEURISTIC_MIN_ACCURACY_EXCL_TRANSIENT:.0%}"
    )


def test_feel_collision_documented():
    """'feel' substring beats opinion cues — corpus documents the known failure."""
    clf = HeuristicClassifier()
    text = "I feel that remote work is better for deep focus"
    assert clf.classify_domain(text) == "emotional_context"
    gold = next(i for i in load_corpus() if i.text == text)
    assert gold.domain == "opinion"
    assert any(t.startswith("collision:feel") for t in gold.tags)


def test_feel_control_still_emotional():
    clf = HeuristicClassifier()
    assert clf.classify_domain("I feel happy about the release") == "emotional_context"


def test_corpus_eval_runs_under_five_seconds():
    clf = HeuristicClassifier()
    t0 = time.perf_counter()
    evaluate_classifier(clf.classify_domain)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"corpus eval took {elapsed:.2f}s"


if __name__ == "__main__":
    tests = [
        test_keyword_classifier_matches_custom_domain,
        test_chained_classifier_falls_through_to_heuristic,
        test_callable_classifier_dict,
        test_create_memory_classifier_string_llm_alias,
        test_domain_registry_restored_on_close,
        test_domain_registry_custom_domain_classified,
        test_legacy_extractor_kwarg_still_works,
        test_corpus_schema_and_coverage,
        test_heuristic_baseline_accuracy_floor,
        test_feel_collision_documented,
        test_feel_control_still_emotional,
        test_corpus_eval_runs_under_five_seconds,
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
