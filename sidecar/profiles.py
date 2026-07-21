"""Domain profiles for the sidecar (fashion / stylens defaults)."""

from __future__ import annotations

from voltmem import (
    ChainedClassifier,
    DomainRegistry,
    HeuristicClassifier,
    KeywordClassifier,
)
from voltmem.classifiers import Classifier


def stylens_domains() -> DomainRegistry:
    """Stable fit/color prefs vs volatile occasion — dogfood priors."""
    domains = DomainRegistry()
    domains.register("style_preference", 0.08)
    domains.register("style_constraint", 0.25)
    domains.register("session_occasion", 0.80, slot=True)
    return domains


def stylens_classifier() -> Classifier:
    return ChainedClassifier(
        [
            KeywordClassifier(
                {
                    "style_preference": [
                        "prefer",
                        "darker colors",
                        "minimal",
                        "loose fits",
                    ],
                    "style_constraint": [
                        "no wool",
                        "tight budget",
                        "must be formal",
                    ],
                    "session_occasion": [
                        "wedding",
                        "job interview",
                        "date night",
                    ],
                }
            ),
            HeuristicClassifier(),
        ]
    )


def build_profile(name: str = "stylens") -> tuple[DomainRegistry, Classifier]:
    if name != "stylens":
        raise ValueError(f"unknown profile {name!r}; only 'stylens' is supported")
    return stylens_domains(), stylens_classifier()
