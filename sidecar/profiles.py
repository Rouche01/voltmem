"""Domain profiles installed when the sidecar process starts.

``VOLTMEM_PROFILE`` selects the built-in registry (default ``stylens``).
``VOLTMEM_DOMAINS_FILE`` merges extra domains and keyword hints from JSON
so an app can define its own kinds without a code profile.
"""

from __future__ import annotations

import json
from pathlib import Path

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


_PROFILES = {
    "stylens": (stylens_domains, stylens_classifier),
}


def build_profile(name: str = "stylens") -> tuple[DomainRegistry, Classifier]:
    key = (name or "stylens").strip().lower()
    builders = _PROFILES.get(key)
    if builders is None:
        known = ", ".join(sorted(_PROFILES))
        raise ValueError(f"unknown profile {name!r}; expected one of: {known}")
    domains_fn, classifier_fn = builders
    return domains_fn(), classifier_fn()


def apply_domains_file(
    domains: DomainRegistry,
    classifier: Classifier,
    path: str,
) -> tuple[DomainRegistry, Classifier]:
    """Merge a JSON domain document into a profile registry.

    ``domains`` entries are registered on top of the profile. ``keywords``
    are checked before the profile classifier. Earlier keyword keys win.
    """
    file_path = Path(path)
    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"VOLTMEM_DOMAINS_FILE {path} could not be read: {exc}") from exc
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"VOLTMEM_DOMAINS_FILE {path} is not valid JSON: {exc}") from exc
    if not isinstance(spec, dict):
        raise ValueError(f"VOLTMEM_DOMAINS_FILE {path} must be a JSON object")

    entries = spec.get("domains", [])
    if not isinstance(entries, list):
        raise ValueError(f"VOLTMEM_DOMAINS_FILE {path} field 'domains' must be a list")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"VOLTMEM_DOMAINS_FILE {path} domains[{index}] must be an object"
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"VOLTMEM_DOMAINS_FILE {path} domains[{index}].name must be a string"
            )
        volatility = entry.get("volatility")
        if isinstance(volatility, bool) or not isinstance(volatility, (int, float)):
            raise ValueError(
                f"VOLTMEM_DOMAINS_FILE {path} domains[{index}].volatility must be a number"
            )
        slot = entry.get("slot", False)
        if not isinstance(slot, bool):
            raise ValueError(
                f"VOLTMEM_DOMAINS_FILE {path} domains[{index}].slot must be a boolean"
            )
        domains.register(name.strip(), float(volatility), slot=slot)

    keywords = spec.get("keywords", {})
    if not isinstance(keywords, dict):
        raise ValueError(f"VOLTMEM_DOMAINS_FILE {path} field 'keywords' must be an object")
    if not keywords:
        return domains, classifier

    known = domains.known_domains()
    cleaned: dict[str, list[str]] = {}
    for domain, phrases in keywords.items():
        if not isinstance(domain, str) or domain not in known:
            raise ValueError(
                f"VOLTMEM_DOMAINS_FILE {path} keyword domain {domain!r} is not registered"
            )
        if not isinstance(phrases, list) or not phrases:
            raise ValueError(
                f"VOLTMEM_DOMAINS_FILE {path} keywords[{domain!r}] must be a non-empty list"
            )
        texts: list[str] = []
        for phrase in phrases:
            if not isinstance(phrase, str) or not phrase.strip():
                raise ValueError(
                    f"VOLTMEM_DOMAINS_FILE {path} keywords[{domain!r}] must be strings"
                )
            texts.append(phrase.strip())
        cleaned[domain] = texts
    return domains, ChainedClassifier([KeywordClassifier(cleaned), classifier])
