"""
Classification eval helpers — labeled corpus for Problem 1 / SCHEDULE 1.4.

Load ``tests/fixtures/classification_corpus.json`` (or any same-shaped file),
score a classifier, and summarize collisions without external deps.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

# Default fixture path relative to repo root
DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "classification_corpus.json"
)

ClassifierFn = Callable[[str], str]


@dataclass(frozen=True)
class CorpusItem:
    text: str
    domain: str
    tags: tuple[str, ...] = ()
    note: str | None = None

    @property
    def is_collision(self) -> bool:
        return any(t.startswith("collision:") for t in self.tags)


@dataclass
class EvalResult:
    n: int
    correct: int
    by_domain: dict[str, dict[str, int]] = field(default_factory=dict)
    confusion: dict[str, Counter] = field(default_factory=dict)
    collisions: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    def domain_accuracy(self, domain: str) -> float:
        s = self.by_domain.get(domain) or {}
        n = s.get("n", 0)
        return (s.get("correct", 0) / n) if n else 0.0


def load_corpus(path: Path | str | None = None) -> list[CorpusItem]:
    """Load labeled utterances. Raises if schema is invalid."""
    p = Path(path) if path is not None else DEFAULT_CORPUS_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    items_raw = raw.get("items")
    if not isinstance(items_raw, list) or not items_raw:
        raise ValueError(f"corpus {p} missing non-empty 'items' list")
    out: list[CorpusItem] = []
    for i, row in enumerate(items_raw):
        if not isinstance(row, Mapping):
            raise ValueError(f"item {i} is not an object")
        text = row.get("text")
        domain = row.get("domain")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"item {i} missing text")
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError(f"item {i} missing domain")
        tags = row.get("tags") or []
        if not isinstance(tags, Sequence) or isinstance(tags, (str, bytes)):
            raise ValueError(f"item {i} tags must be a list")
        note = row.get("note")
        out.append(
            CorpusItem(
                text=text.strip(),
                domain=domain.strip(),
                tags=tuple(str(t) for t in tags),
                note=str(note) if note is not None else None,
            )
        )
    return out


def evaluate_classifier(
    classify: ClassifierFn,
    items: Iterable[CorpusItem] | None = None,
    *,
    path: Path | str | None = None,
) -> EvalResult:
    """Score ``classify(text) -> domain`` against the labeled corpus."""
    corpus = list(items) if items is not None else load_corpus(path)
    by_domain: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    confusion: dict[str, Counter] = defaultdict(Counter)
    collisions: list[dict] = []
    errors: list[dict] = []
    correct = 0

    for item in corpus:
        pred = classify(item.text)
        by_domain[item.domain]["n"] += 1
        confusion[item.domain][pred] += 1
        ok = pred == item.domain
        if ok:
            correct += 1
            by_domain[item.domain]["correct"] += 1
        else:
            err = {
                "text": item.text,
                "gold": item.domain,
                "pred": pred,
                "tags": list(item.tags),
            }
            errors.append(err)
            if item.is_collision:
                collisions.append(err)

    return EvalResult(
        n=len(corpus),
        correct=correct,
        by_domain=dict(by_domain),
        confusion={k: Counter(v) for k, v in confusion.items()},
        collisions=collisions,
        errors=errors,
    )


def format_report(result: EvalResult, *, top_errors: int = 15) -> str:
    """Human-readable baseline report."""
    lines = [
        f"Classification baseline: {result.correct}/{result.n} "
        f"= {result.accuracy:.1%}",
        "",
        "Per-domain accuracy:",
    ]
    for domain in sorted(result.by_domain):
        s = result.by_domain[domain]
        acc = s["correct"] / s["n"] if s["n"] else 0.0
        lines.append(f"  {domain:24s} {s['correct']:3d}/{s['n']:<3d}  {acc:5.1%}")
    lines.append("")
    lines.append(
        f"Collision-tagged misses: {len(result.collisions)} "
        f"(of {sum(1 for e in result.errors if e.get('tags'))} tagged errors)"
    )
    if result.errors:
        lines.append("")
        lines.append(f"Sample errors (up to {top_errors}):")
        for e in result.errors[:top_errors]:
            tag = ",".join(e["tags"]) if e["tags"] else "-"
            lines.append(
                f"  gold={e['gold']:20s} pred={e['pred']:20s} [{tag}] {e['text'][:70]}"
            )
    return "\n".join(lines)
