"""
Structured extract-then-join — match on (subject, attribute), not cosine.
========================================================================

Write-time extract of ``(subject, attribute, value, cardinality)``. Matching is
a join. The conservative rule (default) prefers a duplicate over a wrong
overwrite:

    same subject + same attribute + specific slot     → UPDATE
    same subject + same attribute + multi + mark      → UPDATE
    generic slot (current_task / mood / project /
        manager) only with value overlap, anaphora,
        or a positive manager/employer fill           → UPDATE
    named ending of an entity that is not the stored
        value                                         → KEEP_BOTH
    otherwise                                         → KEEP_BOTH

Attribute is the *question* a fact answers, never the answer. Two skills are
two attributes (``skill_python`` vs ``skill_japanese``); a city correction is
one attribute (``residence_city``) with two values. That is the Graphiti
entity-pair constraint and HippoRAG's triples, without a graph.

Known-frame cards are persisted on ``MemoryItem.facts`` at write time.
Heuristic recall is constrained by subject, then joined on attribute.
Empty extract means "not this path" — insert, do not guess.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

EXTRACT_SYSTEM = (
    "You extract structured memory facts about one user. Attribute is the "
    "question a fact answers, never the answer. Reply with JSON only."
)

# DEV examples only (Berlin/Paris near-copy, Python/Japanese trap). Held-out
# pairs must not appear here or the 56-pair number stops meaning anything.
EXTRACT_PROMPT = """Statement: "{text}"

Extract every atomic fact this statement asserts.

For each fact:
  subject      who it is about. Use "user" for I / User / the user. Relatives
               and colleagues keep their own subject ("user_parents",
               "user_spouse", "user_father", the named person).
  attribute    the QUESTION, as a short snake_case slug. Never put the answer
               in the attribute.
               Single-valued questions (one value at a time) use a generic
               slot: residence_city, workplace_city, occupation, employer,
               birth_year, birthplace, native_language, current_mood,
               current_task, current_project, current_manager,
               marital_status, work_arrangement.
               Multi-valued collections INCLUDE WHICH MEMBER: skill_python,
               skill_japanese, child_mia, preference_dark_mode,
               appointment_dentist, appointment_flight. Two skills are two
               attributes. Two appointments are two attributes.
               A feeling about a specific topic is mood_<topic>, not
               current_mood.
  value        the answer, short
  cardinality  "slot" (at most one value) or "multi" (siblings can coexist)
  replaces     true if this statement cancels or replaces a previous value of
               THIS SAME attribute. A named-entity ending ("no longer works
               with Bob", "no longer studies French") is about THAT named
               entity, not about a generic slot. A slot replacement
               ("changed careers and now work as a nurse", "now reports to
               Priya", "moved to Paris") uses the generic slot attribute.

Examples of the shape, not a complete list:
  "I live in Berlin" → user / residence_city / Berlin / slot / false
  "I live in Paris" → user / residence_city / Paris / slot / false
  "User is proficient in Python" → user / skill_python / proficient / multi / false
  "User is proficient in Japanese" → user / skill_japanese / proficient / multi / false

Reply with JSON only:
{{"facts": [{{"subject": "...", "attribute": "...", "value": "...", "cardinality": "slot"|"multi", "replaces": true|false}}]}}"""


# Applied only after subject and attribute already match, so a marker on a
# different skill cannot merge Python with French.
_CHANGE_MARKERS = (
    "no longer", "not anymore", "used to", "changed", "switched", "instead",
    "moved", "finished", "quit", "left", "now", "actually", "these days",
    "as of", "since then", "updated", "gave up", "relocated", "promoted",
    "separated", "clarified", "wrapped",
)
_MARKER_RE = re.compile("|".join(re.escape(m) for m in _CHANGE_MARKERS))


def has_change_marker(text: str) -> bool:
    return bool(_MARKER_RE.search(text.lower()))


# Drawers that are too coarse to auto-join on slug match alone. Every false
# merge in the naive extract-then-join eval used one of these.
GENERIC_SLOT_ATTRIBUTES = frozenset({
    "current_task",
    "current_mood",
    "current_project",
    "current_manager",
})

_NAME_STOP = frozenset({
    "user", "the", "a", "an", "i", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "january", "february",
    "march", "april", "june", "july", "august", "september", "october",
    "november", "december",
})
_CONTENT_STOP = frozenset({
    "user", "the", "a", "an", "is", "in", "to", "of", "and", "or", "for",
    "on", "at", "as", "their", "they", "them", "this", "that", "with",
    "from", "was", "are", "be", "been", "being", "it", "its", "my", "me",
    "i", "we", "you", "has", "have", "had", "do", "does", "did", "not",
    "no", "now", "new", "old", "very", "much", "more", "than", "then",
})
_ENDING_RE = re.compile(
    r"\b(?:no longer|not anymore|gave up(?: on)?|quit|stopped|"
    r"changed their (?:mind|view)(?: about| on)?)\b",
    re.I,
)
_LEFT_NAME_RE = re.compile(r"\bleft\s+([A-Z][\w'-]+)")
_PROPER_NAME_RE = re.compile(r"\b([A-Z][\w'-]+)\b")
_POSITIVE_FILL_RE = re.compile(
    r"\b(?:now|joined|became|moved to|relocated|promoted)\b", re.I)
_ANAPHORA_RE = re.compile(
    r"\b(?:wrapped|finished|moved on)\b.*\b(?:that|it|those)\b|"
    r"\b(?:that|it|those)\b.*\b(?:wrapped|finished|moved on)\b",
    re.I,
)


def _content_tokens(*parts: str) -> set[str]:
    got: set[str] = set()
    for part in parts:
        for tok in re.findall(r"[a-z0-9]+", (part or "").lower()):
            if len(tok) >= 4 and tok not in _CONTENT_STOP:
                got.add(tok)
    return got


def _proper_names(text: str) -> list[str]:
    return [
        m.group(1) for m in _PROPER_NAME_RE.finditer(text or "")
        if m.group(1).lower() not in _NAME_STOP
    ]


def ended_names(new_text: str) -> list[str]:
    """Named entities the new statement is *ending*, not substituting."""
    if not new_text:
        return []
    names: list[str] = []
    for m in _ENDING_RE.finditer(new_text):
        names.extend(_proper_names(new_text[m.end():]))
    names.extend(_LEFT_NAME_RE.findall(new_text))
    # de-dupe, keep order
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def named_ending_blocks(stored_text: str, new_text: str, stored: StructuredFact | None = None) -> bool:
    """True when the new sentence ends a name that is not the stored fact."""
    names = ended_names(new_text)
    if not names:
        return False
    hay = " ".join(filter(None, [
        stored_text,
        stored.value if stored is not None else "",
        stored.attribute if stored is not None else "",
    ])).lower()
    return not any(n.lower() in hay for n in names)


def values_overlap(stored: StructuredFact, new: StructuredFact,
                   stored_text: str = "", new_text: str = "") -> bool:
    """Content-word overlap between values (and value ↔ the other sentence)."""
    s_val = _content_tokens(stored.value)
    n_val = _content_tokens(new.value)
    if s_val and n_val and s_val & n_val:
        return True
    if s_val and s_val & _content_tokens(new_text):
        return True
    if n_val and n_val & _content_tokens(stored_text):
        return True
    return False


def is_positive_slot_fill(new_text: str) -> bool:
    """Asserts a new slot holder (now / joined / became), not an ending."""
    if not new_text or _ENDING_RE.search(new_text):
        return False
    return bool(_POSITIVE_FILL_RE.search(new_text))


def has_anaphora_marker(new_text: str) -> bool:
    return bool(_ANAPHORA_RE.search(new_text or ""))


@dataclass(frozen=True)
class StructuredFact:
    subject: str
    attribute: str
    value: str
    cardinality: str = "multi"
    replaces: bool = False

    @property
    def is_slot(self) -> bool:
        return self.cardinality == "slot"


def normalize_subject(raw: str) -> str:
    t = (raw or "").strip().lower()
    t = re.sub(r"^(the|a|an)\s+", "", t)
    t = t.replace("the user's", "user's").replace("user's ", "user_")
    t = t.replace("user's", "user_")
    t = re.sub(r"[^a-z0-9_]+", "_", t).strip("_")
    aliases = {
        "i": "user", "me": "user", "myself": "user",
        "they": "user", "them": "user",
        "the_user": "user",
    }
    return aliases.get(t, t or "user")


def normalize_attribute(raw: str) -> str:
    t = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    t = re.sub(r"_+", "_", t)
    t = re.sub(r"[^a-z0-9_]+", "", t)
    return t.strip("_")


def _norm_cardinality(raw) -> str:
    t = str(raw or "").strip().lower()
    if t in ("slot", "single", "single-valued", "single_valued", "one"):
        return "slot"
    return "multi"


def _fact_from_row(row: dict) -> StructuredFact | None:
    if not isinstance(row, dict):
        return None
    attr = str(row.get("attribute", "")).strip()
    if not attr:
        return None
    replaces = row.get("replaces", False)
    if isinstance(replaces, str):
        replaces = replaces.strip().lower() in ("true", "yes", "1")
    return StructuredFact(
        subject=str(row.get("subject", "user")),
        attribute=attr,
        value=str(row.get("value", "")).strip(),
        cardinality=_norm_cardinality(row.get("cardinality")),
        replaces=bool(replaces),
    )


def fact_to_dict(fact: StructuredFact) -> dict:
    return {
        "subject": fact.subject,
        "attribute": fact.attribute,
        "value": fact.value,
        "cardinality": fact.cardinality,
        "replaces": fact.replaces,
    }


def facts_to_dicts(facts: list[StructuredFact]) -> list[dict]:
    return [fact_to_dict(f) for f in facts]


def facts_from_dicts(rows) -> list[StructuredFact]:
    if not rows:
        return []
    out: list[StructuredFact] = []
    for row in rows:
        fact = _fact_from_row(row)
        if fact is not None:
            out.append(fact)
    return out


def parse_structured(raw: str) -> list[StructuredFact]:
    """Read extractor JSON. Unreadable output is an empty list (keep both)."""
    if not raw:
        return []
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        start, end = raw.find("["), raw.rfind("]")
        if start < 0 or end <= start:
            return []
        try:
            rows = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return []
        obj = {"facts": rows} if isinstance(rows, list) else {}
    else:
        try:
            obj = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return []
    rows = obj.get("facts", obj if isinstance(obj, list) else [obj])
    if isinstance(obj, dict) and "attribute" in obj and "facts" not in obj:
        rows = [obj]
    if not isinstance(rows, list):
        return []
    return facts_from_dicts(rows)


def _generic_slot_may_join(
    stored: StructuredFact,
    new: StructuredFact,
    stored_text: str,
    new_text: str,
) -> bool:
    if values_overlap(stored, new, stored_text, new_text):
        return True
    if has_anaphora_marker(new_text):
        return True
    attr = normalize_attribute(new.attribute)
    if attr in ("current_manager", "employer") and is_positive_slot_fill(new_text):
        return not named_ending_blocks(stored_text, new_text, stored)
    return False


def join_structured(
    stored: list[StructuredFact],
    new: list[StructuredFact],
    new_text: str = "",
    stored_text: str = "",
    *,
    conservative: bool = True,
) -> bool:
    """True = same fact (UPDATE). Empty extract never merges.

    Conservative (default) will not auto-join a generic slot on slug match
    alone, and will not join when the new sentence ends a name that is not
    the stored value. Naive mode is the first extract-then-join eval.
    """
    if not stored or not new:
        return False
    marker = has_change_marker(new_text) if new_text else False
    for s in stored:
        s_subj = normalize_subject(s.subject)
        s_attr = normalize_attribute(s.attribute)
        if not s_attr:
            continue
        for n in new:
            if s_subj != normalize_subject(n.subject):
                continue
            if s_attr != normalize_attribute(n.attribute):
                continue
            if conservative and named_ending_blocks(stored_text, new_text, s):
                continue
            generic = s_attr in GENERIC_SLOT_ATTRIBUTES
            if conservative and generic:
                if _generic_slot_may_join(s, n, stored_text, new_text):
                    return True
                continue
            if s.is_slot or n.is_slot:
                return True
            if n.replaces or s.replaces or marker:
                return True
    return False


# High-precision frames only. Incomplete on purpose: a miss is a duplicate,
# and sleeptime reconcile_twins is allowed to finish the job. Do not add
# current_task / current_mood / current_manager — those were the false merges.
_SUBJECT_RE = (
    (re.compile(r"\b(?:user's|my)\s+parents\b", re.I), "user_parents"),
    (re.compile(r"\b(?:user's|my)\s+spouse\b", re.I), "user_spouse"),
    (re.compile(r"\b(?:user's|my)\s+father\b", re.I), "user_father"),
    (re.compile(r"\b(?:user's|my)\s+mother\b", re.I), "user_mother"),
)
_CITY_RE = re.compile(
    r"\b(?:live|lives|living|reside|resides|based)\s+in\s+([A-Z][a-zA-Z]+)"
    r"|\b(?:moved|relocated)\s+to\s+([A-Z][a-zA-Z]+)",
)
_BIRTH_YEAR_RE = re.compile(r"\bborn\s+in\s+(19\d{2}|20\d{2})\b", re.I)
_SKILL_RE = re.compile(
    r"\b(?:proficient|fluent)\s+in\s+([A-Za-z][A-Za-z+#]*)"
    r"|\blearning\s+(?:to\s+)?([A-Za-z][A-Za-z+#]*)"
    r"|\bno longer\s+(?:uses|use|studies|study)\s+([A-Za-z][A-Za-z+#]*)",
    re.I,
)
_OCCUPATION_RE = re.compile(
    r"\b(?:work|works|working|worked)\s+as\s+(?:an?\s+)?(.+?)(?:[.!?]|$)",
    re.I,
)
_EMPLOYER_RE = re.compile(
    r"\b(?:work|works|working)\s+at\s+([A-Z][A-Za-z0-9]+)"
    r"|\bjoined\s+([A-Z][A-Za-z0-9]+)"
    r"|\bleft\s+(?:their\s+job\s+at\s+)?([A-Z][A-Za-z0-9]+)",
)


def _heuristic_subject(text: str) -> str:
    for pat, subj in _SUBJECT_RE:
        if pat.search(text):
            return subj
    return "user"


def _skill_slug(raw: str) -> str:
    tok = re.sub(r"[^a-z0-9+]+", "", raw.lower())
    return f"skill_{tok}" if tok else ""


def _occupation_value(raw: str) -> str:
    t = raw.strip().rstrip(".,;:")
    t = re.sub(r"^(?:an?\s+)", "", t, flags=re.I)
    t = re.split(r"\band\b", t, maxsplit=1)[0].strip()
    return t


class HeuristicStructuredExtractor:
    """Millisecond SAV extract for a few known frames. Empty = no card."""

    def __init__(self) -> None:
        self._mem: dict[str, list[StructuredFact]] = {}

    def extract(self, text: str) -> list[StructuredFact]:
        if text in self._mem:
            return self._mem[text]
        facts = self._extract(text)
        self._mem[text] = facts
        return facts

    def _extract(self, text: str) -> list[StructuredFact]:
        if not text or not text.strip():
            return []
        subj = _heuristic_subject(text)
        replacing = has_change_marker(text)
        out: list[StructuredFact] = []

        for m in _CITY_RE.finditer(text):
            city = next((g for g in m.groups() if g), None)
            if city:
                out.append(StructuredFact(
                    subj, "residence_city", city, "slot", replacing))

        for m in _BIRTH_YEAR_RE.finditer(text):
            out.append(StructuredFact(
                subj, "birth_year", m.group(1), "slot", replacing))

        for m in _SKILL_RE.finditer(text):
            name = next((g for g in m.groups() if g), None)
            slug = _skill_slug(name or "")
            if slug:
                ended = "no longer" in m.group(0).lower()
                out.append(StructuredFact(
                    subj, slug, "ended" if ended else "proficient",
                    "multi", replacing or ended))

        for m in _OCCUPATION_RE.finditer(text):
            role = _occupation_value(m.group(1))
            if role:
                out.append(StructuredFact(
                    subj, "occupation", role, "slot", replacing))

        for m in _EMPLOYER_RE.finditer(text):
            firm = next((g for g in m.groups() if g), None)
            if firm:
                left = m.group(0).lower().startswith("left")
                out.append(StructuredFact(
                    subj, "employer", firm, "slot", replacing or left))

        return out


class LLMStructuredExtractor:
    """One Ollama call per statement. Empty list on any failure (keep both)."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:14b",
        ollama_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        disk_cache: dict | None = None,
        persist=None,
    ) -> None:
        self.model = model
        self.url = ollama_url.rstrip("/") + "/api/generate"
        self.timeout = timeout
        self._mem: dict[str, list[StructuredFact]] = {}
        self._disk = disk_cache
        self._persist = persist
        self.calls = 0
        self.failures = 0

    def _generate(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "system": EXTRACT_SYSTEM,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 400},
        }).encode()
        req = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode()).get("response", "")

    def extract(self, text: str) -> list[StructuredFact]:
        if text in self._mem:
            return self._mem[text]
        if self._disk is not None and text in self._disk:
            facts = parse_structured(self._disk[text])
            self._mem[text] = facts
            return facts
        facts: list[StructuredFact] = []
        raw = ""
        try:
            self.calls += 1
            raw = self._generate(EXTRACT_PROMPT.format(text=text))
            facts = parse_structured(raw)
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            facts = []
        if not facts:
            self.failures += 1
        if self._disk is not None and raw:
            self._disk[text] = raw
            if self._persist is not None:
                self._persist()
        self._mem[text] = facts
        return facts


class StructuredJoinVerifier:
    """Stage-2 join over independently extracted facts.

    Simulates persist-at-write: each unique string is extracted once, then
    matching is ``join_structured``. Not attached to ``remember()`` auto.
    """

    def __init__(
        self,
        extractor: LLMStructuredExtractor | None = None,
        *,
        conservative: bool = True,
    ) -> None:
        self.extractor = extractor or LLMStructuredExtractor()
        self.conservative = conservative

    def verify(self, new_text: str, stored_text: str, domain: str) -> bool:
        stored = self.extractor.extract(stored_text)
        new = self.extractor.extract(new_text)
        return join_structured(
            stored, new, new_text, stored_text,
            conservative=self.conservative,
        )
