"""
Battery H — does an LLM verifier clear the bar the cheap one failed?
===================================================================

Where this sits. Battery G showed no single similarity cutoff can separate
must-link from must-not-link pairs. The recall-then-verify prototype then showed
the two-stage design is sound — embedding recall at a 0.20 bar keeps 27 of 28
held-out must-link pairs alive, so a perfect verifier reaches 55/56 — but that
its dependency-free verifier (domain cardinality OR change-marker language) does
NOT work: 19/24 on the pairs it was fitted to, 35/56 on held-out pairs, against
a single-threshold ceiling of 49/56. It lost to the thresholds it was meant to
replace.

So the second stage needs a signal that reads subject, attribute and value. The
held-out cell breakdown named exactly two shapes the lexical signals cannot see:

  attribute cardinality  "was born in 1990" -> "born in 1991" must link, while
                         a dentist appointment and a flight must coexist. The
                         DOMAIN's cardinality is the wrong question; the
                         ATTRIBUTE's cardinality is the right one.
  subject identity       "no longer reports to Miguel" against a stored fact
                         about Dana carries perfect replacement language and is
                         still a different fact.

The prompt below asks for precisely those two judgements and derives the
decision from them. It was written from those two category names — which are
properties of the grid, not of any particular pair — and deliberately NOT
iterated against held-out failures. Any prompt tuning must happen on the dev
split only, or the held-out number stops meaning anything, which is the exact
mistake this battery exists to avoid repeating.

Bars to beat, held-out, 56 pairs (all at recall bar 0.20, embeddings):

    31/56   stage 1 alone, no verification
    35/56   cheap verifier (cardinality + change marker)
    41/56   shipped ladder with embeddings, no second stage
    49/56   best possible single similarity threshold  <-- the bar that matters
    55/56   perfect verifier (ceiling imposed by stage-1 recall)

Every model response is cached on disk keyed by (model, prompt), so a re-run
costs nothing, is deterministic, and the raw answers stay auditable.

Run:
    VERIFY_BACKEND=openai  .venv/bin/python experiments/llm_verify_eval.py
    VERIFY_BACKEND=ollama  .venv/bin/python experiments/llm_verify_eval.py
"""

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voltmem.embeddings import EmbeddingSimilarity             # noqa: E402
from voltmem.verify import VERIFY_PROMPT, VERIFY_SYSTEM        # noqa: E402
from linking_pairs import SPLITS                               # noqa: E402
from link_verify_prototype import (                            # noqa: E402
    BAR,
    cell,
    verify_cardinality_or_marker,
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:14b"

SYSTEM = VERIFY_SYSTEM

# Two prompt versions, both reported, because the iteration between them is
# itself a result worth keeping rather than hiding.
#
# v1 was written from the two failure-shape names alone, with no examples. It
# fails in one direction almost totally: the model reads "same attribute" as
# "same value", so "based in Toronto" -> "based in Vancouver" comes back
# same_attribute=False and every genuine change is kept as a second fact.
#
# v2 fixes exactly that confusion and nothing else. Its two examples are both
# DEV pairs (the Berlin/Paris near-copy and the Python/Japanese trap), so the
# held-out split stays clean.
PROMPT_V1 = """Stored memory: "{stored}"
New statement: "{new}"

Answer two questions about them.

1. same_subject — do both statements describe the SAME subject? The user
   themselves and the user's parents, spouse, or colleague are different
   subjects.
2. same_attribute — do both describe the SAME attribute of that subject? Their
   city and their workplace city are different attributes. Two of their skills
   are different attributes. Their birth year stated twice is one attribute.

Then decide:
  UPDATE     — same subject AND same attribute, so the new statement restates,
               corrects, or replaces the stored one. Only one should remain.
  KEEP_BOTH  — anything else. Both are true at once and both must be kept.

Wording is not evidence. Near-identical phrasing can describe two different
attributes, and phrases like "no longer", "changed" or "now" can appear in a
statement about an entirely different subject or attribute.

Reply with JSON only, no prose:
{{"same_subject": true|false, "same_attribute": true|false, "decision": "UPDATE"|"KEEP_BOTH"}}"""

# The shipped prompt itself, so the measured artifact and the deployed one
# cannot drift apart. Editing it in voltmem/verify.py invalidates this cache and
# re-runs the battery, which is the intended coupling.
PROMPT_V2 = VERIFY_PROMPT

PROMPTS = [("v1 principled, no examples", PROMPT_V1),
           ("v2 attribute vs value", PROMPT_V2)]


# ── model access ──────────────────────────────────────────────────────────────

def load_dotenv(path=".env"):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full = os.path.join(root, path)
    if not os.path.exists(full):
        return
    with open(full) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


class Cache:
    """(model, prompt) -> raw response text, persisted as JSON."""

    def __init__(self, name):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.path = os.path.join(CACHE_DIR, f"llm_verify_{name}.json")
        self.data = {}
        if os.path.exists(self.path):
            with open(self.path) as fh:
                self.data = json.load(fh)
        self.hits = self.misses = 0

    @staticmethod
    def key(model, prompt):
        return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()[:32]

    def get(self, model, prompt):
        v = self.data.get(self.key(model, prompt))
        self.hits += v is not None
        return v

    def put(self, model, prompt, value):
        self.misses += 1
        self.data[self.key(model, prompt)] = value
        with open(self.path, "w") as fh:
            json.dump(self.data, fh, indent=1, sort_keys=True)


def _post(url, payload, headers, timeout=60):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class OpenAIBackend:
    def __init__(self, model=None):
        self.model = model or os.environ.get("VERIFY_MODEL", DEFAULT_OPENAI_MODEL)
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY not set")

    def generate(self, prompt):
        out = _post(
            "https://api.openai.com/v1/chat/completions",
            {"model": self.model, "temperature": 0,
             "messages": [{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}]},
            {"Authorization": f"Bearer {self.key}"})
        return out["choices"][0]["message"]["content"]


class OllamaBackend:
    """Local model. ``format=json`` and a token cap matter here in a way they do
    not for a hosted model: small models otherwise wrap the object in prose or
    keep going after it, which the parser scores as a refusal to link."""

    def __init__(self, model=None, url="http://localhost:11434"):
        self.model = model or os.environ.get("VERIFY_MODEL", DEFAULT_OLLAMA_MODEL)
        self.url = url.rstrip("/") + "/api/generate"

    def generate(self, prompt):
        out = _post(self.url, {
            "model": self.model, "stream": False, "format": "json",
            "options": {"temperature": 0.0, "num_predict": 120},
            "system": SYSTEM, "prompt": prompt,
        }, {}, timeout=180)
        return out.get("response", "")


def build_backend(model=None):
    name = os.environ.get("VERIFY_BACKEND", "openai").lower()
    return OllamaBackend(model) if name == "ollama" else OpenAIBackend(model)


def requested_models():
    """VERIFY_MODELS=a,b compares several models in one run; empty = the default."""
    raw = os.environ.get("VERIFY_MODELS") or os.environ.get("VERIFY_MODEL") or ""
    return [m.strip() for m in raw.split(",") if m.strip()] or [None]


def requested_prompts():
    """VERIFY_PROMPTS=v2 skips a version. Each one costs a full pass of calls,
    which on a local model is minutes rather than seconds."""
    want = [w.strip().lower() for w in
            os.environ.get("VERIFY_PROMPTS", "").split(",") if w.strip()]
    if not want:
        return PROMPTS
    keep = [(label, tpl) for label, tpl in PROMPTS
            if any(label.lower().startswith(w) for w in want)]
    if not keep:
        raise SystemExit(f"VERIFY_PROMPTS={want} matched none of "
                         f"{[label for label, _ in PROMPTS]}")
    return keep


# ── the verifier ──────────────────────────────────────────────────────────────

def parse(raw):
    """Returns (decision, same_subject, same_attribute) or None if unparseable."""
    if not raw:
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    dec = str(obj.get("decision", "")).strip().upper()
    if dec not in ("UPDATE", "KEEP_BOTH"):
        return None
    return dec, bool(obj.get("same_subject")), bool(obj.get("same_attribute"))


class LLMVerifier:
    def __init__(self, backend, cache, template):
        self.backend = backend
        self.cache = cache
        self.template = template
        self.unparseable = 0
        self.errors = 0
        self.live_calls = 0
        self.live_seconds = 0.0

    def __call__(self, new, stored, domain, sim):
        prompt = self.template.format(stored=stored, new=new)
        raw = self.cache.get(self.backend.model, prompt)
        if raw is None:
            started = time.monotonic()
            try:
                raw = self.backend.generate(prompt)
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
                    OSError) as exc:
                self.errors += 1
                print(f"    [error] {type(exc).__name__}: {exc}")
                return False
            self.live_calls += 1
            self.live_seconds += time.monotonic() - started
            self.cache.put(self.backend.model, prompt, raw)
        parsed = parse(raw)
        if parsed is None:
            self.unparseable += 1
            return False          # unparseable = refuse to link = keep both
        return parsed[0] == "UPDATE"

    def detail(self, new, stored):
        raw = self.cache.get(self.backend.model,
                             self.template.format(stored=stored, new=new))
        return parse(raw)


# ── scoring ───────────────────────────────────────────────────────────────────

def score(sim_fn, verifier, pairs, bar=BAR):
    """(correct, total, recall_misses, link_ok, n_pos, coexist_ok, n_neg)."""
    must_link, must_not_link = pairs
    link_ok = coexist_ok = recall_miss = 0
    for (_sd, st, nd, nt, _note) in must_link:
        s = sim_fn(nt, st)
        if s < bar:
            recall_miss += 1
            continue
        link_ok += bool(verifier(nt, st, nd, s))
    for (_sd, st, nd, nt, _note) in must_not_link:
        s = sim_fn(nt, st)
        if s < bar or not verifier(nt, st, nd, s):
            coexist_ok += 1
    return (link_ok + coexist_ok, len(must_link) + len(must_not_link),
            recall_miss, link_ok, len(must_link), coexist_ok, len(must_not_link))


def best_single_threshold(sim_fn, pairs):
    """Fewest errors any single similarity cutoff can achieve on these pairs."""
    must_link, must_not_link = pairs
    pos = [sim_fn(p[3], p[1]) for p in must_link]
    neg = [sim_fn(p[3], p[1]) for p in must_not_link]
    best = len(pos) + len(neg)
    for t in sorted(set(pos) | set(neg) | {0.0, 1.01}):
        errs = sum(1 for s in pos if s < t) + sum(1 for s in neg if s >= t)
        best = min(best, errs)
    return len(pos) + len(neg) - best


def main():
    load_dotenv()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sim_fn = EmbeddingSimilarity(backend="sentence-transformers")

    models = requested_models()
    prompts = requested_prompts()
    runs = []            # (model, prompt_label, verifier, [score per split])
    for m in models:
        backend = build_backend(m)
        cache = Cache(backend.model.replace(":", "-").replace("/", "-"))
        for plabel, tpl in prompts:
            runs.append([backend, plabel, LLMVerifier(backend, cache, tpl), [], cache])
    llm = runs[-1][2]    # last model, newest prompt, drives the detail sections

    print("=" * 94)
    print("BATTERY H — LLM VERIFIER as the precision stage")
    print("=" * 94)
    print(f"  backend={type(runs[0][0]).__name__}  recall bar={BAR:.2f}")
    for backend, _p, _v, _s, cache in runs[::len(prompts)]:
        print(f"    {backend.model:<28} {len(cache.data):>4} cached responses")

    splits = list(SPLITS.items())
    baselines = []
    for sname, pairs in splits:
        baselines.append((
            score(sim_fn, lambda *_a: True, pairs),          # no verifier
            score(sim_fn, verify_cardinality_or_marker, pairs),
            best_single_threshold(sim_fn, pairs),
        ))
    for run in runs:
        backend, plabel, verifier = run[0], run[1], run[2]
        print(f"  scoring {backend.model} / {plabel} ...")
        run[3] = [score(sim_fn, verifier, pairs) for _s, pairs in splits]

    print("\n" + "=" * 94)
    print("RESULT")
    print("=" * 94)
    header = (f"  {'approach':<46}" + "".join(f"{s:>16}" for s, _ in splits))
    print(header)
    print("  " + "-" * (len(header) - 2))

    def line(label, values):
        print(f"  {label:<46}" +
              "".join(f"{got:>10}/{total:<5}" for got, total in values))

    line("stage 1 only, no verifier",
         [(b[0][0], b[0][1]) for b in baselines])
    line("cheap verifier (card + marker)",
         [(b[1][0], b[1][1]) for b in baselines])
    line("best possible single threshold",
         [(b[2], b[0][1]) for b in baselines])
    for backend, plabel, _v, scores, _c in runs:
        line(f"{backend.model} — {plabel}", [(r[0], r[1]) for r in scores])
    line("ceiling (stage-1 recall limit)",
         [(r[1] - r[2], r[1]) for r in runs[-1][3]])

    print("\n  Each verifier by side of the ledger. False merges are the")
    print("  irreversible errors; missed links only create recoverable duplicates.")
    print(f"    {'model':<24}{'prompt':<28}{'split':<14}{'must-link':>12}"
          f"{'must-not-link':>15}{'false merges':>14}")
    for backend, plabel, _v, scores, _c in runs:
        for (sname, _pairs), res in zip(splits, scores):
            _c2, _t, _miss, lo, npos, co, nneg = res
            print(f"    {backend.model:<24}{plabel:<28}{sname:<14}"
                  f"{f'{lo}/{npos}':>12}{f'{co}/{nneg}':>15}{nneg - co:>14}")

    print(f"\n  calls: "
          f"{sum(c.misses for *_x, c in runs[::len(prompts)])} new, "
          f"{sum(c.hits for *_x, c in runs[::len(prompts)])} cached, "
          f"{sum(r[2].unparseable for r in runs)} unparseable, "
          f"{sum(r[2].errors for r in runs)} failed")

    # Latency decides whether verification can sit inside remember() or has to
    # be deferred to a background pass. Only live (uncached) calls are timed.
    timed = [(r[0].model, r[2].live_calls, r[2].live_seconds)
             for r in runs if r[2].live_calls]
    if timed:
        print(f"\n  latency of live calls (cached runs report nothing):")
        for model, n, secs in timed:
            print(f"    {model:<28}{secs / n:>6.2f} s/call over {n} calls")

    # ── which recall bar to ship: the bar trades stage-1 misses against calls ─
    print("\n" + "=" * 94)
    print(f"RECALL BAR SWEEP — {runs[-1][0].model}, {runs[-1][1]}, held-out")
    print("=" * 94)
    print("  A lower bar recovers must-link pairs stage 1 would drop, at the cost of")
    print("  sending more pairs to the model and giving it more chances to merge.\n")
    ml_h, mnl_h = SPLITS["held-out"]
    print(f"    {'bar':>6}{'total':>10}{'must-link':>12}{'must-not-link':>16}"
          f"{'lost to recall':>16}{'model calls':>13}")
    for bar in (0.40, 0.30, 0.20, 0.10, 0.00):
        res = score(sim_fn, llm, (ml_h, mnl_h), bar=bar)
        correct, total, miss, lo, npos, co, nneg = res
        calls = sum(1 for p in ml_h + mnl_h if sim_fn(p[3], p[1]) >= bar)
        print(f"    {bar:>6.2f}{f'{correct}/{total}':>10}{f'{lo}/{npos}':>12}"
              f"{f'{co}/{nneg}':>16}{miss:>16}{calls:>13}")

    # ── per-cell, held-out: is any structural shape still systematically wrong?
    print("\n" + "=" * 94)
    print(f"CELL BREAKDOWN — {runs[-1][0].model}, {runs[-1][1]}, held-out")
    print("=" * 94)
    ml, mnl = SPLITS["held-out"]
    cells = {}
    for side, plist, want in (("link", ml, True), ("coexist", mnl, False)):
        for (_sd, st, nd, nt, _note) in plist:
            key = f"{side:<8}{cell(nd, nt)}"
            row = cells.setdefault(key, [0, 0])
            row[1] += 1
            s = sim_fn(nt, st)
            row[0] += ((s >= BAR and llm(nt, st, nd, s)) == want)
    for key in sorted(cells):
        ok, total = cells[key]
        print(f"    {key:<26}{ok:>3}/{total:<3}"
              f"{'' if ok == total else '   <-- still wrong for this shape'}")

    print("\n" + "=" * 94)
    print("REMAINING FAILURES — held-out")
    print("=" * 94)
    for side, plist, want in (("must-link missed", ml, True),
                              ("must-not-link merged", mnl, False)):
        print(f"\n  {side}:")
        none = True
        for (_sd, st, nd, nt, note) in plist:
            s = sim_fn(nt, st)
            linked = s >= BAR and llm(nt, st, nd, s)
            if linked == want:
                continue
            none = False
            d = llm.detail(nt, st)
            verdict = ("stage 1 missed it" if s < BAR else
                       f"subject={d[1]} attribute={d[2]} -> {d[0]}" if d
                       else "unparseable")
            print(f"    sim={s:.2f} [{verdict}] {nd:<21} {note}")
            print(f"      stored: {st}")
            print(f"      new:    {nt}")
        if none:
            print("    (none)")

    print("\n" + "=" * 94)
    print("READING")
    print("=" * 94)
    print("  The held-out column is the only one that means anything: the cheap")
    print("  verifier's signals and this prompt's framing both came from looking at")
    print("  dev. The LLM verifier earns its cost only by beating the best single")
    print("  threshold on held-out — matching it is not enough, because a threshold")
    print("  is free and needs no model call per candidate pair.")


if __name__ == "__main__":
    main()
