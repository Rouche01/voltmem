# VoltMem — open problems & suggested directions

Tracking doc from feedback and internal eval findings (Jul 2026).
For reproduction commands see [RESEARCH.md](RESEARCH.md).

---

## Problem 1 — Classification brittleness (write-time labeling)

### What it is

Assigning a domain (and therefore a volatility prior) at write time is its own judgment call.
If the label is wrong, write protection and retrieval freshness are wrong for that fact.

### Symptoms

- Keyword heuristics misfire on phrasing or rule order (see `voltmem/extract.py` `_KEYWORDS`)
- Optional Ollama LLM classification (`classifier="llm"`) can drift between runs/models
- `auto_discover=True` learns volatility from confirm/contradiction **rates**, but does not
  relabel a fact that was assigned the wrong domain at write time
- Cold-start: `auto_discover` needs observations before empirical volatility is useful
  (`discovery.py`: `MIN_OBSERVATIONS = 3`)

### What exists today

| Mechanism | Location | Notes |
|---|---|---|
| Keyword heuristics (default) | `HeuristicClassifier` / `extract.py` | Zero deps, fast, brittle |
| Ollama LLM classifier | `LLMClassifier`, `classifier="llm"` | Higher quality, needs local Ollama |
| Custom keyword maps | `KeywordClassifier` | App-specific vocabulary |
| Chained classifiers | `ChainedClassifier` | Keyword first, heuristic fallback |
| Custom domain registry | `DomainRegistry`, `create_memory(domains=...)` | Register domains + volatility upfront |
| Confirm/contradiction learning | `auto_discover=True` | Blends empirical volatility with priors |

Example — custom domains: `examples/custom_classifier.py`

### Suggested directions

- [ ] **Smarter LLM classification** — cloud LLM support, structured JSON output, pinned prompts
- [ ] **Better defaults** — reduce keyword collision (e.g. `"feel"` → emotion vs opinion)
- [ ] **Automatic domain discovery** — infer new domains + priors from context (not just tune existing ones)
- [ ] **Classification confidence** — low-confidence label → wider volatility band or human/audit path
- [ ] **Deferred labeling** — store raw fact, assign domain on retrieval or after N observations
- [ ] **Eval suite for classification** — labeled corpus per domain, measure mislabel rate

### Practical mitigation (production today)

If you already know useful domain types for your app, register them upfront:

```python
from voltmem import create_memory, DomainRegistry, KeywordClassifier, ChainedClassifier, HeuristicClassifier

domains = DomainRegistry()
domains.register("client_relationship", 0.35)
domains.register("active_deal_stage", 0.70)

mem = create_memory(
    "app.db",
    user_id="alice",
    domains=domains,
    classifier=ChainedClassifier([
        KeywordClassifier({
            "client_relationship": ["client", "account", "stakeholder"],
            "active_deal_stage": ["deal", "pipeline", "closing"],
        }),
        HeuristicClassifier(),
    ]),
)
```

This does **not** remove write-time judgment; it narrows the label space to domains you control.

---

## Problem 2 — Stable facts that genuinely change (escalation gap)

### What it is

Protection that blocks noisy updates on stable domains can also block **legitimate** updates
(job change, career shift, long-held preference that truly changed).

Core tension: **don't corrupt on noise** vs **don't miss a real change**.

### Symptoms

- Berlin → Paris works: `location` prior is volatile (`V_d ≈ 0.60`) → updates readily
- Career change on `professional_context` (`V_d ≈ 0.30`) used to fail even with explicit
  statement + high mismatch magnitude — **fixed** via band θ-cap (see below)
- Regression suite: `python experiments/voltmem_eval.py` (Battery A, 20 probes)

### Why the math makes this hard

In `voltmem/scoring.py`, low volatility penalizes escalation twice:

```
E_t     = (M_t · R_t / C^α) · V_d · G_factor   ← V_d in numerator (smaller E_t)
θ_t     = θ_0 · (1 / V_d) · L_t                 ← V_d in denominator (larger threshold)
```

Strong `M_t` and `explicit_statement` are **designed** to override protection, but for
stable-ish domains the combined penalty can still block audit.

Logged mismatches increment `mismatch_count` but do **not** currently lower the threshold
on subsequent attempts (cumulative override at N strikes is separate — see below).

### Drift-safe explicit override (implemented)

Fixed `EXPLICIT_THETA_CAP` was sensitive to `auto_discover` drift. Replaced with:

| Constant | Value | Role |
|---|---|---|
| `EXPLICIT_MIN_VD` | 0.15 | Below: very stable — no θ-cap (cumulative fallback) |
| `EXPLICIT_MAX_VD` | 0.55 | Above: volatile — raw `E_t > θ_t` suffices |
| `EXPLICIT_E_RATIO` | 0.85 | Cap θ at `V_d × ratio` within the band |

Re-calibrate after changing globals: `python experiments/calibrate_escalation.py`

Regression grid: `test_escalation_medium_stable_v_grid_explicit_updates`,
`test_escalation_very_stable_v_grid_explicit_retains` in `tests/test_voltmem.py`.

### What exists today

| Mechanism | Notes |
|---|---|
| Source reliability `R_t` | `explicit_statement` = 1.0 vs `weak_inference` = 0.4 |
| Mismatch magnitude `M_t` | From classifier or caller |
| Repetition count `C` | Entrenched facts harder to overwrite |
| Per-item volatility EMA | Drifts ±0.2 from prior; does not bypass θ_t |

Relevant eval probes: `experiments/voltmem_eval.py` → `ESCALATION_PROBES`

### Suggested directions

- [x] **High-M_t explicit override** — band `[EXPLICIT_MIN_VD, EXPLICIT_MAX_VD]` with
  relative θ cap `V_d × EXPLICIT_E_RATIO` (see `explicit_theta_cap()` in `scoring.py`)
- [x] **Cumulative mismatch escalation** — after N `logged_mismatch` events, force audit
- [ ] **Life-event domains or triggers** — detect "changed careers", "moved", "divorced" language;
  route to medium-volatility slot or temporary override
- [ ] **Separate write vs retrieve volatility** — stable at retrieval, more plastic at write for
  certain domains
- [ ] **User-confirmed override API** — `mem.add(..., force_update=True)` for explicit corrections
- [x] **Expand eval** — below-band one-shot retain + cumulative update for `biographical` /
  `core_preference`; medium-band `skill` / `relationship` / `long_term_goal`; CI via
  `tests/test_voltmem.py` + `experiments/voltmem_eval.py`

---

## Problem 3 — Under-specified retrieval (similarity plateaus)

*From dev.to feedback (Jul 2026). Retrieval-time issue; not write-path classification.*

### What it is

Volatility re-ranking assumes a useful similarity gap between candidates:

```
staleness = 1 - exp(-V_d · age_days)
score     = similarity · (1 - mix · V_d · staleness)
```

`mix` defaults to 1 (full freshness). When top-candidate similarity spread is
flat, `mix` drops toward `MIX_MIN` so volatility cannot dominate near-ties.

When the **query** is specific ("where do I live?", "what's my allergy?"), similarity
already separates good hits from noise, and the staleness term is a useful second filter.

When the **query** is under-specified ("what was I working on?", "remind me about
that thing"), many memories look about equally related. Similarity scores flatten.
Then a slightly more similar high-volatility memory can outrank a stabler memory that
is the better answer — the similarity gap is smaller than the volatility penalty.

### Symptoms

- Open / temporal / multi-session prompts where several domains are vaguely relevant
- LongMemEval-S (n=60): `voltmem_real` **ties** cosine answer@5 (0.700) and loses
  slightly to flat (0.717) — consistent with "freshness helps on some axes, not a
  free win when similarity is uninformative"
- Haystack and scripted benches (specific current-truth probes) still favor volatility
  re-rank — the failure mode is query shape, not "re-rank is always wrong"

### What exists today (Jul 2026)

| Mechanism | Location | Notes |
|---|---|---|
| Cosine (or index) candidate fetch | `MemoryLayer` | First-stage semantic proximity |
| `retrieval_score(..., mix=)` | `scoring.py` | `sim · (1 − mix · V_d · staleness)` |
| Adaptive `freshness_mix` | `scoring.py` / `retrieve()` | Plateau dampening from sim spread |
| Specificity + sim-spread report | `experiments/longmemeval.py` | `--report-specificity` (default on) |
| Synthetic plateau probe | `experiments/retrieval_plateau_probe.py` + tests | CI-fast failure-mode check |
| Vector index (v0.2) | `vector_index.py` | Faster candidates; same re-rank + mix |

Specificity buckets (fixed):

- **specific:** `knowledge-update`, `single-session-preference`, `single-session-user`, `single-session-assistant`
- **open:** `temporal-reasoning`, `multi-session`

Plateau constants: `SIM_SPREAD_FLAT=0.05`, `SIM_SPREAD_FULL=0.15`, `MIX_MIN=0.25`.
Spread is measured on the top similarity pool (`top_k · candidate_multiplier`).

No answerability / cross-encoder second stage yet.

### Suggested directions

- [x] **Eval slice by query specificity** — LongMemEval reports specific vs open + avg sim spread
- [ ] **Answerability / task-affinity rerank** — second stage after volatility score:
  entity grounding, session/task recency, "current project" affinity, or a small
  cross-encoder — not a replacement for `V_d` staleness; only if open slice still lags
- [x] **Query-aware penalty scaling** — dampen freshness via `mix` when top-pool similarity
  spread is low (plateau detected)
- [ ] **Keep TTL separate** — optional hard expiry (Enhancement below) does not fix
  under-specified ranking

### What this does not fix

- Wrong domain at write time (Problem 1)
- Stable-fact escalation gap (Problem 2)
- Multi-facet / multimodal events (Problem 4)
- Known calendar end dates (optional TTL)
- Full answerability ranking (deferred)

### Priority

P2 — specificity eval + adaptive mix shipped; answerability only if open-slice gap persists.

---

## Problem 4 — Multi-facet multimodal events (beyond chat memory)

*From dev.to feedback (Jul 2026). Scope expansion: any agent that needs memory,
not only chat-style text facts.*

### What it is

VoltMem today assumes roughly: **one observation → one text fact → one domain →
one volatility prior**. That fits conversational agents. It underfits domains where
a single real-world tick carries several kinds of signal at once — e.g. a robot
sample with GPS (location), IMU (activity), voice command (task), and a map patch
(spatial). Those facets must stay **linked as one event** while decaying and
auditing **independently**.

The volatility thesis is modality-agnostic: a corridor map should not forget at the
same rate as a battery estimate from the same 50 ms tick. The gap is API + item
shape, not the math in `scoring.py`.

**Framing:** not “every fact must be multimodal,” but “every observation *can* be a
multi-facet event.” Plain text facts remain first-class.

**Scope discipline:** VoltMem stays the **policy layer** (domain → \(V_d\), write
protection, freshness re-rank, audit). It should not become a multimodal database
or embedding model suite — storage and encoders stay pluggable.

### Symptoms / current limits

- `MemoryItem.content` is a single string; no `event_id`, modality, or binary/structured payload
- `mem.add(...)` classifies and stores one domain per call
- Classifiers and embeddings are text-oriented (`extract.py`, `embeddings.py`)
- No retrieve-by-event (“give me the whole tick”) vs retrieve-by-facet (“stable map only”)

### What exists today

| Mechanism | Notes |
|---|---|
| Per-item domain + \(V_d\) | Already supports different decay if facets are separate items |
| Optional TTL (Enhancement) | Composes with per-facet expiry once facets exist |
| Pluggable `similarity_fn` / embeddings | Path to non-text encoders without rewriting escalation math |
| Custom `DomainRegistry` | App can register robot/sensor domains + priors |

### Suggested directions (phased)

- [ ] **`event_id` + multi-write** — one observation → N domain-tagged items that share
  an event key (and optional `modality`: text / image / audio / sensor / structured)
- [ ] **Modality-agnostic item shape** — text *or* structured/binary payload + embedding
  reference; same `effective_volatility`, escalation, and `retrieval_score` path
- [ ] **Per-facet volatility / TTL** — each facet keeps its own domain prior (and optional
  `expires_at`); event linkage does not force a shared forget rate
- [ ] **Event-aware retrieve** — APIs for “expand event”, “prefer facet domain”, or
  assemble a unified view for the agent without collapsing policies
- [ ] **Classifier path for multi-label** — emit several (domain, content/payload) facets
  per observation; confidence per facet (ties to Problem 1)
- [ ] **Eval** — synthetic multi-facet ticks (stable map + volatile battery); assert
  linked retrieval + independent stale@k / audit behavior
- [ ] **Later: store adapters** — optional backends (SQLite today, vector DB, multimodal
  stores) behind the policy API — not a rewrite of VoltMem as the store

Sketch:

```python
mem.add_event(
    event_id="tick-50ms-001",
    facets=[
        {"content": "corridor map patch A12", "domain": "spatial_map", "modality": "structured"},
        {"content": "battery 37%", "domain": "power_state", "modality": "sensor"},
        {"content": "go to charging dock", "domain": "current_task", "modality": "text"},
    ],
)
# Same event_id; each facet has its own V_d / audit / staleness.
```

### What this does not fix

- Classification quality for each facet (Problem 1) — multi-label makes it harder
- Stable-fact escalation (Problem 2)
- Under-specified “what’s going on?” queries (Problem 3) — multimodal open prompts
  make answerability rerank *more* important
- Building a general multimodal DB product

### Priority

P3 for full multimodal payloads / store adapters; **P2 for `event_id` + multi-write**
as the enabling API — unblocks robotics/IoT/tool agents without changing core math.

---

## Enhancement — Optional TTL hybrid (time-limited memories)

*From dev.to feedback (Jul 2026). Complements volatility-weighted decay; does not replace it.*

### What it is

Some facts have a **known shelf life** that domain volatility priors do not express well.
Volatility answers "how fast does this *kind* of fact usually go stale?"
Optional TTL answers "this *specific* fact should not live past time T."

Examples:

- "User is in Berlin for a conference until Friday"
- Active deal stage or campaign context with a known end date
- Session-scoped or trial-period facts the app already knows will expire

### What exists today

VoltMem uses **soft, domain-weighted decay at retrieval** (not hard expiry):

```
staleness = 1 - exp(-V_d · age_days)
score     = similarity · (1 - mix · V_d · staleness)
```

See `voltmem/scoring.py` → `staleness()`, `retrieval_score()`, `freshness_mix()`.

`MemoryItem` has `created_at` and `last_confirmed_at` but **no** `expires_at` field.
Expired facts are down-ranked (volatile domains faster), not dropped by a hard cutoff.

| | Volatility staleness (today) | Optional TTL (proposed) |
|---|---|---|
| Question answered | What *type* of fact is this? | When must *this* fact die? |
| Behavior | Soft penalty at search | Hard cutoff or heavy penalty past `expires_at` |
| Good for | Mood, prefs, location patterns | Session context, deals, dated events |
| Bad for | Facts with known calendar end | Stable prefs, biography (wrong if TTL'd) |

### Why a hybrid is worth building

- **Complement, not replacement** — volatility handles fact-type behavior; TTL handles
  app-defined lifetimes
- **Low effort, low risk** if strictly optional (`expires_at=None` by default)
- **Production ergonomics** — explicit contract vs hoping high `V_d` decay is fast enough

### What TTL does not fix

- Classification brittleness (Problem 1)
- Stable-fact escalation gap (Problem 2)
- Under-specified retrieval / similarity plateaus (Problem 3)
- Multi-facet / multimodal event linkage (Problem 4) — TTL is complementary once facets exist
- Wrong TTL on a stable fact is its own failure mode (same class as misclassification)

### Suggested design

- [ ] **`expires_at` on `MemoryItem`** — optional unix timestamp (or `ttl_seconds` on write)
- [ ] **Write API** — `mem.add(text, expires_at=...)` or `ttl_seconds=86400`
- [ ] **Retrieval** — exclude or score=0 when `now > expires_at`; compose with existing
  volatility staleness (both apply)
- [ ] **Optional purge** — background or on-read deletion of expired rows
- [ ] **Domain defaults** — e.g. register `active_deal_stage` with suggested TTL template (app sets per fact)
- [ ] **Eval** — haystack bench with expired items; assert 0% retrieval past expiry

Sketch:

```python
mem.add(
    "User is trialing Product X for two weeks",
    domain="current_project",
    expires_at=time.time() + 14 * 86400,
)
```

At retrieve: if `expires_at` is set and in the past → skip item (or rank last).

### Priority

P2 — useful for production apps with known-lifetime context; not urgent vs P0/P1 escalation fixes.

---

## Enhancement — Prior calibration telemetry

*From dev.to feedback (Jul 2026). Observability / calibration — not a new memory
mechanism. Makes hand-tuned domain priors and audit thresholds checkable in real use.*

### What it is

Domain priors (\(V_d\)) and the audit decision \(E_t > \theta_t\) encode assumptions
about how often each kind of fact should change. Without per-domain counts of what
actually happens at write time, it is hard to tell which priors are too conservative
(almost never audit) and which are too permissive (audit constantly).

This enhancement is **measurement**: expose rates so operators and researchers can
validate the library’s defaults — and tune `DomainRegistry` / `auto_discover` with
evidence.

### What “right” looks like (informal)

| Signal | Too conservative | Too permissive |
|---|---|---|
| Audit rate for volatile domains (e.g. mood, location) | Near zero despite contradictions | — |
| Audit rate for stable domains (e.g. biographical) | — | High on weak / noisy mismatches |
| `logged_mismatch` without eventual audit | Mismatches pile up; threshold never crossed | — |
| Confirm vs contradict mix | — | Contradicts dominate “stable” domains |

A simple **histogram (or table) of audit / mismatch / confirm counts per domain**
answers the commenter’s ask and pairs well with the REAL > SHUFFLE > SWAP causal story.

### What exists today (Jul 2026)

| Mechanism | Notes |
|---|---|
| Per-item `mismatch_count` | Cumulative below-threshold mismatches on that item |
| Always-on `VolatilityTracker` | Records insert / confirm / mismatch / audit per domain |
| `MemoryLayer.domain_stats()` / `Memory.domain_stats()` | Public calibration table + rates |
| `auto_discover=True` | Optional blend of empirical \(V_d\) into scoring (separate from telemetry) |
| `calibrate_escalation.py` | Offline θ-band calibration |
| Eval footprint | `experiments/voltmem_eval.py` prints Battery A domain_stats replay |

```python
stats = mem.domain_stats()
# {
#   "location": {
#     "prior": 0.6, "inserted": 3, "confirmed": 10,
#     "logged_mismatch": 2, "audited": 4,
#     "audit_rate": 0.25, "mismatch_rate": 0.125, ...
#   },
# }
```

**How to read:** high `audit_rate` on stable domains (e.g. biographical) → prior/threshold
too permissive; near-zero `audit_rate` on volatile domains despite mismatches → too
conservative. Pair with REAL > SHUFFLE > SWAP causal benches.

### Suggested design

- [x] **Counters per domain** — `audited`, `logged_mismatch`, `confirmed`, `inserted`
- [x] **Export API** — `mem.domain_stats()` → JSON-ready table with rates
- [ ] **Optional log sink** — append-only events for offline analysis without holding
  state in the hot path
- [x] **Docs** — reading guide above + OPEN_PROBLEMS / RESEARCH pointers
- [x] **Eval hook** — Battery A calibration footprint in `voltmem_eval.py`
- [x] **Histogram script** — `experiments/prior_calibration_hist.py` (ASCII + SVG; optional PNG)
- [ ] **Mismatch magnitude buckets** — optional finer histogram

### What this does not fix

- Wrong labels (Problem 1) — telemetry *reveals* miscalibration; it does not relabel
- Escalation math gaps (Problem 2) — may show θ is too high/low; fixing still needs
  scoring / API changes
- Under-specified retrieval (Problem 3) — write-path rates, not search ranking
- Multi-facet events (Problem 4) — extend counters by `(domain, modality)` later

### Priority

P2 — **shipped** for counters + `domain_stats()`; log sink / magnitude buckets remain optional.

---

## How the problems interact

```
Wrong domain label  →  wrong volatility prior  →  wrong escalation + retrieval
        ↑                                              ↓
   Problem 1                                      Problem 2
 (classification)                          (stable-fact update gap)
        ↑                                          ↓
   Problem 4                                  Problem 3
 (multi-facet events:                  (under-specified queries:
  N labels / payloads                   similarity flat → V_d
  per observation)                      penalty can invert rank)

Prior calibration telemetry (Enhancement)
  → measures audit / mismatch / confirm rates per domain
  → feeds tuning of Problem 1 priors and Problem 2 thresholds
```

Fixing one without the others is incomplete:

- Better escalation alone does not fix a fact labeled `core_preference` when it should be
  `professional_context`
- Better classification alone does not fix the `professional_context` probe failure when the
  label is correct but protection is too strong
- Correct labels + escalation still leave open prompts where similarity is uninformative
  and volatility re-rank can pick the wrong winner (Problem 3)
- Multi-facet events (Problem 4) multiply Problem 1 (N labels per tick) and make
  Problem 3 more common (“what’s going on?” across sensors); they do not replace
  volatility math — they require event linkage + per-facet policy
- Telemetry does not replace any of the above — it makes prior/threshold assumptions
  visible so fixes are evidence-driven

---

## Priority (suggested)

| Priority | Item | Rationale |
|---|---|---|
| P0 | Fix `professional_context` escalation probe | Done — band + relative θ cap; `voltmem_eval` 20/20 |
| P1 | Cumulative mismatch escalation | Done — see `CUMULATIVE_MISMATCH_ESCALATE` |
| P1 | High-M explicit override (drift-safe) | Done — `explicit_theta_cap()` + `calibrate_escalation.py` |
| P1 | Expand escalation eval + CI | Done — below/medium-band + cumulative probes |
| P2 | Classification eval corpus | Measure Problem 1; needed before LLM classifier work |
| P2 | Prior calibration telemetry | Done — `domain_stats()` always on; optional log sink later |
| P2 | Under-specified retrieval (Problem 3) | Done — specificity report + adaptive mix; answerability deferred |
| P2 | Multi-facet `event_id` + multi-write (Problem 4) | Enabling API for non-chat agents; core math unchanged |
| P2 | Optional TTL hybrid (`expires_at`) | App-defined lifetimes; complements volatility staleness |
| P2 | Cloud LLM classifier | Roadmap item; does not alone fix drift |
| P3 | Multimodal payloads / store adapters (Problem 4) | After event linkage; keep VoltMem as policy layer |
| P3 | Automatic domain discovery | Larger research scope |

---

## References

- Dev.to article draft: [DEVTO_POST_MERGED.md](DEVTO_POST_MERGED.md)
- Escalation math: [../voltmem/scoring.py](../voltmem/scoring.py)
- Domain priors: [../voltmem/domains.py](../voltmem/domains.py)
- Auto-discovery: [../voltmem/discovery.py](../voltmem/discovery.py)
- Classifiers: [../voltmem/classifiers.py](../voltmem/classifiers.py), [../voltmem/extract.py](../voltmem/extract.py)
- End-to-end eval: [../experiments/voltmem_eval.py](../experiments/voltmem_eval.py)
- Escalation calibration: [../experiments/calibrate_escalation.py](../experiments/calibrate_escalation.py)
- LongMemEval / research notes: [RESEARCH.md](RESEARCH.md)
- arXiv checklist: [../paper/ARXIV_SUBMISSION.md](../paper/ARXIV_SUBMISSION.md)
