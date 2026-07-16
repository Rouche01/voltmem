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
score     = similarity · (1 - V_d · staleness)
```

See `voltmem/scoring.py` → `staleness()`, `retrieval_score()`.

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

## How the two problems interact

```
Wrong domain label  →  wrong volatility prior  →  wrong escalation + retrieval
        ↑                                              ↓
   Problem 1                                      Problem 2
 (classification)                          (stable-fact update gap)
```

Fixing one without the other is incomplete:

- Better escalation alone does not fix a fact labeled `core_preference` when it should be
  `professional_context`
- Better classification alone does not fix the `professional_context` probe failure when the
  label is correct but protection is too strong

---

## Priority (suggested)

| Priority | Item | Rationale |
|---|---|---|
| P0 | Fix `professional_context` escalation probe | Done — band + relative θ cap; `voltmem_eval` 20/20 |
| P1 | Cumulative mismatch escalation | Done — see `CUMULATIVE_MISMATCH_ESCALATE` |
| P1 | High-M explicit override (drift-safe) | Done — `explicit_theta_cap()` + `calibrate_escalation.py` |
| P1 | Expand escalation eval + CI | Done — below/medium-band + cumulative probes |
| P2 | Classification eval corpus | Measure Problem 1; needed before LLM classifier work |
| P2 | Optional TTL hybrid (`expires_at`) | App-defined lifetimes; complements volatility staleness |
| P2 | Cloud LLM classifier | Roadmap item; does not alone fix drift |
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
- arXiv checklist: [../paper/ARXIV_SUBMISSION.md](../paper/ARXIV_SUBMISSION.md)
