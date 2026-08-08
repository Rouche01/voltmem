# VoltMem Development Roadmap — Post-Sleeptime Compute Reframe

**Version:** 0.2.2+ (in development)  
**Date:** 2026-08-06  
**Based on:** [Sleeptime Compute blog post](../blog-sleeptime-compute.md) + [OPEN_PROBLEMS.md](OPEN_PROBLEMS.md)

---

## TL;DR

The sleeptime compute insight reframes VoltMem's open problems: the real gap is not just better escalation math, but a **maintenance layer** that operates during idle cycles to integrate, relabel, and restructure memories after write-time constraints have passed. The next phase of VoltMem development shifts from reactive write-path fixes to **asynchronous maintenance operations** that bridge the gap between fast write-time decisions and long-term memory integrity.

**Immediate priority:** Build the enabling API for maintenance-window operations (`event_id` + multi-facet events), then add TTL support. These two features together unlock the architectural foundation for all subsequent maintenance work.

---

## 1. How Sleeptime Compute Reframes the Four Open Problems

| Problem | Original Framing | Sleeptime Reframe | New Priority |
|---|---|---|---|
| **P1 — Classification brittleness** | Better classifiers at write time | **Nightly re-classification** with full context; confidence-driven relabeling | P2 (infrastructure dependency) |
| **P2 — Stable facts that change** | Better escalation math | **Retrospective pattern detection** on logged_mismatches; emergent-change audits | P1 (primary beneficiary) |
| **P3 — Under-specified retrieval** | Answerability rerank | **Associative link building** during idle hours; co-occurrence graphs | P3 (weakest connection) |
| **P4 — Multi-facet events** | Synchronous multi-write API | **Asynchronous facet re-parsing** after hours; event linkage without write-path cost | P2 (enabling API) |

**Key insight:** The escalation math is reactive (new evidence → check threshold). Some changes are *emergent* (pattern builds over time → needs retrospective review). Maintenance windows catch what real-time thresholds miss.

### What sleeptime is / isn't

Keep write-path fixes and idle-path maintenance distinct — they solve different failure modes.

| Concern | Prefer | Sleeptime role |
|---|---|---|
| **Emergent change** (weak mismatches that never cross live θ) | — | **Primary job** — `pattern_audit` / real `consolidate` |
| **Keyword collisions** (e.g. `feel` → mood vs opinion) | Fix keywords / write-path classifier first | **Secondary** — `reclassify_ambiguous` with fuller context; score with the classification corpus |
| **Undated fleeting facts** | Label as `transient_fact` (high \(V_d\)) so retrieval ages them out fast | Not a reclassify problem |
| **Dated session facts** (“until Friday”) | Optional **TTL** (`expires_at` / `ttl_seconds`) | **`expire_cleanup`** only — purge after hard expiry |
| **Classification corpus (~84% heuristic)** | Grades labels and future relabelers | Not the sleeptime engine; the **answer key** for whether reclassify helped |

**One line:** Sleeptime is mostly for emergent change + cleanup; collisions → keywords first; transient → high \(V_d\) when undated, TTL when dated; the corpus measures Problem 1, it does not run at night.

---

## 2. What's Done (P0/P1 — Escalation Math)

These shipped in v0.2.2 and are confirmed stable:

- High-M explicit override with drift-safe θ band (`EXPLICIT_MIN_VD`, `EXPLICIT_MAX_VD`, `EXPLICIT_E_RATIO`)
- Cumulative mismatch escalation (`CUMULATIVE_MISMATCH_ESCALATE`)
- Expanded eval grids in `tests/test_voltmem.py` + `experiments/voltmem_eval.py` (20/20 Battery A)
- Adaptive freshness mix with plateau detection for under-specified retrieval
- Prior calibration telemetry (`domain_stats()` always-on)

---

## 3. Next Up: P2 Items (Ranked by Sleeptime Relevance)

### 3.1 Multi-Facet `event_id` + Multi-Write API **[START HERE]**

**Why first:** This is the **enabling API** for all maintenance operations. Without event linkage, facets of the same observation drift apart in the store and maintenance cannot reconstruct relationships.

**Realizes:** Blog thesis §4 ("Sleeptime compute semesterates the tradeoff") + OPEN_PROBLEMS Problem 4

**Goal:** One observation → N domain-tagged items sharing an `event_id`, each with independent `V_d` / audit / staleness.

**Concrete tasks:**

1. **Schema change:** Add `event_id` and `modality` to `MemoryItem` (default `None` for backward compat)
2. **Write API:** `MemoryLayer.add_event(event_id, facets=[...])`  
   Each facet: `{content, domain, modality?}`
3. **Store layer:** SQLite schema migration; indexed `event_id` column
4. **Retrieval:** `retrieve_by_event(event_id)` to reassemble full observations
5. **Vector index:** Include `event_id` in index metadata for filtered queries
6. **Eval:** Synthetic multi-facet ticks (stable map + volatile battery); assert linked retrieval + independent stale@k

**Does NOT include:**
- Full multimodal payloads (P3 — defer until event linkage is proven)
- Store adapters / vector DB backends (P3)

**Minimum viable:** `event_id` string on `MemoryItem`, `add_event()` writes N items, `retrieve_by_event()` returns them ordered by creation time. Modality field is optional decoration in MVP.

---

### 3.2 Optional TTL Hybrid (`expires_at`)

**Why second:** Complements `event_id` for session-scoped facts ("User is in Berlin for a conference until Friday"). These expire on calendar time, not volatility. Gives maintenance window a clear signal: "this event is definitely dead, don't bother auditing."

**Realizes:** OPEN_PROBLEMS TTL Enhancement

**Concrete tasks:**

1. **Schema change:** Add `expires_at` (unix timestamp, nullable) to `MemoryItem`
2. **Write API:** `mem.add(text, expires_at=...)` and/or `ttl_seconds=86400`
3. **Retrieval:** Skip or score=0 when `now > expires_at`; compose with existing `staleness`
4. **Optional purge:** Background or on-read deletion of expired rows (maintenance hook)
5. **Eval:** Haystack bench with expired items; assert 0% retrieval past expiry

---

### 3.3 Classification Eval Corpus **[DONE — baseline]**

**Why:** Needed before any LLM classifier improvements. Gives us a labeled benchmark to measure whether maintenance-based relabeling (the eventual P2/P3 work) actually helps.

**Shipped:**

1. Labeled corpus: [`tests/fixtures/classification_corpus.json`](../tests/fixtures/classification_corpus.json) — **230** utterances across all **14** built-in domains (SCHEDULE originally said 8; corpus covers the full `DOMAIN_VOLATILITY` set)
2. Loader + metrics: [`voltmem/classification_eval.py`](../voltmem/classification_eval.py)
3. Baseline report: `python experiments/classification_baseline.py` — HeuristicClassifier ≈ **84%** overall (≈0% on `transient_fact`; keyword domains ≈ **90%+**)
4. Collision cases tagged (`collision:feel`, etc.) — e.g. `"I feel that…"` gold=`opinion`, heuristic→`emotional_context`
5. CI: `python tests/test_classifiers.py` (schema, accuracy floors, feel collision, &lt;5s)

**Still open (not blocking the corpus):**

- Optional Ollama / cloud LLM accuracy pass on the same fixture (local-only; not CI)
- Raise floors after keyword fixes; grow corpus toward 500 if dogfood shows new collision classes

**Deferred:** Cloud LLM classifier (needs this corpus first so we can justify the complexity).

---

### 3.4 Maintenance Window Infrastructure **[FOUNDATIONAL, NOT A FEATURE]**

Not a user-facing feature. The substrate that enables P2-sleeptime operations.

**Concrete tasks:**

1. **Maintenance runner:** `MaintenanceWindow` class with pluggable tasks
2. **Task registry:** `register_maintenance_task(name, callable, interval_seconds)`
3. **Default tasks:**
   - `reclassify_ambiguous`: review facts with confidence < threshold across full history
   - `pattern_audit`: scan `logged_mismatch` clusters for emergent change signals
   - `expire_cleanup`: purge rows past `expires_at`
4. **Scheduling:** Thread-based (default) or external cron trigger
5. **Safety:** Runs on copy of state or with WAL mode; never blocks write path

**This is what the blog post calls "the maintenance window."** We don't ship it as a whole system. We ship the hooks that let it exist, then populate it task by task.

---

## 4. What's Deferred to P3

| Item | Reason |
|---|---|
| **Cloud LLM classifier** | Needs eval corpus (P2 task 3). Not a maintenance operation; it's a write-path improvement that can wait. |
| **Full multimodal payloads / store adapters** | Requires `event_id` proven + `modality` field populated. Don't build the storage layer before the abstraction is tested. |
| **Answerability rerank for Problem 3** | Weakest sleeptime connection. Query specificity is a retrieval-ranking issue, not an integration issue. Revisit if adaptive mix is insufficient in practice. |
| **Automatic domain discovery** | Research scope; needs corpus + classifier eval first. |

---

## 5. Work Schedule (Suggested Order)

```
Phase 1 — Enabling API (this cycle)
├── 1.1 event_id on MemoryItem + store migration          ✅
├── 1.2 add_event() + retrieve_by_event() API             ✅
├── 1.3 expires_at on MemoryItem + retrieval filtering    ✅
└── 1.4 Classification eval corpus (curate, measure)      ✅ baseline

Phase 2 — Maintenance Launch (next cycle)
├── 2.1 MaintenanceWindow scaffold + task registry
├── 2.2 expire_cleanup task (simplest: purge dead rows)
├── 2.3 reclassify_ambiguous task (needs corpus + confidence metric)
└── 2.4 pattern_audit task (scans mismatch clusters, flags for user review)

Phase 3 — Integration & Polish
├── 3.1 Sidecar: expose maintenance trigger endpoint
├── 3.2 TypeScript client: event_id + expires_at support
├── 3.3 Cloud LLM classifier (now that corpus exists)
└── 3.4 Full multimodal payloads (after event linkage proven)
```

---

## 6. How to Decide Within Each Phase

Ask three questions before starting any task:

1. **Does this block maintenance operations?**  
   If yes → do it first. (`event_id`, `expires_at`, `MaintenanceWindow` scaffold)

2. **Does this give us measurement we don't have?**  
   If yes → do it before any "smarter" version of the same thing. (Eval corpus before cloud LLM classifier.)

3. **Is this a quick win that validates the architecture?**  
   If yes → interleave with heavier tasks. (`expire_cleanup` is trivial and proves the maintenance loop works.)

---

## 7. Files to Touch

| File | Purpose |
|---|---|
| `voltmem/domains.py` | `MemoryItem` dataclass: add `event_id`, `modality`, `expires_at` |
| `voltmem/store.py` | Schema migration, `insert()`/`all_active()` updates, `purge_expired()` |
| `voltmem/memory.py` | `add_event()`, `retrieve_by_event()`, `expires_at` passthrough on `write()`/`observe()` |
| `voltmem/scoring.py` | `retrieval_score()` skip if expired |
| `voltmem/vector_index.py` | Include `event_id` in index metadata |
| `voltmem/maintenance.py` | New: `MaintenanceWindow`, task registry, default tasks |
| `tests/test_voltmem.py` | Event linkage tests, TTL retrieval tests |
| `tests/test_classifiers.py` | New: eval corpus loader + accuracy tests |
| `docs/SCHEDULE.md` | This file — update as tasks complete |

---

## 8. References

- Blog: `/Users/richardemate/Documents/blog-sleeptime-compute.md`
- Current roadmap: `/Users/richardemate/Projects/voltmem/docs/OPEN_PROBLEMS.md`
- Core memory layer: `/Users/richardemate/Projects/voltmem/voltmem/memory.py`
- Scoring: `/Users/richardemate/Projects/voltmem/voltmem/scoring.py`
- Store: `/Users/richardemate/Projects/voltmem/voltmem/store.py`

---

*This is a living document. Update it as P2 items ship and P3 priorities shift. The north star remains: make VoltMem not just reactive at write-time, but integrated through maintenance.*
