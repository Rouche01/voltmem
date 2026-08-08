# VoltMem Development Roadmap — Post-Sleeptime Compute Reframe

**Version:** 0.3.0  
**Date:** 2026-08-08  
**Based on:** [Sleeptime Compute blog post](../blog-sleeptime-compute.md) + [OPEN_PROBLEMS.md](OPEN_PROBLEMS.md)

---

## TL;DR

The sleeptime compute insight reframes VoltMem's open problems: the real gap is not just better escalation math, but a **maintenance layer** that operates during idle cycles to integrate, relabel, and restructure memories after write-time constraints have passed.

**Shipped in 0.3.0:** enabling API (`event_id` + multi-facet + TTL), classification corpus, and maintenance substrate (tasks, ledger/rollback, sidecar daemon, WAL). **Next:** real `consolidate` content, keyword/collision fixes, dogfood (stylens), then smarter reclassify.

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

### 3.1 Multi-Facet `event_id` + Multi-Write API **[DONE — 0.3.0]**

**Shipped:** `event_id` / `modality` on `MemoryItem`; store + vector index metadata; `add_event()` + `retrieve_by_event()`; sidecar + TS client; tests for linked retrieval.

**Still open:** multimodal payloads / store adapters (P3); richer classifier multi-label path.

---

### 3.2 Optional TTL Hybrid (`expires_at`) **[DONE — 0.3.0]**

**Shipped:** `expires_at` / `ttl_seconds` on write APIs; retrieval skips expired; `expire_cleanup` via `MemoryLayer.purge_expired()` (store + vector index); sidecar / TS support.

**Still open:** domain-level TTL templates; optional haystack expiry bench expansion.

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

### 3.4 Maintenance Window Infrastructure **[DONE substrate — 0.3.0]**

Not a polish product feature — the substrate that enables P2-sleeptime operations.

**Shipped:**

1. `MaintenanceWindow` + task registry (`voltmem/maintenance.py`)
2. Tasks: `expire_cleanup`, `reclassify_ambiguous`, `pattern_audit`, `consolidate` (content still a **stub** — dry-run / opt-in only by default)
3. Ledger tables + `rollback_maintenance(run_id)` for supersede / purge snapshots
4. Sidecar: `POST .../maintenance/trigger`, `.../rollback`; background `MaintenanceScheduler` (`VOLTMEM_MAINTENANCE=1`) for expire / pattern_audit / reclassify (**not** consolidate)
5. File-backed SQLite WAL + busy timeout on store and vector index

**Still open:** replace consolidate stub with real merge; confidence-driven reclassify quality; richer scheduling UX.

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
Phase 1 — Enabling API                              ✅ 0.3.0
├── 1.1 event_id on MemoryItem + store migration          ✅
├── 1.2 add_event() + retrieve_by_event() API             ✅
├── 1.3 expires_at on MemoryItem + retrieval filtering    ✅
└── 1.4 Classification eval corpus (curate, measure)      ✅ baseline

Phase 2 — Maintenance Launch                         ✅ substrate 0.3.0
├── 2.1 MaintenanceWindow scaffold + task registry        ✅
├── 2.2 expire_cleanup (purge + index sync + ledger)      ✅
├── 2.3 reclassify_ambiguous (flag / secondary path)      ✅ scaffold
├── 2.4 pattern_audit (mismatch clusters → review)        ✅ scaffold
├── 2.5 Ledger + rollback_maintenance                     ✅
├── 2.6 Sidecar daemon + trigger/rollback HTTP            ✅
└── 2.7 Real consolidate (merge supersedes)               ❌ next

Phase 3 — Integration & Polish
├── 3.1 Sidecar maintenance endpoints                     ✅
├── 3.2 TypeScript client: event_id + expires_at + maint  ✅
├── 3.3 Keyword / collision fixes (feel→opinion, …)       ❌ next
├── 3.4 Cloud LLM classifier (now that corpus exists)     ❌ later
└── 3.5 Full multimodal payloads (after event linkage)    ❌ later
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
