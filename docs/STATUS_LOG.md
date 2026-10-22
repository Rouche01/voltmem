# VoltMem — status log (plain English)

Living chronological log of where the project is and what we discussed.
Keep language simple. **Newest entries at the top.**

---

## Plain-English snapshot (updated 2026-10-21)

### What we are trying to do

Build a memory system for AI apps that treats facts differently based on how
fast they change — so “where I live” is hard to overwrite by mistake, and
“what I’m wearing today” can update easily and fade when old.

Ship it so real apps can use it: a Python library, a small HTTP service
(sidecar), and a TypeScript client — without baking any one app’s vocabulary
into the core product.

### Where we are now

The hard science and engine work is largely done. Recent work is about making
it clean, shippable, and fair for any app — not just one product.

### Done so far (big picture)

- Core idea works and is tested (including a formal Lean check of the scoring rules)
- Python package on PyPI (library ~0.5.0)
- TypeScript client on npm (~0.6.2)
- Sidecar Docker images published (through ~0.8.0)
- Clear “tenant” (whose memory bucket) vs “domain” (what kind of fact) wording and APIs
- Apps can bring their own domain list via a file — not hardcoded into VoltMem
- Memory browser UI: graph/list, clear tenant or one domain
- Default profile is neutral (`default`), not tied to GoStylens
- Sidecar builds aimed at both Apple Silicon (arm64) and typical servers (amd64)

### Still ahead (high level)

- **Primary dogfood: Quick Question** (sibling repo) — see [DOGFOOD.md](DOGFOOD.md)
- Small classifier / keyword fixes
- Cut another sidecar release if multi-arch + default-profile changes need a fresh tag

---

## The science / mechanism (plain English)

### The bet

Most AI memory systems treat every stored fact the same. That forces a bad
choice: either **everything** is easy to overwrite (stable facts get corrupted),
or **everything** is hard to overwrite (current truth never updates).

VoltMem’s bet: **how fast a kind of fact usually changes** should control two
things —

1. **How hard it is to overwrite** when something new arrives (write protection)
2. **How quickly an old version should lose rank** when you search (retrieval freshness)

So “hometown” and “today’s mood” are not peers. Same memory store; different
rules. We call that “current-truth” memory: remember what is true *now*, without
trashing what should stay sticky.

This is a **control knob** on the classic stability–plasticity tradeoff — not a
promise that we beat plain similarity search on every benchmark. We measure that
honestly (including cases where cosine still wins overall).

### What we have implemented

| Mechanism | Plain meaning |
|---|---|
| **Volatility by domain** | Each fact kind gets a “how changeable is this?” number. Sticky kinds resist updates; changeable kinds update and age out faster at search. |
| **Escalation / audit at write** | A new claim does not blindly replace an old one. Strength of evidence, reliability of the source, and how entrenched the old fact is all matter. Sticky facts need a higher bar. |
| **Composite control law (default)** | Default mode mixes “protect unless surprised” with “when evidence is strong or unexpected, allow change.” Older modes stay available. Spec’d in Lean; runtime is Python `scoring.py`. |
| **Heuristic write matching + twins** | Live writes stay fast: match by subject/attribute, not only vector similarity. When unsure, keep both versions (“twins”) rather than false-merge. False merge is worse than a duplicate. |
| **Sleeptime maintenance** | Background jobs clean up later: expire timed facts, reconcile twins (optional local model), consolidate mismatch evidence, pattern audit / reclassify hooks. Heavy thinking off the live path. |
| **Retrieval with freshness** | Search is not raw nearest-neighbor alone. Candidates are re-ranked so stale, high-volatility memories rank lower — unless the query is so vague that similarity scores are all tied, in which case we **dampen** freshness so it cannot dominate a coin-flip. |
| **Optional TTL** | Some facts have a hard end date (“until Friday”); retrieval skips them and cleanup can purge them. |
| **Multi-facet events (text)** | One real-world moment can store several linked facts (`event_id`) that still age and audit separately. |
| **Classification (write-time labeling)** | To apply the right volatility, we must guess the fact’s kind. Heuristics (and optional local LLM) do that; apps can force a domain or ship their own domains file. |
| **Prior learning (`auto_discover`)** | Over time, confirm/contradict rates can nudge how volatile a domain seems — without inventing wholly new domain names yet. |

### What we plan to introduce (not there yet)

These are on the roadmap; some have scaffolding only.

| Planned mechanism | Plain meaning | Status |
|---|---|---|
| **Better keyword / collision fixes** | Stop mislabeling phrases like “I feel that…” as mood instead of opinion. | Next polish |
| **Smarter / cloud LLM classifier** | Higher-quality labeling at write time, measured against our corpus. | Deferred until dogfood + keyword fixes justify it |
| **Confidence-aware labeling** | When unsure of the kind, use a wider band or defer / audit instead of a confident wrong prior. | Open |
| **Deferred labeling** | Store the raw fact first; assign domain after more context or N observations. | Open |
| **Automatic new domain discovery** | Infer brand-new fact kinds (not just tune existing ones). | Research / later |
| **Life-event triggers** | Phrases like “I changed careers” temporarily lower the bar for sticky domains. | Open |
| **Stronger night reclassify** | Maintenance already has a hook; make it good enough that wrong write-time labels get fixed with fuller context. | Scaffold exists; quality open |
| **Answerability / second-stage rerank** | After similarity + freshness, a smarter “does this actually answer the question?” step for vague queries. | Deferred unless adaptive mix is not enough in practice |
| **Full multimodal payloads** | Same event can carry GPS, image, audio, etc., with pluggable storage — VoltMem stays the policy layer, not the media DB. | Later (event linkage API exists; adapters do not) |
| **Separate write vs retrieve volatility** | Some kinds might be sticky when reading but a bit more plastic when writing. | Idea only |

Deeper technical tracking: [OPEN_PROBLEMS.md](OPEN_PROBLEMS.md), [SCHEDULE.md](SCHEDULE.md), [RESEARCH.md](RESEARCH.md).

---

## Chronological log

### 2026-10-21 — Quick Question as dogfood; how to understand a person

**Dialogue**

- Asked how fitting Quick Question is as dogfood, and what we can test.
- Asked how to elevate it to a winning showcase with a personal wow in 1–2 weeks.
- Compared two understanding mechanisms: onboarding/daily questions vs connecting apps the user already uses; asked what we are not thinking about yet.
- Asked specifically about connecting existing AI chat apps.
- Asked for a dedicated dogfood doc, plus insights from this thread that might not be obvious.

**Outcome**

- Judgment: QQ is the best first dogfood (confirm gate, own domains, current-truth story).
- Recommended wow: live agent + before/after/after-update on a tension (career vs evenings).
- Mechanism rank: **questions (tradeoffs) first**; apps as evidence; **surprise → one question** as the product loop. AI chats: distill + confirm, never bulk-ingest.
- Wrote [DOGFOOD.md](DOGFOOD.md). Highest missed insight: QQ upsert is delete+add, so it may not exercise VoltMem’s actual overwrite law yet.
- Cursor plan in sibling repo: `quick-question/.cursor/plans/current-truth-vault.plan.md`.

### 2026-10-21 — Status log + mechanism section

**Dialogue**

- Asked for a very simple, brief, layman update of what we were trying to do
  and what we have done, plus a document to keep a chronological log of the
  back-and-forth.
- Clarified: the log should live **in this repo**, not Notion.
- Asked to add a plain-language section on the science/mechanism we are betting
  on and have implemented, plus mechanisms we plan but have not built yet.

**Outcome**

- Created this file: `docs/STATUS_LOG.md`.
- Added **The science / mechanism** section (bet, shipped table, planned table).

### 2026-10-20 — Neutral product + multi-chip images

**Dialogue**

- Asked how to make the sidecar image work on both arm64 and amd64 chips.
- Questioned why the Dockerfile defaulted the profile to `stylens`.
- Said the app using the sidecar should set its own profile — and that the
  default should not be stylens at all.
- Agreed stylens should not live inside the library; it is a separate app concern.
- Asked again how to finish the arm64/amd64 build, and which releases the
  recent changes require.

**Outcome (code / product)**

- Default profile moved from `stylens` → `default` (generic library priors).
- Docs / AGENTS guidance updated so app-specific domains stay in a domains file.
- Sidecar publish workflow enhanced for multi-architecture (amd64 + arm64).
- Related commits landed on `main` (e.g. default profile + multi-arch publish support).

### 2026-10-19 — “Where are we and how do we proceed?”

**Dialogue**

- Asked for an honest take on the repo: current state and how to proceed.

**Outcome (judgment)**

- Past “does the idea work?” — into “make it survive real apps.”
- Bottleneck is productization and dogfood, not more control-law papers.
- Suggested focus: dogfood with domain files, small keyword/classifier fixes,
  not new infrastructure.

### Earlier arc (through ~Oct 2026) — Tenant, domains, browser, releases

**What we were solving**

- “User” in the API was confusing when the bucket is really an app/tenant.
- App-specific fact kinds (e.g. stylens / relay) should not be compiled into VoltMem.
- Operators needed a browser to inspect and clear memories safely.

**Outcome**

- Plan “Tenant semantics + domain profiles” completed (tenants API, optional
  domain on write, `VOLTMEM_DOMAINS_FILE`, UI labels, dogfood handoff notes).
- Memory browser: graph, pagination, clear tenant / clear domain.
- Release plumbing documented (PyPI / npm / sidecar tags; trusted publishing).

---

## Template for the next entry

Copy this block to the **top** of the chronological log after each session:

```markdown
### YYYY-MM-DD — short title

**Dialogue**

- …

**Outcome**

- …

**Open questions / next**

- …
```

---

## Quick glossary

| Word | Plain meaning |
|---|---|
| Tenant | Whose memory bucket (one app user, or one app environment) |
| Domain | What kind of fact (preference, location, etc.) and how sticky it is |
| Sidecar | Small HTTP service wrapping VoltMem so non-Python apps can talk to it |
| Profile / domains file | How the sidecar knows which fact kinds exist — apps own this, not the library |
