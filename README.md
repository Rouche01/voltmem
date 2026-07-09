# VoltMem

**Volatility-aware memory for LLM agents — protect stable facts, let volatile ones move.**

Most agent memory treats everything the same: your hometown and today's mood get
equal weight. That forces a bad tradeoff — either you **go stale** on fast-changing
facts, or you **get corrupted** when a confident-but-wrong update overwrites
something durable. VoltMem scales protection and retrieval freshness by **how fast
each kind of fact actually changes**.

---

## At a glance

| Capability | Best result | Control |
|---|---|---|
| **Update policy** (current truth under noise) | balanced **0.597** (`llm_memory_bench`) | real > flat > swap |
| **Retrieval** (current fact in a noisy haystack) | **0%** stale@1 vs **60%** cosine-only | `retrieval_haystack_bench` PASS |
| **Continual learning** (Split-MNIST) | causal stability knob (+0.055 REAL−SWAP) | `--sabotage` passes |
| **LongMemEval oracle** (public benchmark) | ~72–78% all systems tie | harness ready; easy split |

**What we claim:** a validated **control knob** on the stability–plasticity tradeoff,
for both neural nets (EWC) and agent memory — backed by **negative controls** (shuffle
/ swap), not just raw accuracy.

**What we don't claim:** a universal accuracy booster, free parameter savings, or
beating plain cosine on tiny haystacks (LongMemEval oracle).

---

## Try it in 60 seconds

```bash
# batteries-included API (needs Ollama + nomic-embed-text, or falls back offline)
.venv/bin/python examples/quickstart_batteries.py

# felt win: VoltMem 3/3 vs naive policies 1/3
.venv/bin/python experiments/memory_demo.py

# quantitative: update policy + retrieval haystack
.venv/bin/python experiments/llm_memory_bench.py
.venv/bin/python experiments/retrieval_haystack_bench.py
```

```python
from voltmem import MemoryLayer, EmbeddingSimilarity

mem = MemoryLayer("app.db", similarity_fn=EmbeddingSimilarity())
mem.remember("I live in Berlin")
mem.remember("Actually I moved to Paris")   # updates location, keeps stable prefs safe
print(mem.recall("where does the user live?"))
```

### LangChain adapter

VoltMem plugs into legacy LangChain chains via `VoltMemMemory` — the same
`load_memory_variables` / `save_context` hooks `ConversationChain` expects.
`session_id` maps to VoltMem's per-user namespace.

```bash
pip install -r requirements-integrations.txt
.venv/bin/python examples/langchain_agent.py
```

```python
from voltmem import EmbeddingSimilarity
from voltmem.integrations.langchain import VoltMemMemory

memory = VoltMemMemory(
    session_id="user-42",
    db_path="app.db",
    similarity_fn=EmbeddingSimilarity(),
)
vars = memory.load_memory_variables({"input": "Where do I live?"})
memory.save_context({"input": "I moved to Paris"}, {"output": "Noted."})
```

**Examples:** `examples/quickstart_batteries.py` · `examples/multi_tenant.py` ·
`examples/langchain_agent.py` · `experiments/memory_demo.py`

---

## The core idea

Standard continual learning approaches (like EWC) protect old knowledge with
one fixed global strength, forcing a tradeoff between remembering old things
and adapting to new ones. VoltMem scales that protection per memory item by
an independently-measured domain volatility estimate:

$$
w_d = \frac{1}{V_d^{\gamma}}
$$

High volatility (current task, emotional context) → low protection weight →
memory updates readily. Low volatility (personality trait, core preference) →
high protection weight → memory resists overwriting.

The escalation decision (audit vs. retrieve-as-is) is governed by:

$$
E_t = \left[\frac{M_t \cdot R_t}{C^{\alpha}}\right] \cdot V_d \cdot G_t
$$

$$
\theta_t = \theta_0 \cdot \frac{1}{V_d} \cdot L_t
$$

Where $M_t$ is mismatch magnitude, $R_t$ is source reliability, $C$ is
repetition count, $V_d$ is domain volatility, $G_t$ is goal-attainment
delta, and $L_t$ is cognitive load. Escalate (audit + update) if
$E_t > \theta_t$, otherwise retrieve directly from existing calibration.

Retrieval ranking also down-weights stale volatile memories:

$$
\text{staleness} = 1 - e^{-V_d \cdot \text{age}_{\text{days}}}
$$

$$
\text{score} = \text{semantic similarity} \cdot \left(1 - V_d \cdot \text{staleness}\right)
$$

Full derivation and experimental results: [arXiv link — coming soon]

---

## Quickstart

### Batteries-included (recommended)

Just hand it raw statements — it infers the domain, finds related memories, and
lets the volatility engine decide whether to update or keep them. No manual
`domain` / `mismatch` bookkeeping.

```python
from voltmem import MemoryLayer, EmbeddingSimilarity

# EmbeddingSimilarity auto-detects a backend: sentence-transformers → Ollama →
# offline hashing fallback. Omit it to use the built-in keyword scorer.
mem = MemoryLayer("my_app.db", similarity_fn=EmbeddingSimilarity())

mem.remember("I live in Berlin")                 # → stored as 'location'
mem.remember("I prefer concise, direct answers") # → stored as 'core_preference'
mem.remember("Actually I moved to Paris")        # → updates the location memory
mem.remember("I really like short replies")      # → protected: stable, kept

for line in mem.recall("where does the user live?", top_k=1):
    print(line)     # -> Actually I moved to Paris
```

### Multi-tenant (one database, many users)

Scope memories per user/tenant with `namespace` or `for_user()`. All views share
one SQLite file and connection — reads and writes are isolated per namespace:

```python
from voltmem import MemoryLayer

mem = MemoryLayer("app.db")
alice = mem.for_user("alice")
bob   = mem.for_user("bob")

alice.remember("I live in Berlin")
bob.remember("I live in Paris")

alice.recall("where does the user live?")  # -> Berlin only
bob.recall("where does the user live?")    # -> Paris only
```

You can also pass `namespace=` directly: `MemoryLayer("app.db", namespace="alice")`.
Existing databases without a namespace column are migrated automatically on open.

For higher-quality domain/contradiction inference, pass an LLM extractor:

```python
from voltmem import MemoryLayer, LLMExtractor
mem = MemoryLayer("my_app.db", extractor=LLMExtractor())  # uses local Ollama
```

### Low-level control

If you'd rather supply the signals yourself (e.g. you already compute mismatch):

```python
from voltmem import MemoryLayer

mem = MemoryLayer("my_app.db")
mem.write("User is currently job hunting", domain="current_project")

result = mem.observe(
    content="User accepted a job offer, no longer job hunting",
    domain="current_project",
    mismatch_magnitude=0.9,       # strongly contradicts stored memory
    source="explicit_statement",
)
print(result.action)   # "audited" — volatile domain + high mismatch → updated

results = mem.retrieve("career and work context", top_k=4)
for item, score in zip(results.items, results.scores):
    print(f"[{score:.2f}] {item.content}")
```

---

## Built-in domain volatility priors

| Domain | Volatility | Protection |
|---|---|---|
| `personality_trait` | 0.05 | Very high — rarely changes |
| `core_preference` | 0.08 | Very high |
| `biographical` | 0.10 | High |
| `professional_context` | 0.30 | Medium |
| `current_project` | 0.55 | Low — changes often |
| `emotional_context` | 0.80 | Very low — changes fast |
| `current_task` | 0.90 | Minimal |

Custom domains can be added to `voltmem/domains.py`.

---

## Plugging into any LLM system

VoltMem has no LLM-specific dependencies. The integration pattern is always
the same three steps:

```python
# 1. At the start of each turn, retrieve relevant memories
results = mem.retrieve(user_message, top_k=4)
memory_context = "\n".join(f"- {item.content}" for item in results.items)

# 2. Inject into your LLM system prompt
system_prompt = f"What you know about this user:\n{memory_context}"

# 3. After the turn, observe any new facts extracted from the conversation
mem.observe(new_fact, domain="current_project", mismatch_magnitude=0.7)
```

See `demo.py` for a complete worked example.

---

## Repository structure

```
voltmem/
  voltmem/
    __init__.py       clean public API
    memory.py         MemoryLayer — remember() / recall() / multi-tenant
    scoring.py        core equations: E_t, θ_t, staleness, retrieval_score
    domains.py        domain volatility priors + MemoryItem type
    store.py          SQLite persistence + namespace migration
    embeddings.py     pluggable semantic similarity (ST / Ollama / hashing)
    extract.py        domain + mismatch inference (heuristic / LLM)
    integrations/
      langchain.py    VoltMemMemory — load/save hooks for LangChain chains
  examples/
    quickstart_batteries.py   remember() / recall() in ~30 lines
    multi_tenant.py           one DB, many users via for_user()
    langchain_agent.py        VoltMemMemory demo (no API key)
  experiments/
    llm_memory_bench.py       update policy vs naive baselines
    retrieval_haystack_bench.py  freshness ranking in a noisy haystack
    longmemeval.py            public LongMemEval oracle (HF stream)
    memory_demo.py            readable 3/3 vs 1/3 transcript
    voltmem_eval.py           unit-style end-to-end (real / flat / swap)
    ewc_volatility_v*.py      continual-learning EWC experiments
  tests/
    test_voltmem.py   23 tests
```

---

## What we actually claim (and what we don't)

The honest one-line summary: **volatility weighting is a validated *control knob*
on the stability–plasticity tradeoff — not a free-lunch accuracy booster.** Every
claim below is paired with the negative control that backs it and the limit that
bounds it.

### Claim 1 — On a synthetic benchmark, both axes improve at once

On the 2D-blob continual-learning benchmark (`ewc_volatility_v2.py`), volatility-
adjusted consolidation beat a uniformly-tuned EWC baseline on *both* sides of the
tradeoff at the same time:

| Metric | Baseline EWC | VoltMem |
|---|---|---|
| Early-task stable retention | 0.935 | **0.962** (+2.7pp) |
| Late-task volatile adaptation | 0.576 | **0.628** (+5.2pp) |

**Limit:** this "improve both" result is *regime-specific*. It holds when stable
and volatile tasks use disjoint input regions over a shared trunk, so protecting
stable features does not directly fight volatile inputs. It does **not**
generalize to a universal Pareto win (see Claim 2).

### Claim 2 — On real data (Split-MNIST) it is a causal control knob

`ewc_volatility_v3_mnist.py` moves the idea to real MNIST images with genuine
feature interference. Here the effect is **real but modest and regime-dependent**:

- A big apparent retention gain at high capacity (hidden=512, +7pp) is an
  **artifact** — it survives shuffling/inverting the domain→volatility pairing,
  so it's generic non-uniform regularization, not the volatility signal.
- In a genuinely-competing regime (hidden=128, λ≈300, 16 tasks, 6 runs), the
  effect **passes the negative control**. By stability index (retention − adaptation):

  | Condition | baseline | REAL | SHUFFLE | SWAP |
  |---|---|---|---|---|
  | stability index | 0.317 | **0.460** | 0.457 | 0.405 |

  `REAL − SWAP = +0.055` in the predicted direction: inverting which domains are
  "volatile" reliably tilts the model toward plasticity. The volatility signal
  *causally steers* the tradeoff.

**Limit:** it is not a Pareto win on real data. It trades adaptation for
retention along a frontier a well-tuned uniform baseline could also reach.

### Claim 3 — Its practical value is robustness to under-tuned protection

`capacity_efficiency.py` tested whether volatility weighting saves parameters
(lets a small model match a bigger baseline). That hypothesis was **falsified**.
The favorable, causal benefit instead shows up in **under-protected large models**,
where uniform EWC catastrophically forgets and volatility rescues the stable
domain (hidden=256, λ=300, 4 runs):

- retention: baseline 0.877 → **0.954** (+7.7pp), adaptation only −2.9pp
- passes `--sabotage` (shuffle/swap keep only +1.3pp)

**Takeaway:** volatility weighting auto-allocates a fixed protection budget to
stable knowledge, so you don't have to hand-tune λ per capacity. A well-tuned
uniform baseline can match it; volatility mainly reduces the *need* to tune.

### Claim 4 — The library's memory behavior is driven by volatility, causally

`voltmem_eval.py` runs the whole library end-to-end under three volatility
profiles: **real** (the priors), **flat** (all domains equal), and **swap**
(priors inverted). If behavior were an accident of thresholds, the profile
wouldn't matter. It does:

| Capability | real | flat | swap |
|---|---|---|---|
| Selective updating (accuracy) | **100%** | 83% | 50% |
| Freshness-aware retrieval (separation) | **+0.589** | +0.202 | −0.267 |

The `swap` control degrading *below* flat is the causal evidence: flipping which
domains are volatile flips the behavior in the wrong direction.

### Claim 5 — Volatility estimates update in proportion to source trust

`ema_erosion_test.py` covers the multi-turn robustness of `observe()`. The EMA
learning rate is scaled by source reliability, so a low-trust observation barely
moves a stable memory:

- one weak-inference hit moves volatility **2.5× less** than the old logic
- reliable-source updates are unchanged (backward compatible)
- a sustained weak stream crosses V≥0.5 at **turn 8** vs the old logic's **turn 2**,
  and stable content is never wrongly overwritten

**Limit (honest):** reliability scales the *step size*, not the EMA's *fixed
point*. Persistent, repeated contradictions will still raise volatility over many
turns — which is arguably correct (sustained conflict *is* evidence of
volatility), but means this is not a hard "never drifts" guarantee.

### Claim 6 — As an LLM-agent memory, it beats naive policies on *both* failure modes

`llm_memory_bench.py` simulates a user over many noisy sessions where facts drift
at domain-appropriate rates, and asks "what is the user's *current* X?" A memory
layer can fail two ways at once: go **stale** on volatile facts, or get
**corrupted** by a confident-but-wrong observation on stable facts. Naive policies
are forced onto one side; VoltMem escapes the tradeoff (24 sessions × 20 runs,
accuracy):

| system | overall | stable | volatile | balanced* |
|---|---|---|---|---|
| **voltmem (real priors)** | 0.522 | 0.578 | 0.617 | **0.597** |
| voltmem (flat) | 0.361 | 0.578 | 0.188 | 0.283 |
| voltmem (swap) | 0.347 | 0.407 | 0.194 | 0.263 |
| always-overwrite | 0.573 | 0.360 | 0.767 | 0.490 |
| never-overwrite | 0.361 | 0.578 | 0.188 | 0.283 |
| reliability-threshold | 0.579 | 0.437 | 0.548 | 0.486 |

\*balanced = harmonic mean of stable & volatile (rewards being good at *both*).

VoltMem-real is the only policy strong on both axes; each naive policy
catastrophically fails one (always-overwrite corrupts stable facts;
never-overwrite goes stale on volatile facts; a pure reliability heuristic is
mediocre on both — it wrongly adopts confident blips on stable facts and wrongly
ignores weak-but-true updates on volatile ones). The **real > flat > swap**
ordering is the causal control. `memory_demo.py` shows a concrete 3/3 vs 1/3
transcript. Retrieval uses a **pluggable embedding backend**
(`voltmem/embeddings.py`: sentence-transformers → Ollama → offline hashing
fallback), so all systems share the same similarity and the comparison isolates
the volatility logic.

**Design fix found while building this:** the demo exposed a real bug — `observe()`
updated the volatility EMA *before* the escalation decision, so one confident blip
inflated measured volatility and lowered its own threshold, overwriting a stable
fact on the first hit. Fixed by deciding against the volatility known *before* the
observation, then folding the observation into the EMA. This strengthened the
`voltmem_eval` causal separation (swap 25% vs 50% previously).

### Claim 7 — Public benchmark harness (LongMemEval oracle, retrieval mode)

`experiments/longmemeval.py` streams the official **LongMemEval** dataset
(HuggingFace: `xiaowu0162/longmemeval-cleaned`, oracle split) and scores memory
**retrieval** — whether top-k recalled chunks contain the annotated answer or
evidence turns. No LLM API required.

On the **oracle split** (evidence-only sessions; stratified sample), all systems
score similarly (~72–78% answer@5) — staleness weighting has little room to help
when the haystack is tiny. **Volatility does not beat plain cosine on this easy
split**; that is reported honestly. The harness is still the credibility asset:
official data, reproducible commands, negative controls (real / flat / swap /
similarity-only). **LongMemEval-S** (full ~115k-token haystack) is the next step
where freshness ranking should matter.

```bash
.venv/bin/python experiments/longmemeval.py --quick
.venv/bin/python experiments/longmemeval.py --per-type 5
```

### Claim 8 — Retrieval haystack: freshness ranking beats cosine in noise

`retrieval_haystack_bench.py` stores dozens of memories per query slot, including
**semantically similar stale volatile decoys** (90-day-old project names, moods).
All systems share the same chunks; only retrieval ranking differs (30 runs × 5
slots, 8 decoys, real Ollama embeddings):

| system | current@5 | stale@1 ↓ | separation |
|---|---|---|---|
| **voltmem_real** | **1.000** | **0.000** | **+0.187** |
| similarity_only | 0.800 | 0.600 | −0.033 |

Plain cosine retrieval ranks a **stale volatile decoy #1 in 60%** of cases;
voltmem_real **never** does, and always places the current fact in top-5. This is
the load-bearing retrieval result that LongMemEval oracle (tiny haystack) could
not show.

```bash
.venv/bin/python experiments/retrieval_haystack_bench.py
```

### How we know it's causal — the negative control

Every non-synthetic claim above is gated on a **sabotage / negative control**:
we re-run the exact same pipeline with the domain→volatility mapping (a) shuffled
and (b) inverted (`swap`). A real effect must degrade monotonically
(REAL > SHUFFLE > SWAP). Several early "wins" *failed* this test and were
discarded as artifacts — that filter is why the surviving claims are trustworthy.
Always run `--sabotage` before trusting a raw accuracy gain.

## Writeup 
- [PDF (pre-arXiv draft)](paper/volatility_ewc_portfolio.pdf)
- [arXiv — coming soon]

---

## Limitations and next steps

- **Not a free-lunch accuracy method.** On real data it is a control knob /
  tuning-robustness tool, not a universal Pareto improvement (see Claims 2–3).
- **Regime-dependent.** The "improve both axes" result needs disjoint task inputs
  over a shared trunk; benefits are strongest in under-protected large models.
- **Small-scale benchmarks only.** Validated on 2D blobs and Split-MNIST;
  Split-CIFAR / larger nets and longer task streams are pending.
- **EMA fixed-point drift.** Reliability weighting slows erosion but does not
  anchor the estimate to the domain prior; a prior-anchored update (relaxing only
  as *reliable* evidence accumulates) is the natural next step.
- **Retrieval quality depends on the embedding backend.** `voltmem/embeddings.py`
  provides pluggable semantic similarity (sentence-transformers → Ollama →
  offline hashing fallback); the default keyword scorer is still used unless you
  pass `similarity_fn=EmbeddingSimilarity(...)`. The offline fallback is
  bag-of-tokens (not semantic) — install a real backend for production.
- **Domain partitioning is manual.** Automatic volatility detection from
  gradient conflict signals is the main open research direction.
- **No comparison to replay-based methods** yet (GEM, A-GEM, experience replay).

---

## Origin

This library grew out of a longer conversation about how human minds handle
stale social calibrations — when to trust an old habit vs. audit it against
present context. The computational model maps almost directly onto the
stability-plasticity problem in continual learning. The philosophical
derivation is in the arXiv paper.

---

## Requirements

```
torch
numpy
```

No LLM SDK required. SQLite is part of Python's standard library.

---

## License

MIT