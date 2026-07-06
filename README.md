# VoltMem

**A volatility-adjusted persistent memory layer for LLM applications.**

Most LLM memory systems treat all stored knowledge the same — protecting a
user's current mood with the same strength as their core personality, or
holding a three-year-old job title as tightly as a preference stated this
session. VoltMem applies a different protection strength to each memory based
on how fast that memory's domain actually changes, caught before it goes stale.

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

```python
from voltmem import MemoryLayer

mem = MemoryLayer("my_app.db")

# Write known facts at startup
mem.write("User prefers direct, concise responses", domain="core_preference")
mem.write("User is currently job hunting",          domain="current_project")

# Observe new information — layer decides whether to update or retain
result = mem.observe(
    content="User accepted a job offer, no longer job hunting",
    domain="current_project",
    mismatch_magnitude=0.9,       # strongly contradicts stored memory
    source="explicit_statement",
)
print(result.action)   # "audited" — volatile domain + high mismatch → updated

# Retrieve relevant memories to inject into LLM system prompt
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
    memory.py         MemoryLayer — primary pluggable interface
    scoring.py        core equations: E_t, θ_t, staleness, retrieval_score
    domains.py        domain volatility priors + MemoryItem type
    store.py          SQLite persistence layer
  experiments/
    ewc_baseline_v1.py      first attempt — null result, kept for transparency
    ewc_volatility_v2.py    fixed version — main experimental result
  tests/
    test_voltmem.py   16 tests covering all equation behaviours
  demo.py             end-to-end demo plugged into an LLM assistant loop
```

---

## Experimental result

On a synthetic continual learning benchmark (12 tasks, 8 runs), volatility-
adjusted consolidation improved both sides of the stability-plasticity tradeoff
simultaneously over a uniformly-tuned EWC baseline:

| Metric | Baseline EWC | VoltMem |
|---|---|---|
| Early-task stable retention | 0.935 | **0.962** (+2.7pp) |
| Late-task volatile adaptation | 0.576 | **0.628** (+5.2pp) |

The experiment went through two iterations — the first produced a null result
due to measurement bugs (documented in `experiments/ewc_baseline_v1.py`).
The fix and the diagnosis are both part of the record.

Full writeup: [arXiv — coming soon]

---

## Limitations and next steps

- **Synthetic benchmark only.** Split-MNIST / Split-CIFAR validation is pending.
- **Keyword-based retrieval.** The `_similarity()` method in `memory.py` uses
  simple word overlap. Replace with cosine similarity over embeddings
  (OpenAI, Cohere, or a local model) for production use.
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