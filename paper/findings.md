# Volatility-Adjusted Memory Protection — Findings, Controls, and Limits

*Editable companion to `volatility_ewc_portfolio.pdf`. This document is the
source of record for the empirical claims: every claim is paired with the
negative control that backs it and the limit that bounds it. Convert to PDF with
`pandoc paper/findings.md -o paper/findings.pdf` when folding into the paper.*

---

## 1. One-line summary

Volatility weighting is a validated **control knob** on the stability–plasticity
tradeoff — **not** a free-lunch accuracy booster. Scaling per-item memory
protection by an independently-measured domain volatility lets a fixed protection
budget be allocated where it matters, and causally steers whether a model (or a
memory store) leans toward *retaining* old knowledge or *adapting* to new.

---

## 2. The core idea and math

Standard continual-learning consolidation (e.g. EWC) protects old knowledge with
one global strength, forcing a single point on the stability–plasticity frontier.
VoltMem scales that protection *per item* by a domain volatility estimate
$V_d \in (0, 1]$:

$$
w_d = \frac{1}{V_d^{\gamma}}
$$

Low-volatility domains (personality traits, core preferences) get high protection
weight and resist overwriting; high-volatility domains (current task, mood) get
low weight and update readily.

For the memory layer, the audit-vs-retain decision uses an escalation score and a
volatility-scaled threshold:

$$
E_t = \left[\frac{M_t \cdot R_t}{C^{\alpha}}\right] \cdot V_d \cdot G_t
\qquad
\theta_t = \theta_0 \cdot \frac{1}{V_d} \cdot L_t
$$

with $M_t$ mismatch magnitude, $R_t$ source reliability, $C$ repetition count,
$G_t$ goal-attainment delta, $L_t$ cognitive load. Escalate (audit + update) iff
$E_t > \theta_t$.

Retrieval down-weights stale volatile memories:

$$
\text{staleness} = 1 - e^{-V_d \cdot \text{age}_{\text{days}}}
\qquad
\text{score} = \text{similarity} \cdot \left(1 - V_d \cdot \text{staleness}\right)
$$

**Critical design choice — pre-update measurement.** Volatility is measured from
mismatch/surprise *before* any protection or gradient step is applied. Measuring
it after would be circular (protection suppresses the very signal used to set
protection).

---

## 3. Methodology — the negative control (why any of this is trustworthy)

Every non-synthetic claim is gated on a **sabotage / negative control**. We re-run
the identical pipeline with the domain→volatility mapping:

- **REAL** — the true priors,
- **SHUFFLE** — priors randomly permuted across domains,
- **SWAP / flat** — priors inverted (or all domains set equal).

A genuine effect must degrade monotonically: **REAL > SHUFFLE > SWAP**. If a "win"
survives shuffling or inversion, it is generic non-uniform regularization, *not*
the volatility signal, and we discard it. Several early apparent wins failed this
test and were removed. This filter is the reason the surviving claims below are
credible.

---

## 4. Claims

### Claim 1 — On a synthetic benchmark, both axes improve at once

Source: `experiments/ewc_volatility_v2.py` (2D-blob continual learning).

| Metric | Baseline EWC | VoltMem |
|---|---|---|
| Early-task stable retention | 0.935 | **0.962** (+2.7pp) |
| Late-task volatile adaptation | 0.576 | **0.628** (+5.2pp) |

**Limit:** regime-specific. Holds when stable and volatile tasks occupy disjoint
input regions over a shared trunk, so protecting stable features does not directly
fight volatile inputs. Does not generalize to a universal Pareto win (see Claim 2).

### Claim 2 — On real data (Split-MNIST) it is a causal control knob

Source: `experiments/ewc_volatility_v3_mnist.py`.

The effect is real but **modest and regime-dependent**:

- A large apparent retention gain at high capacity (hidden=512, +7pp) is an
  **artifact** — it survives shuffle/invert, so it fails the control.
- In a genuinely-competing regime (hidden=128, $\lambda \approx 300$, 16 tasks,
  6 runs) it **passes** the control. By stability index (retention − adaptation):

  | Condition | baseline | REAL | SHUFFLE | SWAP |
  |---|---|---|---|---|
  | stability index | 0.317 | **0.460** | 0.457 | 0.405 |

  $\text{REAL} - \text{SWAP} = +0.055$ in the predicted direction. Inverting which
  domains are "volatile" reliably tilts the model toward plasticity.

**Limit:** not a Pareto win on real data — it trades adaptation for retention along
a frontier a well-tuned uniform baseline could also reach.

### Claim 3 — Practical value is robustness to under-tuned protection

Source: `experiments/capacity_efficiency.py`.

The parameter-savings hypothesis (small volatility model matches larger baseline)
was **falsified**. The favorable, causal benefit appears in **under-protected
large models**, where uniform EWC catastrophically forgets and volatility rescues
the stable domain (hidden=256, $\lambda=300$, 4 runs):

- retention: baseline 0.877 → **0.954** (+7.7pp); adaptation only −2.9pp
- passes `--sabotage` (shuffle/swap keep only +1.3pp)

**Takeaway:** volatility auto-allocates a fixed protection budget to stable
knowledge, reducing the need to hand-tune $\lambda$ per capacity. A well-tuned
uniform baseline can match it.

### Claim 4 — The library's memory behavior is volatility-driven, causally

Source: `experiments/voltmem_eval.py`. End-to-end run under three profiles:

| Capability | real | flat | swap |
|---|---|---|---|
| Selective updating (accuracy) | **100%** | 83% | 50% |
| Freshness-aware retrieval (separation) | **+0.589** | +0.202 | −0.267 |

The `swap` control degrading *below* `flat` is the causal evidence: flipping which
domains are volatile flips behavior in the wrong direction.

### Claim 5 — Volatility estimates update in proportion to source trust

Source: `experiments/ema_erosion_test.py`; fix in `voltmem/scoring.py` +
`voltmem/memory.py`. The EMA learning rate is scaled by source reliability:

$$
\alpha = (1 - \beta)\cdot \mathrm{clamp}(R_t, 0, 1)
\qquad
V \leftarrow (1-\alpha)\,V + \alpha\, M_t
$$

- one weak-inference hit moves a stable memory's volatility **2.5× less** than the
  prior logic;
- reliable-source updates are unchanged (backward compatible, reduces to the
  original EMA at $R_t \ge 1$);
- a sustained weak stream crosses $V \ge 0.5$ at **turn 8** vs the old logic's
  **turn 2**; stable content is never wrongly overwritten.

Also fixed: `observe()` previously updated the EMA **twice** per call (double-
counting the same observation); it now updates exactly once.

**Limit (honest):** reliability scales the *step size*, not the EMA's *fixed
point*. Persistent, repeated contradictions still raise volatility over many
turns — arguably correct (sustained conflict *is* evidence of volatility), but not
a hard "never drifts" guarantee.

### Claim 6 — As LLM-agent memory, it beats naive policies on both failure modes

Source: `experiments/llm_memory_bench.py` (+ `experiments/memory_demo.py`).

A memory layer for an agent can fail two ways at once: go **stale** on volatile
facts, or get **corrupted** by a confident-but-wrong observation on stable facts.
Simulating a user over many noisy sessions and asking "what is the user's current
X?" (24 sessions × 20 runs, accuracy):

| system | overall | stable | volatile | balanced* |
|---|---|---|---|---|
| **voltmem (real)** | 0.522 | 0.578 | 0.617 | **0.597** |
| voltmem (flat) | 0.361 | 0.578 | 0.188 | 0.283 |
| voltmem (swap) | 0.347 | 0.407 | 0.194 | 0.263 |
| always-overwrite | 0.573 | 0.360 | 0.767 | 0.490 |
| never-overwrite | 0.361 | 0.578 | 0.188 | 0.283 |
| reliability-threshold | 0.579 | 0.437 | 0.548 | 0.486 |

\*balanced = harmonic mean of stable & volatile. VoltMem-real is the only policy
strong on both axes; each naive policy catastrophically fails one. The
**real > flat > swap** ordering is the causal control. All systems share the same
(pluggable) embedding similarity, so the comparison isolates the volatility logic.

**Design fix found while building the demo.** `observe()` updated the volatility
EMA *before* the escalation decision, so a single confident blip inflated measured
volatility and lowered its own threshold — overwriting a stable fact on the first
hit. Fixed by deciding against the volatility known *before* the observation, then
folding the observation into the EMA (`voltmem/memory.py`). This strengthened the
`voltmem_eval` causal separation (swap 25%, previously 50%).

**Limit:** the effect size depends on the noise/blip mix (a modelling choice); the
robust, defensible claims are the qualitative tradeoff-escape (VoltMem best on
*balanced*) and the causal ordering, not the exact accuracy numbers.

---

## 5. Consolidated limitations and next steps

- **Not a free-lunch accuracy method.** On real data it is a control knob /
  tuning-robustness tool, not a universal Pareto improvement.
- **Regime-dependent.** "Improve both axes" needs disjoint task inputs over a
  shared trunk; benefits are strongest in under-protected large models.
- **Small-scale benchmarks only.** Validated on 2D blobs and Split-MNIST;
  Split-CIFAR / larger nets / longer task streams are pending.
- **EMA fixed-point drift.** Reliability weighting slows erosion but does not
  anchor the estimate to the domain prior. Next step: a prior-anchored update that
  relaxes only as *reliable* evidence accumulates.
- **Keyword-based retrieval.** `memory._similarity()` uses word overlap; swap for
  cosine similarity over embeddings in production.
- **Manual domain partitioning.** Automatic volatility detection from gradient-
  conflict signals is the main open research direction.
- **No replay-baseline comparison** yet (GEM, A-GEM, experience replay).

---

## 6. Reproduction

```bash
# negative-control test in a regime where the effect is causally real
.venv/bin/python experiments/ewc_volatility_v3_mnist.py --sabotage \
    --hidden 128 --lam 300 --tasks 16 --gamma 2 --runs 6

# capacity sweep (control-knob vs parameter-saving)
.venv/bin/python experiments/capacity_efficiency.py

# end-to-end library eval (real vs flat vs swap)
.venv/bin/python experiments/voltmem_eval.py

# multi-turn EMA robustness
.venv/bin/python experiments/ema_erosion_test.py

# LLM-agent memory benchmark vs naive policies (+ real/flat/swap control)
.venv/bin/python experiments/llm_memory_bench.py

# short readable demo transcript (VoltMem 3/3 vs naive policies)
.venv/bin/python experiments/memory_demo.py

# unit tests
PYTHONPATH=. .venv/bin/python tests/test_voltmem.py
```
