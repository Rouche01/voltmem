# VoltMem — Research

Empirical claims, negative controls, reproduction commands, and limitations.
For the product quickstart see [README.md](../README.md).

**Paper:** [PDF (pre-arXiv draft)](../paper/volatility_ewc_portfolio.pdf) · arXiv — coming soon

**Detailed findings companion:** [paper/findings.md](../paper/findings.md)

---

## The claim

VoltMem scales memory protection and retrieval freshness by **how fast each kind of
fact actually changes**. That is a validated **control knob** on the
stability–plasticity tradeoff — not a free-lunch accuracy booster.

Every empirical result is gated on a **negative control**: we re-run the same
pipeline with volatility priors shuffled or inverted (`real > flat > swap`).

### Related work

Uniform EWC, Online-EWC, and EWC-DR apply **one global protection strength**.
VoltMem applies the same stability–plasticity idea at **per-domain granularity**.
Continual-learning experiments (`ewc_volatility_v*.py`) validate the knob on
Split-MNIST; the agent-memory library validates it on update policy and retrieval.

---

## Evidence at a glance

| Capability | Best result | Control |
|---|---|---|
| Agent update policy | balanced **0.597** | real > flat > swap |
| Agent retrieval (noisy haystack) | **0%** stale@1 vs **60%** cosine | PASS |
| LongMemEval-S (pilot, n=12) | **91.7%** vs **75.0%** answer@5 | real > flat > swap |
| Continual learning (Split-MNIST) | +0.055 REAL−SWAP | `--sabotage` passes |

---

## Core equations

Protection weight: \(w_d = 1 / V_d^{\gamma}\)

Escalation: \(E_t = [(M_t \cdot R_t) / C^{\alpha}] \cdot V_d \cdot G_t\), threshold
\(\theta_t = \theta_0 / V_d \cdot L_t\) — audit iff \(E_t > \theta_t\).

Retrieval: \(\text{score} = \text{similarity} \cdot (1 - V_d \cdot \text{staleness})\).

Full derivation: [paper/findings.md §2](../paper/findings.md).

---

## Claims summary

See [paper/findings.md](../paper/findings.md) for full tables and limits. Highlights:

1. **Synthetic CL (2D blobs):** improves stable retention and volatile adaptation together (regime-specific).
2. **Split-MNIST:** causal control knob (+0.055 REAL−SWAP); not a universal Pareto win.
3. **Capacity sweep:** parameter-savings hypothesis falsified; helps under-protected large models.
4. **Library eval:** selective updating 100% (real) vs 50% (swap); retrieval separation +0.589 vs −0.267.
5. **EMA robustness:** reliability-scaled updates; weak blips move volatility 2.5× less.
6. **LLM memory bench:** only policy strong on both stable and volatile axes (balanced 0.597).
7. **LongMemEval-S:** voltmem_real 0.917 vs similarity_only 0.750 answer@5 (12-instance pilot).
8. **Retrieval haystack:** 0% stale@1 vs 60% for cosine-only.

---

## Reproduction

```bash
pip install -e ".[embeddings]"

# end-to-end library (real / flat / swap)
python experiments/voltmem_eval.py

# agent memory vs naive policies
python experiments/llm_memory_bench.py

# retrieval haystack
python experiments/retrieval_haystack_bench.py

# public benchmark
python experiments/longmemeval.py --split s --quick
python experiments/longmemeval.py --split s --per-type 5

# continual learning (causal regime)
python experiments/ewc_volatility_v3_mnist.py --sabotage --hidden 128 --lam 300 --tasks 16 --runs 6

# unit tests
python tests/test_voltmem.py
python tests/test_client.py
```

Always run `--sabotage` on CL experiments before trusting raw accuracy gains.

---

## Limitations

- Not a free-lunch accuracy method; a control knob / tuning-robustness tool.
- Regime-dependent benefits; strongest in under-protected large models (CL).
- Domain priors are manual; automatic volatility from gradient conflict is open work.
- EMA can drift under sustained contradiction (prior-anchored update is planned).
- Retrieval quality depends on embedding backend for production use.

---

## Origin

The model grew out of work on stale social calibrations — when to trust an old habit
vs. audit against present context — and maps onto the stability–plasticity problem in
continual learning.
