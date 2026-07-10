# VoltMem — Research

Empirical claims, negative controls, reproduction commands, and limitations.
For the product quickstart see [README.md](../README.md).

**Paper:** [PDF (pre-arXiv draft)](../paper/volatility_ewc_portfolio.pdf) · [arXiv submission guide](../paper/ARXIV_SUBMISSION.md)

**Package:** [PyPI voltmem](https://pypi.org/project/voltmem/) · [GitHub](https://github.com/Rouche01/voltmem)

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
| Agent retrieval (noisy haystack) | **0%** stale@1 vs **20%** cosine; sep **0.153** | PASS (real > swap) |
| LongMemEval-S (n=30, stratified) | **70.0%** answer@5 (real); cosine **73.3%** | swap **60.0%** < real; gains on `knowledge-update` |
| LongMemEval-S (n=60, scaled) | real **70.0%** ties cosine; flat **71.7%** | swap **66.7%** < real; chunk-calibrated ingest |
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
7. **LongMemEval-S:** at n=30 (5 per type), voltmem_real **0.700** vs
   similarity_only **0.733** answer@5 — **does not beat cosine overall**. At
   n=60 with **chunk-calibrated ingest** (user → `stated_preference`, assistant →
   `opinion`), real **0.700** ties cosine; flat **0.717** leads slightly; swap
   **0.667** < real. Preference type: **0.700** (was 0.300 with heuristic domains).
8. **Retrieval haystack:** 0% stale@1 vs 20% for cosine-only; separation 0.153
   (real > swap 0.111). current@5: 100% vs 80%.
9. **Slot-aware linking:** paraphrased turns reach `observe()` via domain-slot
   fallback; 3/3 current-truth vs real Mem0 on scripted scenarios (case study).
10. **Vector index (v0.2):** SQLite ANN + volatility re-rank; parity with full
    scan (`tests/test_vector_index.py`) — engineering, not a separate causal claim.

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
python experiments/longmemeval.py --split s --per-type 10   # scaled n≈60

# retrieval haystack + Mem0 wedge
python experiments/retrieval_haystack_bench.py
python experiments/mem0_side_by_side.py

# vector index
python tests/test_vector_index.py

# continual learning (causal regime)
python experiments/ewc_volatility_v3_mnist.py --sabotage --hidden 128 --lam 300 --tasks 16 --runs 6

# unit tests
python tests/test_voltmem.py
python tests/test_client.py
```

Always run `--sabotage` on CL experiments before trusting raw accuracy gains.

### LongMemEval-S results (scaled, n=30)

| system | answer@5 | evidence@5 |
|---|---|---|
| voltmem_real | 0.700 | 0.800 |
| similarity_only | 0.733 | 0.900 |
| voltmem_flat | 0.733 | 0.900 |
| voltmem_swap | 0.600 | 0.767 |

Strongest type-level signal: `knowledge-update` (real 0.600 > flat 0.400 > swap 0.200).
Overall answer@5 does not beat plain cosine at n=30 or n=60 — report honestly.

### LongMemEval-S results (scaled, n=60)

Run: `python experiments/longmemeval.py --split s --per-type 10` (2026-07-10;
`all-MiniLM-L6-v2`). **Chunk domain profile:** user turns → `stated_preference`,
assistant → `opinion` (session-scoped; avoids mislabeling chat logs as
`core_preference` / `biographical`).

| system | answer@5 | evidence@5 |
|--------|----------|------------|
| voltmem_flat | **0.717** | **0.850** |
| voltmem_real | **0.700** | 0.833 |
| similarity_only | **0.700** | **0.850** |
| voltmem_swap | 0.667 | 0.817 |

Per-type answer hit rate:

| type | real | flat | swap | similarity |
|------|------|------|------|------------|
| single-session-preference | **0.700** | 0.700 | 0.600 | 0.700 |
| single-session-assistant | **0.900** | 0.900 | 0.900 | 0.900 |
| single-session-user | 0.800 | 0.900 | 0.700 | 0.900 |
| knowledge-update | 0.600 | 0.600 | 0.500 | 0.600 |
| temporal-reasoning | 0.600 | 0.600 | 0.700 | 0.600 |
| multi-session | 0.600 | 0.600 | 0.600 | 0.500 |

**Read:** chunk calibration recovers preference retrieval (0.300 → 0.700 vs
heuristic ingest); real ties cosine overall. Flat still +1.7pp — not SOTA, but
validates retrieval-layer parity with documented ingestion assumptions.

---

## Limitations

- Not a free-lunch accuracy method; a control knob / tuning-robustness tool.
- Regime-dependent benefits; strongest in under-protected large models (CL).
- Domain priors are manual; automatic volatility from gradient conflict is open work.
- EMA can drift under sustained contradiction (prior-anchored update is planned).
- Linking thresholds vary by embedding backend; pin or calibrate in production.
- LongMemEval overall does not beat flat at n=60; real ties cosine (0.700) with
  chunk-calibrated ingest; report per-type signals honestly.
- Vector index (v0.2) accelerates retrieval; volatility re-rank is unchanged.

---

## Origin

The model grew out of work on stale social calibrations — when to trust an old habit
vs. audit against present context — and maps onto the stability–plasticity problem in
continual learning.
