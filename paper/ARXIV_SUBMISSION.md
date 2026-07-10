# arXiv submission checklist — Volatility-Adjusted Memory Protection

Pre-submission package for the reconciled VoltMem research draft. Submit manually at [arxiv.org](https://arxiv.org/).

## Artifacts

| Item | Path / URL |
|------|------------|
| PDF (rebuild before upload) | `paper/volatility_ewc_portfolio.pdf` |
| Editable source | `paper/build_paper.py` |
| Findings log | `paper/findings.md` |
| Reproduction guide | `docs/RESEARCH.md` |
| Code | https://github.com/Rouche01/voltmem |
| PyPI package | https://pypi.org/project/voltmem/ (v0.2.0) |

## Rebuild PDF

```bash
cd /path/to/voltmem
uv pip install reportlab --python .venv/bin/python   # once
.venv/bin/python paper/build_paper.py
```

## Suggested metadata

**Title:** Volatility-Adjusted Memory Protection: A Causal Control Knob for Continual Learning and LLM Agent Memory

**Abstract (draft):**

Standard continual-learning methods such as Elastic Weight Consolidation apply a uniform protection strength to all past knowledge, forcing a stability–plasticity tradeoff. We scale protection per domain by measured volatility *before* any update, and validate claims with negative controls (shuffling or inverting the domain→volatility map must degrade performance monotonically). On a synthetic benchmark with disjoint task inputs, both retention and adaptation improve simultaneously. On Split-MNIST the honest result is subtler: volatility weighting is not a free-lunch Pareto win but a **causal control knob** that steers the tradeoff (REAL > SHUFFLE > SWAP). The same principle powers VoltMem, an open-source LLM memory library where volatility-aware write and retrieval policies beat flat and inverted controls on scripted multi-turn scenarios, a noisy retrieval haystack (0% stale@1 vs 20% cosine-only), and a 3/3 current-truth wedge vs Mem0. LongMemEval-S at n=30 does not beat plain cosine overall (70.0% vs 73.3% answer@5); gains concentrate on knowledge-update question types. We report limitations, embedding-backend variance, and failed early wins that did not survive sabotage controls.

**Categories:** cs.LG (primary), cs.AI

**Comments:** 8 pages. Code and reproduction scripts at GitHub; pip install voltmem.

## Pre-upload checklist

- [ ] Rebuild PDF (`python paper/build_paper.py`)
- [ ] Confirm haystack table matches latest `retrieval_haystack_bench.py` run
- [x] Paste scaled LongMemEval-S (n=60) table — real 0.700 ties cosine; chunk-calibrated ingest
- [ ] Verify GitHub repo is public and README badges resolve
- [ ] Confirm PyPI 0.2.0 is published (or note 0.1.0 in abstract if not yet)
- [ ] Author affiliation / ORCID if desired
- [ ] No API keys or `.env` in any uploaded file

## After acceptance

- [ ] Add arXiv ID badge to `README.md` and `docs/RESEARCH.md`
- [ ] Link PDF from portfolio / LinkedIn
- [ ] Tag release `v0.2.0` on GitHub if not already

## Honest framing (do not oversell)

- Lead with **causal control** (REAL > flat > swap), not overall LongMemEval SOTA.
- Slot linking + Mem0 3/3 = **application case study**, not benchmark leaderboard.
- Vector index = engineering acceleration; volatility re-rank unchanged.
- Continual learning = second validation domain; agent memory = front door.
