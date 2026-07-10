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

**Note:** The arXiv abstract lives in `paper/build_paper.py` as the `ABSTRACT`
constant (rendered in the PDF). After editing the abstract, update `ABSTRACT` in
`build_paper.py`, rebuild the PDF, then copy the plain-text version here if needed
for the arXiv web form.

## Suggested metadata

**Title:** Volatility-Adjusted Memory Protection: A Causal Control Knob for Continual Learning and LLM Agent Memory

**Abstract (draft):**

Standard continual-learning methods such as Elastic Weight Consolidation apply a uniform protection strength to all past knowledge, forcing a stability–plasticity tradeoff. We scale protection per domain by measured volatility *before* any update, and validate claims with negative controls: shuffling or inverting the domain→volatility map must degrade performance monotonically (REAL > SHUFFLE > SWAP). On a synthetic benchmark with disjoint task inputs, both retention and adaptation improve simultaneously. On Split-MNIST the honest result is subtler: volatility weighting is not a free-lunch Pareto win but a **causal control knob** that steers the tradeoff. The same principle powers VoltMem, an open-source Python memory library for LLM agents: volatility-aware write and retrieval policies beat flat and inverted controls on scripted multi-turn scenarios (balanced score 0.597, only policy strong on both stable and volatile axes), a noisy retrieval haystack (0% stale@1 vs 20% cosine-only; separation 0.153 vs −0.003), and a 3/3 current-truth case study vs Mem0 on mood, preference, and location updates. On LongMemEval-S (n=60, chunk-calibrated RAG ingest), retrieval answer@5 is 70.0% for VoltMem—tying plain cosine and beating the inverted-volatility control (66.7%)—without claiming public-benchmark SOTA (uniform-volatility ablation reaches 71.7%). We report limitations, embedding-backend variance, failed early wins discarded by sabotage controls, and open gaps in automatic domain discovery.

**Abstract (short, if arXiv limits apply):**

We scale continual-learning protection and LLM memory freshness by per-domain volatility, validated with shuffle/invert negative controls (REAL > SHUFFLE > SWAP). On Split-MNIST this is a causal stability–plasticity knob, not a universal accuracy win. VoltMem, an open-source agent memory library, applies the same idea to writes and retrieval: 3/3 current-truth vs Mem0 on scripted scenarios, 0% stale@1 vs 20% cosine on a haystack bench, and 70% answer@5 on LongMemEval-S (n=60)—tying cosine, not beating it. Code: github.com/Rouche01/voltmem; pip install voltmem.

**Categories:** cs.LG (primary), cs.AI

**Comments:** 8 pages. Code and reproduction scripts at https://github.com/Rouche01/voltmem ; PyPI voltmem 0.2.0.

## Pre-upload checklist

- [x] Rebuild PDF (`python paper/build_paper.py`)
- [ ] Confirm haystack table matches latest `retrieval_haystack_bench.py` run
- [x] Paste scaled LongMemEval-S (n=60) table — real 0.700 ties cosine; chunk-calibrated ingest
- [x] Remove portfolio section; add Introduction, Related Work, Conclusion, References
- [x] Author block: Richard Emate, richard@theemate.com, Independent
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
