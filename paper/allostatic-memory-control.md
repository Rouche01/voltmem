# Prior versus Residual: Homeostatic and Allostatic Control Laws for Agent Memory Updates

Richard Emate  
Independent · richard@theemate.com  
20 August 2026

**Preprint.** doi:10.5281/zenodo.22019047. Empirical companion to Emate (2026), doi:10.5281/zenodo.21962419. VoltMem 0.4.0 control-law work. Not a theory of consciousness. Reproduction: `experiments/allostatic_ablation.py`, `experiments/label_noise_eval.py`, `experiments/end_to_end_eval.py`, `experiments/linking_eval.py`, `experiments/llm_verify_eval.py`. Source log: `docs/allostatic-consciousness-to-voltmem.md`.

---

## Abstract

Agent memory systems that scale overwrite protection by a domain volatility prior \(V_d\) treat two different quantities as one number: a slow belief about how often a *kind* of fact changes, and a fast residual about whether *this* observation was unexpected. We separate them. A **homeostatic** law charges \(V_d\) in both the evidence score and the threshold. An **allostatic** law drops \(V_d\) from the score and scales the threshold by leftover surprise \(r_t = |M_t - \hat{E}_t| / \sigma_t\), the distance from predicted mismatch rather than from stored text. A **composite** gate uses the allostatic law only for an explicit high-mismatch correction or an unexpected residual against a learned \(\hat{E}\); otherwise it keeps the homeostatic insurance.

On scripted probes with oracle domain and mismatch, dropping \(V_d\) from the score recovers explicit recency-shift (entrenched career change) as a cliff at exponent \(p=0\), not a blend. The same drop produces 20% more false updates under the classifier's real error structure, because the double \(V_d\) charge was insurance against a mislabeled stable trait plus weak evidence. Composite matches homeostatic's false-update rate (94.9%, 0.93 false updates / 18) while keeping the recency-shift win. Defining surprise as leftover after anticipation makes a predicted weak stream go quiet; catching that stream is a sleeptime job on time-decayed belief mass, not a live EMA of raw mismatch. Sixteen daily weak mentions supersede overnight; the same sixteen monthly do not.

The overwrite law is only reached after a match. Through `remember(text)`, decision error once routed is about six points; similarity and linking dominate. Topic similarity is non-separable for must-link versus must-not-link pairs. Two-stage recall-then-verify with a conservative local model takes irreversible errors to zero on a combined update+coexist harness. We do not claim a public-benchmark win. We claim a measured decomposition: prior and residual are different jobs; a switch beats a blend; linking sits in front of both laws.

---

## 1. Introduction

A memory layer for LLM agents has to decide, on each write, whether a new sentence updates a stored fact or sits beside it. Facts do not share a timescale. Personality traits should resist a single offhand comment. Locations should move. Jobs sit in between: they change, but not every Tuesday.

VoltMem (Emate, 2026) scales protection by a domain volatility prior \(V_d \in (0,1]\). Low \(V_d\) raises the bar for overwrite; high \(V_d\) lowers it. The same idea, applied to Elastic Weight Consolidation (Kirkpatrick et al., 2017), is a causal control knob on the stability–plasticity tradeoff, not a free-lunch accuracy win: shuffling or inverting the domain\(\to\)volatility map must degrade performance monotonically (REAL \(>\) SHUFFLE \(>\) SWAP). That preprint treats \(V_d\) as one sufficient statistic for how readily a memory should move.

It is not. A channel can be historically volatile and currently well-predicted, or historically stable and currently in a regime change. One scalar cannot see the difference. Continual-learning instinct says volatile weights should stay plastic. Predictive-processing instinct says a residual of expected size is not a reason to reopen a belief (Friston, 2010; Yu and Dayan, 2005). Both uses of \(V_d\) are valid. They must not share a multiplier.

This paper tests that split on the VoltMem write path.

**Contributions.**

1. We show that the homeostatic law *double-charges* \(V_d\) (numerator of \(E_t\) and denominator of \(\theta_t\)), and that this is load-bearing for two opposite failures: it blocks explicit career change after entrenchment, and it insures mislabeled stable facts against weak evidence.
2. We define online surprise as leftover mismatch after anticipation, \(r_t = |M_t - \hat{E}_t|/\sigma_t\), with \(V_d\) widening \(\sigma\). An EMA of raw \(M_t\) is not surprise; it sits on a knife-edge under a constant weak stream.
3. We show that blending the \(V_d\) exponent between 0 and 1 inherits neither win. A discrete **composite** gate does.
4. We restore accumulated weak evidence as a sleeptime detector on time-decayed belief mass. Spacing, not count, is the diagnostic.
5. We measure that the control law is a small term in end-to-end `remember()` error, and that topic similarity cannot separate same-fact updates from coexisting facts. Two-stage linking, not a threshold, is the architectural repair.

We do not claim that these control laws implement consciousness, allostasis in Sterling's physiological sense, or a new continual-learning algorithm on Split-MNIST. The four-way EWC comparison outlined in the research log was not run. The evidence here is scripted agent-memory probes.

---

## 2. Related work

**Uniform and volatility-weighted consolidation.** EWC (Kirkpatrick et al., 2017) applies one global elastic penalty, anchored to old weights. Emate (2026) scales that penalty per domain by \(V_d\) measured before the update. The present paper leaves retrieval and EWC aside and asks only when a *symbolic* memory slot should overwrite.

**Surprise-gated plasticity.** Gating learning on prediction error is not new. Liakoni et al. (2017) accumulate reward-rate mismatch across timescales and raise synaptic plasticity when unexpected uncertainty exceeds expected uncertainty. Fuze uses per-synapse EMA statistics of surprise as a metaplastic gate, without task boundaries or Fisher passes. SuRe (Li et al., 2025) ranks replay by negative log-likelihood. Farhang et al. (2026) use a predictor over a frozen encoder both to gate episodic writes and as a metacognitive signal. Our \(r_t\) is the same family of idea, applied to a memory *slot* rather than a synapse or a replay buffer: surprise is distance from predicted mismatch, not from stored text, and not from a lifetime counter.

**Expected versus unexpected uncertainty.** Yu and Dayan (2005) distinguish noise the system should already have budgeted for from a change in the world's contingencies. Domain \(V_d\) is our expected-uncertainty prior. Residual \(r_t\) is unexpectedness at a step. Collapsing them is the defect we measure.

**Agent memory stores.** Mem0, Zep/Graphiti, and related systems retrieve by embedding and then add, update, or skip. We do not compare product quality. We measure a sequential structure they share with VoltMem: matching happens before the overwrite rule. A miss never consults the control law.

---

## 3. Control laws

On `observe()`, an extractor supplies mismatch magnitude \(M_t \in [0,1]\), source reliability \(R_t\), and (optionally) a domain. Let \(C\) be confirmation count, \(G_t\) a goal-delta factor, \(L_t\) load (here 1). Residual evidence is

\[
\mathrm{res}_t = \frac{M_t R_t}{C^{\alpha}} G_t, \qquad \alpha = 0.6.
\]

**Homeostatic** (the original VoltMem law):

\[
E_t = \mathrm{res}_t \cdot V_d, \qquad \theta_t = \theta_0 / V_d \cdot L_t.
\]

Update iff \(E_t > \theta_t\). \(V_d\) shrinks the score and raises the bar. For a medium-stable job (\(V_d \approx 0.3\)), evidence is cut to a third while the bar more than triples.

**Allostatic:**

\[
E_t = \mathrm{res}_t, \qquad \theta_t = \theta_0 / V_{\mathrm{trait}} \cdot L_t \cdot s(m),
\]

where \(V_{\mathrm{trait}}\) is the domain prior (not a drifted EMA), and \(s(m) \in [S_{\min}, 1]\) is a decreasing function of recent leftover surprise. \(V_d\) sets the bar once.

**Residual surprise.** Let \(\hat{E}\) and \(\sigma\) be a per-item running mean and scale of mismatch, with \(\sigma\) widened by \(V_d\) so ordinary noise on a volatile channel is expected:

\[
r_t = \min\bigl(1,\; |M_t - \hat{E}_t| / (\sigma_t \cdot Z)\bigr), \quad Z = 3.
\]

\(E_t\) still uses \(M_t\). Surprise uses \(r_t\). Confirms pull \(\hat{E}\) down. Time decay on a persisted `surprise_at` (30-day half-life) is the route back to settled. A lifetime `mismatch_count` has no such route; it only ratchets open.

**Composite** is a switch, not an exponent \(p \in (0,1)\):

- allostatic if \(M_t \ge 0.85\) and the source is an explicit statement, or if \(\hat{E}\) has been learned and \(r_t \ge 0.5\);
- otherwise homeostatic.

A fresh item has no \(\hat{E}\), so the first weak blip cannot open the easy-update path.

**Sleeptime.** Online \(r_t\) asks whether this *step* was unexpected. After a few similar asides, \(\hat{E} \approx M\) and \(r_t \to 0\). Accumulated weak evidence is scored later as time-decayed mass \(\sum M_t R_t \, \tfrac12^{\mathrm{age}/30\mathrm{d}}\) against a bar \(0.35 / V_d\), ignoring rows before the last confirm. Lifetime counts are not read.

---

## 4. Experimental setup

Probes are scripted. Unless noted, `observe()` is called with `domain=` and `mismatch_magnitude=`, so Batteries A–E isolate the control law from routing. Negative control on A: REAL \(>\) flat \(>\) swap of the \(V_d\) map. Errors on E and J are typed: a **false update** or **false merge** destroys a stored fact (irreversible); a missed update or duplicate leaves both facts retrievable.

| Battery | What it asks | Path |
|---|---|---|
| A | Retain/update labels under real / flat / swap priors | `observe(domain, M)` |
| C | Explicit recency-shift (career change after quiet; preference control) | same |
| D | Weak slow-burn (sixteen casual mentions; daily vs monthly) | same |
| E | Classifier label noise at the real confusion structure, and at 50% mislabel | same |
| F/J | End-to-end `remember(text)`: update + coexist | matcher + law |
| G/H | Must-link vs must-not-link pairs, held-out 56 | linking only |

The heuristic classifier is \(\approx 84\%\) accurate on a 230-utterance corpus. Ground truth on E stays tied to the *true* domain: a mislabel does not change whether the memory ought to update.

---

## 5. Results

### 5.1 Two ingredients, two jobs

Sweeping the \(V_d\) exponent in \(E_t\) (\(p=1\) homeostatic, \(p=0\) allostatic) crossed with \(s(m)\) on/off: only \(p=0\) recovers recency-shift, and it is a cliff. At \(p=0\) the entrenched career change clears the bar by \(\sim 27\%\); at \(p=0.25\) it misses by \(\sim 6\%\). \(s(m)\) changed no outcome at any \(p\) on Battery C, because every C probe ends in explicit \(M=0.90\), which clears a medium-band \(\theta\)-cap regardless of surprise.

Battery A remains 20/20 under real priors with REAL \(>\) flat \(>\) swap intact.

### 5.2 The double charge is insurance

At the classifier's real error rate:

| law | accuracy | false updates / 18 |
|---|---|---|
| homeostatic | 94.9% | 0.93 |
| allostatic | 93.8% | 1.12 |
| composite | 94.9% | 0.93 |

Allostatic loses 1.1 points and produces 20% more false updates. Under 50% mislabel the gap widens (3.44 vs 2.84 false updates). Every allostatic-only failure has the same shape: a very-stable fact (`personality_trait` \(V=0.05\), `biographical` \(V=0.10\)) misread into a more volatile band, then contradicted by weak evidence. Removing \(V_d\) from \(E_t\) fixes career changes and breaks mislabeled traits. It is a trade. Partial \(p\) buys nothing (cliff at 0). Composite is the remaining option: it matches homeostatic on E and allostatic on C.

### 5.3 Surprise is leftover, not raw difference

An EMA of raw \(M_t\) under a constant weak stream drove \(\theta\) from 0.500 to 0.2941 against \(E_t = 0.2940\) — 2e-4 above the trigger. Nine of ten \(S_{\min} \times\) half-life settings caught the change; the shipped 14-day half-life was the failing corner. That is calibration of a quantity that is not surprise.

After \(r_t = |M_t-\hat{E}_t|/\sigma_t\):

| battery | allostatic |
|---|---|
| C recency-shift | hold (still rides dropping \(V_d\) from \(E_t\)) |
| D daily weak stream | never — after a few hits \(\hat{E} \approx M\), \(r_t \to 0\) |
| E label noise | unchanged (fresh item, \(s(m)\) unread) |

Same words after two weeks of asides are not surprising. That is the definition working. Catching the pile is not a live-EMA job.

### 5.4 Composite and sleeptime

| battery | composite |
|---|---|
| A real priors | 20/20 |
| C recency-shift | hold (career U, preference R) |
| D live weak stream | never |
| E real label noise | 94.9% / 0.93 FU (identical to homeostatic) |
| E 50% mislabel | 84.2% / 2.84 FU (identical to homeostatic) |

Sleeptime on the logged pile:

| stream | live | overnight `consolidate()` |
|---|---|---|
| 16 daily weaks, `professional_context` | never | supersedes |
| same 16, monthly | never | does not rewrite |
| 16 daily weaks, `core_preference` | never | does not rewrite |

Identical evidence, identical count, only spacing differs. A counter that never decays cannot see this. Time-decayed belief mass can.

### 5.5 The law is behind the matcher

`remember(text)` only, 18 update probes:

| similarity | labels | routed | correct | correct \| routed |
|---|---|---|---|---|
| keyword | oracle | 22.2% | 22.2% | 100% |
| hashing | oracle | 27.8% | 27.8% | 100% |
| sentence-transformers | oracle | 77.8% | 72.2% | 92.9% |
| sentence-transformers | shipped | 55.6% | 44.4% | 80.0% |

Error budget: similarity \(\approx 50\) points; classifier-via-routing \(\approx 28\); residual routing \(\approx 22\); decision once routed \(\approx 5.6\). Batteries A–E cannot see the first three. Two of four probes that never route even with embeddings are the explicit career-change and goal-change cases allostatic was built to recover. End-to-end allostatic vs homeostatic at the best configuration is 77.8% vs 72.2% — one probe in 18, not significant.

### 5.6 No threshold separates topic from identity

Held-out 56 pairs (28 must-link, 28 must-not-link). Ranking is inverted: “proficient in Python” vs “proficient in Japanese” scores 0.80 on the default scorer; the career change that must link scores 0.25. Non-separability replicates on keyword, hashing, and sentence-transformers. Embeddings double must-link recall (12/28 \(\to\) 25/28) at almost no change in false-merge *count* (13/28 \(\to\) 12/28), and raise *severity*: keyword false merges mostly discard the incoming fact; embedding false merges supersede the stored one.

Stage-1 recall at bar 0.20 is 27/28 held-out. A perfect verifier therefore scores 55/56. A cheap lexical verifier (cardinality + change marker) is 19/24 on the fitted split and 35/56 held-out — worse than the embedding ladder (41/56). An LLM verifier asking “same subject?” and “same attribute?” (attribute = the question a fact answers, never the answer) scores 52/56 hosted (`gpt-4o-mini`) and 49/56 local (`qwen2.5-coder:14b`) with **0 false merges**. The hosted model is three pairs higher and two irreversible losses worse. Under the paper's error taxonomy the local model is preferred. Prompt framing moved the hosted model from 29/56 to 52/56; the local model moved 48 \(\to\) 49 and never collapsed on the badly posed prompt. A 3B rubber-stamp (24 false merges) and a 14B instruct model that almost always says KEEP_BOTH (5/28 must-link) show that model choice does not track “bigger is better.”

On the combined 18 update + 14 coexist harness:

| configuration | update ok | coexist ok | overall | irreversible |
|---|---|---|---|---|
| keyword, shipped labels | 22.2% | 64.3% | 40.6% | 5 |
| keyword, oracle labels | 22.2% | 35.7% | 28.1% | **9** |
| embeddings, shipped | 44.4% | 78.6% | 59.4% | 4 |
| embeddings, oracle | 72.2% | 71.4% | 71.9% | 4 |
| embeddings + verify, either labels | 55.6% | 100% | 75.0% | **0** |

Under the threshold ladder, improving the classifier makes data loss *worse*: correct labels put two distinct facts in the same slot. Two-stage linking removes that incentive (coexist 100% either label condition). Allostatic vs homeostatic on this combined set is 78.1% vs 75.0%, zero irreversible errors either way — still one-probe scale.

---

## 6. Discussion

The prior and the residual are different jobs. Charging \(V_d\) twice is not a bug in general: it is the wrong setting for an explicit correction of an entrenched medium-stable fact, and the right setting for weak evidence on a misfiled trait. A blend does not interpolate those cells. A gate does.

Online surprise must not be an average of contradiction. If it is, a predicted stream impersonates a regime change until a half-life is nursed across a line. Leftover-after-anticipation habituates, which is what the definition asked for, and it therefore cannot catch a slow pile. Horizon belongs to a decayed accumulator, preferably off the interactive path.

None of that matters if the new sentence never finds the slot. The control-law debate is about six points in an error budget whose first fifty are the similarity function. We report that not as a reason to abandon the law, but as a reason not to over-read end-to-end allostatic-vs-homeostatic tables.

---

## 7. Limitations

- Probes are synthetic and small. Battery F/J rows other than the 50- and 28-point gaps should not be read individually.
- Recency-shift still sits on a hand-tuned \(\theta\)-cap (`EXPLICIT_E_RATIO`). Robustness of \(p=0\) to domain misclassification was the original open risk; composite addresses the measured cell (weak evidence on a settled item) rather than proving robustness in general.
- The Split-MNIST four-variant design (EWC vs \(V_d\) vs residual vs residual+mode-switch, with forgetting-after-recency as the key diagnostic) was not executed.
- Related work on surprise-gated *weight* plasticity is active; we did not re-run those methods on our probes. The claim is about a symbolic memory slot.
- Linking results depend on a 24/56 split and on prompt wording for the hosted verifier. Held-out pairs were written from a structural grid after the first 24; they are not a public benchmark.

---

## 8. Conclusion

A volatility prior answers how readily a *kind* of memory should move. Leftover surprise answers whether this step was already priced in. Using one number for both produces a characteristic pair of failures, and blending the number does not fix the pair. A switch, a residual that habituates, and a sleeptime accumulator that sees spacing, together implement the split. The matcher still sits in front. Duplicates are recoverable; silent overwrites are not.

---

## References

Casali, A. G., et al. (2013). A theoretically based index of consciousness independent of sensory processing and behavior. *Science Translational Medicine*.

Farhang, A., et al. (2026). Surprise as a signal for plasticity and metacognition. arXiv:2606.31495.

Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*.

Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*.

Li, T., et al. (2025). SuRe: Surprise-driven prioritised replay for continual LLM learning. arXiv:2511.22367.

Liakoni, V., et al. (2017). Adaptive learning and decision-making under uncertainty by metaplastic synapses guided by a surprise detection system. *eLife*.

Sterling, P. (2012). Allostasis: a model of predictive regulation. *Physiology & Behavior*.

Yu, A. J., & Dayan, P. (2005). Uncertainty, neuromodulation, and attention. *Neuron*.

Emate, R. (2026). Volatility-Adjusted Memory Protection: A causal control knob for continual learning and LLM agent memory. Zenodo. https://doi.org/10.5281/zenodo.21962419

---

*Code: https://github.com/Rouche01/voltmem · Package: voltmem 0.4.0*
