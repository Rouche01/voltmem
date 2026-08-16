# Consciousness Might Not Look Like Anything

## Allostatic range as a hypothesis, and what it predicts for a memory control law

Richard Emate  
Independent · richard@theemate.com  
15 August 2026

**Combined draft.** Section 2 is a philosophical hypothesis. Sections 3–6 test engineering predictions that the analogy suggests. **No result in this paper is evidence that a memory system, a learning rule, or VoltMem possesses experience.** The hard problem is left open. For the empirical claims without the consciousness framing, see `paper/allostatic-memory-control.md`. Source log: `docs/allostatic-consciousness-to-voltmem.md`.

---

## Abstract

If consciousness tracked how violently a system is shoved around, a thermostat would outrank a person. Disturbability is cheap. What matters is the structure of the response — differentiation × integration — and, more than that, whether the system can absorb perturbation *before* it registers as a jolt. On this view a more competent system may look, from the outside, as if nothing happened. Homeostasis waits for deviation and corrects toward a fixed setpoint. Allostasis (Sterling, 2012) moves the setpoint in advance. We take the *range* of allostatic versus homeostatic regulation, not a fixed allostatic amount, as the interesting quantity, and leftover mismatch \(r_t = \mathrm{distance}(\hat{E}_t, O_t)\) as the moment-to-moment residual after anticipation.

That hypothesis is not tested here as a theory of mind. It does make a prediction about any memory that uses a single volatility weight: that weight conflates a slow prior about a *channel* with a fast residual at a *step*. We separate them in an agent-memory overwrite rule. Dropping the prior from the evidence score recovers explicit recency-shift as a cliff, not a blend, and costs insurance against a mislabeled stable fact plus weak evidence. A discrete gate keeps both cells. Surprise defined as leftover after anticipation habituates; a predicted weak stream goes quiet. Accumulated weak evidence is a sleeptime job on time-decayed belief mass: sixteen daily mentions supersede overnight, the same sixteen monthly do not. The overwrite law is only reached after a match. Topic similarity cannot separate same-fact updates from coexisting facts; two-stage recall-then-verify takes irreversible errors to zero.

The consciousness claim remains philosophical. The engineering claim is the measured split: prior versus residual, switch versus blend, leftover versus raw difference, matcher in front of the law. A highly allostatic system, if the analogy holds, is not the one that moves the most. It is the one with range enough to absorb most of the world in advance, and leftover only at the edges.

---

## 1. What this paper is and is not

There are two documents one could write from the same notes.

The first is a position on consciousness: that it does not have a distinctive outward behavior, because competence is neutralizing a perturbation before the flinch. That claim cannot be established by an overwrite rule in a Python library.

The second is a methods paper on agent memory: that a volatility prior and a live residual are different signals, and that using one number for both produces a characteristic pair of failures. That claim can be tested on scripted probes.

This document is the third thing: both, with a seam. Section 2 states the hypothesis and stops short of the hard problem. Sections 3–6 test only the engineering predictions. If those predictions fail, the analogy was a poor searchlight. If they hold, they still do not show that leftover mismatch *is* experience. Casali et al. (2013) already have a clinical test of “perturb and look at the structure of what comes back” (PCI). We are not competing with it.

---

## 2. Hypothesis: range, not dynamism

### 2.1 The thermostat

Start with a simple claim: the most conscious being is the one whose internal state is most disturbed by the world. A thermostat kills it. Its state flips hard when perturbed. Nobody considers it more conscious than a rock. Raw disturbability cannot be the measure. What matters is the *structure* of the response, not its magnitude or frequency.

Integrated Information Theory (Tononi and Edelman, 1998; Tononi et al., 2016) names two joint properties. **Differentiation** is a large repertoire of distinguishable internal states. **Integration** is that repertoire coming from one system that cannot be decomposed into independent parts without losing information. A bundle of switches is differentiated and not integrated. A single wire is integrated and not differentiated. Neither is enough. The framework is contested (Aaronson, 2014; Doerig et al., 2021): \(\Phi\) can be made large for systems with no plausible claim to experience. We borrow the joint requirement, not \(\Phi\).

Even that refinement aims at the wrong target. The interesting system is not the one that gets pushed around a lot. It is the one that can absorb and correct for a wide range of perturbations while remaining coherent. Internal change is a side effect of regulation. Consciousness, if it tracks anything here, tracks regulatory competence rather than disturbability.

### 2.2 Allostatic range

The Free Energy Principle (Friston, 2010) gives that move a formal home. A system that persists as a distinct entity keeps its internal states in a viable range, by updating its model to match the world (perception) or acting on the world to match its model (action). Depth is how much of a perturbation the system can anticipate and neutralize *before* it registers as a raw internal jolt. Paradoxically, a more competent system may look calmer, not more volatile. It is absorbing disturbance predictively. From the outside there may be nothing to watch. That is the sense in which consciousness might not look like anything.

Sterling (2012) draws the operative distinction.

- **Homeostasis** — regulation toward a *fixed* setpoint, reactive, after deviation is detected.
- **Allostasis** — regulation via *anticipated, context-dependent, shifting* setpoints, proactive.

The hypothesis: the measure is the *range* of allostatic versus homeostatic regulation a system can cover, not a fixed allostatic amount. Per controlled variable \(i\),

\[
A_i = H_i \times R_i \times C_i
\]

with **horizon** \(H_i\) (how far ahead the system acts before impact), **retargetability** \(R_i\) (whether the setpoint itself can move, across how many variables, independently), and **context-sensitivity** \(C_i\) (how much of the space of possible contexts actually drives the retargeting). The product is multiplicative because each term is near-useless without the others: horizon without retargetability is a longer reflex; retargetability without context is a clock. System-level range should be integration-weighted. Independent parallel regulators, summed, inflate the score without unification — the bundle of switches again.

Damasio (2010) treats consciousness as built from the organism's representation of its own homeostatic/allostatic states. Barrett and Simmons (2015) connect allostasis to prediction-error minimization. We rely on that bridge; we do not add a new physiology.

### 2.3 Leftover, not total activity

Where does experience itself come from, on this model? Three readings:

1. **Residual / prediction error** (Friston; Clark, 2016; Seth, 2021) — leftover mismatch between what the allostatic model anticipated and what occurred. Predicts vividness at surprise and thinning under habituation.
2. **Mode arbitration** — experience at the seam between homeostatic and allostatic control.
3. **Aggregate** — sum of both systems' activity. This collapses back to dynamism, which was rejected.

We adopt (1) as the strongest candidate *as a trigger condition*, not as a solution to the hard problem. Given expectation \(\hat{E}_t\) and outcome \(O_t\),

\[
r_t = \mathrm{distance}(\hat{E}_t, O_t).
\]

Moment to moment, the “conscious-like” signal is residual-after-anticipation, not raw disturbance. In the trait sense, capacity is allostatic range. A highly allostatic system should show large \(A_i\) and *small typical* \(r_t\). Residual appears at the edges, where the model is under-fit. If you are looking for consciousness in behavior, you may be looking at the edges of a system that is otherwise quiet.

### 2.4 What the hypothesis predicts for a volatility weight

A memory (or a continually learning model) that attaches one volatility \(V_d\) to each *kind* of fact is already a step toward allostasis: the setpoint depends on context of a kind. It is still a slow prior, computed retrospectively, with no live residual. A fixed elastic penalty on old weights (EWC) is pure homeostasis: \(R \approx 0\), \(C \approx 0\).

The hypothesis says \(V_d\) conflates two signals:

1. A slow domain-level **prior** about how noisy this channel usually is.
2. A fast per-step **residual** — live mismatch between expectation and observation.

A domain can be historically volatile (\(V_d\) high) and currently well-predicted (\(r_t\) low). Under one scalar that case is indistinguishable from a domain that is both historically and currently volatile. They should be treated differently.

Further predictions, if range rather than amount is the quantity:

- A channel should be able to *move between modes* (settle into homeostatic protection under sustained low residual; reopen under rising residual), rather than carry one weight for its lifetime.
- The diagnostic that this is a distinct capability, not extra tunable capacity, is **forgetting-after-recency**: a domain that was stable for a long time and then shifted. Static \(V_d\) cannot see the shift; \(r_t\) can. The win should appear specifically there, not uniformly.
- Surprise implemented as “how different from stored text” is the thermostat again. Surprise is leftover after anticipation. Same words after a predicted stream are not surprising.
- Horizon requires decay. Identical evidence at different spacing must not count the same. A counter that only rises has no route back to settled.
- Dropping the prior entirely is not free: the prior is insurance when the *kind* of thing was guessed wrong.

Sections 3–6 test these predictions on an agent-memory overwrite rule. They do not test \(A_i\) as a measure of consciousness. The Split-MNIST four-variant design that would have tested forgetting-after-recency against EWC was not run.

---

## 3. Control laws (the engineering object)

On `observe()`, mismatch \(M_t \in [0,1]\) is distance to stored content, not to a predicted next utterance. Residual evidence \(\mathrm{res}_t = M_t R_t C^{-\alpha} G_t\), \(\alpha=0.6\).

**Homeostatic:** \(E_t = \mathrm{res}_t \cdot V_d\), \(\theta_t = \theta_0 / V_d\). \(V_d\) is charged twice.

**Allostatic:** \(E_t = \mathrm{res}_t\), \(\theta_t = \theta_0 / V_{\mathrm{trait}} \cdot s(m)\). \(V_d\) sets the bar once. \(s(m)\) decreases with recent leftover surprise.

**Residual surprise:** \(r_t = |M_t - \hat{E}_t| / \sigma_t\), \(\sigma\) widened by \(V_d\). \(E_t\) still uses \(M_t\). A 30-day half-life on persisted surprise is the route back to settled.

**Composite** is a switch: allostatic if the statement is explicit and high-\(M\), or if \(\hat{E}\) is learned and \(r_t\) is unexpected; otherwise homeostatic. A fresh item has no \(\hat{E}\).

**Sleeptime:** time-decayed evidence mass \(\sum M_t R_t \tfrac12^{\mathrm{age}/30\mathrm{d}}\) against \(0.35 / V_d\). Online \(r_t\) does not catch a predicted pile.

Related methods already gate *synaptic* plasticity on surprise (Liakoni et al., 2017; Fuze; SuRe, 2025; Farhang et al., 2026). We gate a *slot*. Yu and Dayan (2005) is the expected/unexpected split we are implementing, not discovering.

---

## 4. Experiments

Scripted probes. Batteries A–E pass `domain=` and \(M_t\), isolating the law. F–J call `remember(text)`. Negative control on A: REAL \(>\) flat \(>\) swap. Irreversible errors (false update, false merge) are counted separately from recoverable ones (miss, duplicate). Classifier \(\approx 84\%\) on a 230-utterance corpus.

---

## 5. Results (engineering only)

### 5.1 The prior and the residual do different jobs

Only \(p=0\) (\(V_d\) out of \(E_t\)) recovers explicit recency-shift; \(p=0.25\) misses. It is a cliff. \(s(m)\) changed no C outcome, because C ends in explicit \(M=0.90\). Battery A stays 20/20, REAL \(>\) flat \(>\) swap.

That is the forgetting-after-recency prediction, on a memory slot, for the *explicit* cell. It is not the EWC experiment.

### 5.2 Insurance

At real classifier error: homeostatic 94.9% / 0.93 false updates per 18; allostatic 93.8% / 1.12; composite identical to homeostatic. Every extra allostatic overwrite: very-stable fact, mislabeled, weak evidence. The double charge is a trade, not a defect. Blending \(p\) inherits neither cell. Composite keeps C and E.

This is the prediction that dropping the prior is not free.

### 5.3 Leftover habituates; the pile is horizon

An EMA of raw \(M_t\) sat 2e-4 above the trigger under a constant weak stream. Stretching the half-life from 14 to 30 days shoved it over. That is not surprise.

After \(r_t = |M-\hat{E}|/\sigma\): C holds; D's live stream never fires (\(\hat{E} \approx M\)); E unchanged (fresh item). Same words after two weeks of asides are not surprising — which is why D disappeared from live \(s(m)\), and why that is not a failed experiment.

Sleeptime: 16 daily weaks on `professional_context` stay live-quiet and supersede overnight; monthly drip does not; `core_preference` daily pile does not. Same evidence, same count, spacing differs. That is \(H_i\) as specified. A lifetime counter cannot implement it.

### 5.4 Consciousness, if the analogy held, would still be behind the matcher

A miss never consults homeostatic vs allostatic. Through `remember(text)`, similarity is \(\approx 50\) points of error, classifier-via-routing \(\approx 28\), residual routing \(\approx 22\), the control law once routed \(\approx 5.6\). Two of four never-routed embedding probes are the career-change and goal-change cases the allostatic law was built for. End-to-end allostatic vs homeostatic is one probe in 18.

Topic similarity is non-separable: Python vs Japanese outranks analyst \(\to\) nurse. Two-stage recall-then-verify: stage-1 recall 27/28 at bar 0.20; local `qwen2.5-coder:14b` 49/56 held-out, **0 false merges**; hosted 52/56 with 2 merges. Combined update+coexist harness: embeddings+verify 75% overall, 0 irreversible, vs embeddings 71.9% with 4 irreversible. Under a threshold ladder, a *better classifier makes data loss worse* (keyword irreversible 5 \(\to\) 9 with oracle labels), because correct labels collocate distinct facts. The verifier never consults the label.

If one were tempted to read leftover \(r_t\) as a correlate of experience in this system, one would first need the observation to find the slot. The matcher is not a theory of consciousness. It is the condition under which the control law, and therefore any residual, is even computed.

---

## 6. Discussion

**What the analogy got right as a searchlight.** It forbade treating disturbability as the goal. It split expected noise from unexpected leftover. It forbade calling surprise the distance to stored text. It forbade a lifetime plasticity assigned at birth. It forbade blending two laws that win on different failures. Those forbiddances survived contact with the batteries.

**What it did not earn.** \(A_i = H \times R \times C\) was not measured. PCI was not run. No result shows that small typical \(r_t\) plus large range *is* consciousness, or that a quiet system is more conscious than a noisy one. The title is a hypothesis about outward sign, restated from Friston's “anticipate and neutralize before the jolt.” It is not a finding.

**What we would still believe if the philosophy were withdrawn.** Prior \(\neq\) residual. Switch \(\neq\) blend. Leftover \(\neq\) raw \(M\). Spacing \(\neq\) count. Matcher before law. Duplicates recoverable; false merges not. That list does not require inner life.

---

## 7. Limitations

The same as the empirical companion, plus: this document's title will be misread as a result. Section 1 is there because of that. Philosophical sources (IIT especially) are contested; we do not adjudicate \(\Phi\). Related surprise-gated *weight* methods were not re-implemented. Probes are small and synthetic. The EWC forgetting-after-recency battery remains a design.

---

## 8. Conclusion

A thermostat looks alive. A system that saw the perturbation coming may look like nothing happened. That is a reason to stop using raw volatility as a proxy for contact with the world, in theories of mind and in memory control laws alike.

In the memory setting, the operational split is prior versus residual, implemented as a gate, a habituating leftover, and a sleeptime horizon. In the consciousness setting, the same split is a hypothesis about leftover-after-anticipation and a warning about outward sign. Only the first was tested. The second remains what it was at the start of Section 2: a structural analogy, separable from the engineering, and silent on why any of this should feel like anything.

---

## References

Aaronson, S. (2014). Public exchange with G. Tononi.

Barrett, L. F., & Simmons, W. K. (2015). Interoceptive predictions in the brain. *Nature Reviews Neuroscience*.

Casali, A. G., et al. (2013). A theoretically based index of consciousness independent of sensory processing and behavior. *Science Translational Medicine*.

Clark, A. (2016). *Surfing Uncertainty*. Oxford University Press.

Damasio, A. (2010). *Self Comes to Mind*. Pantheon.

Doerig, A., Schurger, A., & Herzog, M. H. (2021). Hard criteria for empirical theories of consciousness. *Cognitive Neuroscience*.

Farhang, A., et al. (2026). Surprise as a signal for plasticity and metacognition. arXiv:2606.31495.

Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*.

Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*.

Li, T., et al. (2025). SuRe: Surprise-driven prioritised replay for continual LLM learning. arXiv:2511.22367.

Liakoni, V., et al. (2017). Adaptive learning and decision-making under uncertainty by metaplastic synapses guided by a surprise detection system. *eLife*.

Seth, A. (2021). *Being You*. Faber.

Sterling, P. (2012). Allostasis: a model of predictive regulation. *Physiology & Behavior*.

Tononi, G., & Edelman, G. M. (1998). Consciousness and complexity. *Science*.

Tononi, G., Boly, M., Massimini, M., & Koch, C. (2016). Integrated information theory: from consciousness to its physical substrate. *Nature Reviews Neuroscience*.

Yu, A. J., & Dayan, P. (2005). Uncertainty, neuromodulation, and attention. *Neuron*.

Emate, R. (2026). Prior versus residual: homeostatic and allostatic control laws for agent memory updates. VoltMem draft. `paper/allostatic-memory-control.md`

---

*Code: https://github.com/Rouche01/voltmem · Package: voltmem 0.4.0*
