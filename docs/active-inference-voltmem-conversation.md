# Active Inference, Continual Learning & VoltMem — Conversation Log

*Discussion anchored to: Martin Maier, ["From Artificial Intelligence to Active Inference: The Key to True AI and 6G World Brain [Invited]"](https://arxiv.org/abs/2505.10569) (arXiv:2505.10569v1) — [HTML version*](https://arxiv.org/html/2505.10569v1)
*Referenced project: [Rouche01/voltmem](https://github.com/Rouche01/voltmem) — "Current-truth memory for LLM agents."*
*Date: 26–27 August 2026*

---

## 1. Active inference vs. AI/ML in optical networks: beyond pattern recognition

**Point:** Active inference goes beyond pattern recognition to "action performing" — but the paper draws the line more sharply.

- AI/ML in optical networks is pattern recognition (QoT estimation, anomaly detection in OPM).
- The "active" in active inference literally means actions in the POMDP: agents actively infer using the available actions in `U`, rather than passively inferring without acting.
- Sharper framing:
  - It's a closed action-perception loop, not just "action." The agent acts *in order to* test and reduce its own uncertainty.
  - The real differentiator is a causal world model — learning causal structure, not just correlations (largely absent in AI/ML including RL).
  - The stated payoff is solving three open gaps: training (no big-data requirement), lifelong/continual learning, explainability.
  - The paper's own terms: *passive AI* vs *active AI* (networks as living organisms, not machines).
- One-liner: AI/ML = recognize patterns in signals (perception only). Active inference = model causes, act to minimize surprise, learn continuously — a closed loop that makes the network a self-organizing agent.



## 2. Continual learning as the crux; context/memory engineering as the route

- Continual learning is the crux — the paper lists "lifelong (continual) learning" as a core open gap and ties it to the action-perception loop.
- Pushback on "we've achieved active inference via context engineering": context memory (RAG, long context, scratchpads, agentic loops) gives a loop and in-context adaptation, but NOT the causal generative model. It's non-parametric and ephemeral — closer to working memory than continual learning.
- Honest read: context engineering is a bridge, not the source. Real continual learning likely needs a parametric mechanism (test-time training, fast weights, online world model) PLUS the free-energy framing for deciding what to update.
- Tension: active inference is pitched as needing no big data and being energy-efficient — the opposite of LLM scaling. LeCun: "we don't know how to do this today with AI systems."



## 3. Refinement + VoltMem

Refined thesis (correct): the orchestration policy for memory is the same control problem continual learning must solve.

- VoltMem scales protection by volatility (personality_trait 0.05 … current_task 0.90). Volatility = a prior precision on a state, playing the role of the transition-dynamics matrix B.
- Composite escalation ("homeostatic on weak evidence; allostatic on unexpected residual") = precision-weighted Bayesian updating; "unexpected residual" = surprise / prediction error. This is the inference half of active inference over an external fact store.
- Bridge to continual learning: catastrophic forgetting is a failure to know what to protect. EWC protects weights by Fisher information — the same as VoltMem's volatility prior, transposed from facts to weights. High-Fisher weight ≈ personality_trait 0.05; low-Fisher weight ≈ current_task 0.90.
- consolidate / sleeptime maintenance = Complementary Learning Systems theory (fast episodic store + slow consolidation), a credible route to continual learning without forgetting.
- Gaps: (1) discrete slots vs distributed weights — the principle transfers, the slot mechanism may not; (2) it's inference, not yet "active" — closest to active is epistemic action over its own memory (reconcile_twins, pattern_audit, verify_on_write).
- Provocation: VoltMem has learned, self-calibrating priors (domain_stats, auto_discover); EWC's Fisher matrix is frozen. If online-calibrated volatility beats a static importance estimate, the transferable claim is: adaptive precision estimation is the missing piece in both.



## 4. What's transferable / what to do now

- Concept to transfer: expected free energy G (the "active" half). add()/remember() are pure F (retrospective). Sleeptime maintenance is where G belongs — make it value-driven (audit slots with highest expected uncertainty reduction).
- Do this week: adaptive-vs-static precision experiment. Add a static-prior vs online-calibrated-prior condition to voltmem_eval.py. Does calibrated recover hand-tuned "real" performance without tuning? Product value regardless: self-tuning to a deployment's real domains.
- Benchmark axis: stability–plasticity frontier — corruption rate on stable slots vs staleness rate on volatile slots. Exactly how continual learning measures forgetting.
- Feature that ships value + advances thesis: budget-aware maintenance via information gain — rank pattern_audit / reconcile_twins / consolidate by expected surprise reduction per unit cost (the epistemic term of G).
- Hold off on: global preference C, hierarchical/nested Markov blankets. Bank the adaptive-precision result first. (Update: §7 gives both C and hierarchy a concrete home inside the generative model — deferred in priority, not discarded.)



## 5. Eval design sketch

(Working from README description of voltmem_eval.py; snap signatures to actual function names.)

New axis — static vs online-calibrated priors:

- static_real: hand-tuned table, no online update — ceiling
- static_flat: all 0.5, no update — uninformed control
- static_swap: inverted, no update — adversarial control (7/20 floor)
- calib_coldstart: starts flat, auto_discover on — thesis probe
- calib_fromprior: starts real, auto_discover on — ship config

Key comparison: does calib_coldstart climb from flat up to static_real as it observes the confirm/mismatch stream?

Metrics:

- Corruption rate (stable slots): P(accepted confident-false blip). Lower better.
- Staleness rate (volatile slots): P(missed true update). Lower better.
- Selective-update score: existing 20-probe escalation battery.
- Retrieval separation: existing current-vs-stale margin (+0.589).
- Calibration convergence (new, key evidence): #observations until learned volatility lands within ε of steady state; how close to the real table.
- Calibration error (optional): |learned_volatility − empirical_mismatch_rate| per domain.

Results table (mirrors Battery A/B):


| Battery A — selective updating        | corruption | staleness | sel-score |
| ------------------------------------- | ---------- | --------- | --------- |
| static_real (ceiling)                 | 0.05       | 0.05      | 20/20     |
| static_flat (uninformed)              | ~          | ~         | 15/20     |
| static_swap (adversarial)             | ~          | ~         | 7/20      |
| calib_coldstart @ 50 obs              | ~          | ~         | ?         |
| calib_coldstart @ 200 obs             | ~          | ~         | ?         |
| calib_coldstart @ 500 obs -> recover? | ->0.05     | ->0.05    | ->20/20   |
| calib_fromprior (ship)                | ~          | ~         | ?         |


Battery B — retrieval separation:
static_real +0.589 | flat +0.202 | swap −0.267 | calib_coldstart@500 -> ?

Minimal code skeleton:

```
# experiments/precision_calibration_eval.py
CONDITIONS = {
    "static_real":     dict(domains=REAL_PRIORS,  auto_discover=False),
    "static_flat":     dict(domains=FLAT_PRIORS,  auto_discover=False),
    "static_swap":     dict(domains=SWAP_PRIORS,  auto_discover=False),
    "calib_coldstart": dict(domains=FLAT_PRIORS,  auto_discover=True),
    "calib_fromprior": dict(domains=REAL_PRIORS,  auto_discover=True),
}

CHECKPOINTS = [50, 200, 500]

def run(name, cfg, stream, probes):
    mem = create_memory(":memory:", user_id=name, **cfg)
    rows = []
    for i, obs in enumerate(stream, 1):
        mem.add(obs)
        if i in CHECKPOINTS:
            rows.append(dict(
                obs=i,
                corruption = corruption_rate(mem, probes.stable),
                staleness  = staleness_rate(mem, probes.volatile),
                sel_score  = selective_update_score(mem, probes.escalation),
                retr_sep   = retrieval_separation(mem, probes.retrieval),
                vol_err    = volatility_calib_error(mem.domain_stats(), REAL_PRIORS),
            ))
    return rows

def corruption_rate(mem, stable_probes):
    hits = sum(mem.search(p.query)[0]["memory"] != p.truth for p in stable_probes)
    return hits / len(stable_probes)
```

The one plot: sweep escalation threshold θ (from calibrate_escalation.py's E_t vs θ) and plot corruption (x) vs staleness (y) per condition — the stability–plasticity frontier. Pareto-dominant curve wins; if calib_coldstart converges onto static_real as obs grow, that's the headline figure.

Decision rule:

- calib_coldstart@500 within noise of static_real on all four columns -> thesis confirmed; ship calib_fromprior; write up as adaptive-beats-static.
- Plateaus above static_real -> priors carry information data can't recover; hybrid warm-start + drift; calib_fromprior is the product answer.
- Never separates from flat -> mismatch signal too sparse; add a cold-start guard.



## 6. Is VoltMem's policy layer active inference?

Verdict: close on the inference and precision half; absent on the generative model and the active (prospective) machinery. Best described as Bayesian belief updating with adaptive precision over a discrete symbolic state space — a large, legitimate chunk of active inference, but not yet active inference proper.

Component scorecard:

- Prior beliefs D — YES: domain priors at cold start
- Precision / transition B — YES: volatility priors
- Precision learning — YES: domain_stats / auto_discover (beyond vanilla AIF)
- Belief update / F — YES: escalation = precision-weighted update
- Generative model p(s,o) — PARTIAL: classifier + volatility table, not a predictive causal model
- Markov blanket b=(u,y) — PARTIAL: boundary exists (user_id namespace), not formalized
- Expected free energy G / policies — NO: maintenance scheduled, not value-selected
- Preferences C — NO: implicit at best
- Action on environment — NO: internal writes only

Strong matches: volatility = per-domain precision on state change (role of B); escalation = discrete approximation of minimizing F; domain_stats/auto_discover = online precision learning (most AIF tutorials don't).

Biggest gap: no generative model in the AIF sense — VoltMem doesn't model p(s,o) or predict observations; it classifies and stores current-truth. The paper: "the generative model lies at the heart of active inference." (Designed in §7.)

Caveat: assessed from README, not source. A predictive component would shift the generative-model row toward YES.

## 7. Designing VoltMem's generative model

Core shift: today VoltMem stores a value and re-ranks it; a generative model predicts the next observation about a fact before seeing it, then updates on prediction error. That turns the volatility table from a heuristic penalty into a model of how facts behave.

Generative model vs generative process (paper): the generative process = the real world (users move cities, hold stable prefs, have fast-changing moods); the generative model = VoltMem's internal model of it, which should "closely biomimic" the process. The volatility taxonomy is already a coarse sketch; the upgrade is making it predictive and probabilistic per slot.

State: for each slot (subject, attribute) the hidden state s = true current value, as a categorical distribution over candidate values, plus time since last observation. Keep a belief q(s), not a stored string. Stays a tractable categorical POMDP (zero-dep / millisecond — no neural model).

POMDP tuple → VoltMem primitives:

- s ∈ S: true current value of a slot (categorical over candidates)
- o ∈ O: incoming extracted fact/utterance about the slot
- u ∈ U: write / overwrite / insert-as-twin / verify / reconcile / consolidate / expire
- A (likelihood p(o|s)): observation/noise model — the genuinely new piece
- B (transition p(s_t|s_{t-1})): volatility prior, now a real transition kernel
- C (preferred outcomes): consistency / low-contradiction preference (drives maintenance)
- D (prior over initial states): cold-start domain prior

Two matrices that matter:

- B — volatility as a transition kernel. Volatility = off-diagonal mass. personality_trait 0.05 → near-identity; current_task 0.90 → mass leaks each step. Parameterize B by elapsed time and staleness falls out for free: old volatile belief diffuses toward uniform → high entropy → ranks lower. "Stale volatile memories rank lower" becomes derived, not hardcoded.
- A — the piece not there yet. Likelihood p(o|s): given my belief is true, how likely is this utterance? Encodes source reliability (direct user = high precision; inferred/paraphrased = low). A confident-but-wrong blip has low likelihood under a correct-belief model with a noise term, so it barely moves the posterior. A is why corruption-resistance is right, not a threshold.

Inference loop:

```
1. PREDICT   ŝ = B(Δt) · q(s_k)      # let belief drift by elapsed time
             ô = A · ŝ                # what observation do I expect?
2. OBSERVE   o = extracted fact
3. SURPRISE  F = −ln p(o | ô)         # variational free-energy contribution
4. ATTRIBUTE the surprise:
     - to observation noise (A)  → belief barely moves  → RESIST  (stable slot)
     - to a state change (B)     → belief shifts to o    → UPDATE  (volatile slot)
   Automatic: low-volatility B can't explain the change → noise;
   high-volatility B can → update.
5. POSTERIOR q(s_k) ← Bayes(q, o)     # escalation, re-derived
6. ACT (optional) if expected free energy of verifying is favorable
```

Corruption-resistance and update-readiness stop being two policies — both fall out of how B and A compete to explain a surprising observation. homeostatic/allostatic/composite become special cases of one Bayesian update.

Where C and actions plug in: C = prior over preferred observations = "prefer a low-contradiction, internally consistent memory." Makes consolidate / reconcile_twins purposeful: pick action u minimizing expected free energy G — the audit whose expected surprise reduction (toward C) is highest per unit cost. This is the §4 G/maintenance upgrade with a home in the model.

Build first (categorical, not neural):

- B(Δt): 2-parameter kernel per domain (volatility + time constant). Add elapsed-time mixing to existing volatility.
- A: per-observation precision from extraction confidence × source-type reliability. Start with 2–3 source tiers.
- Replace the escalation threshold with predict → surprise → attribute → update.
- Keep domain_stats/auto_discover as the upper level learning B's volatility parameter — a two-level hierarchical model (slow: how volatile is this domain; fast: what's the value) = the paper's "hierarchical Markov blankets with temporal and spatial depth."

Thesis payoff: once B is a real kernel and A a real likelihood, the corruption-vs-staleness frontier isn't tuned — it's a consequence of the model's parameters, and "adaptive precision beats static" becomes "the model learns B and A from data better than hand-set priors."

Caveat: designed against the README's description of primitives. Pasting the current escalation/consolidate internals would pin down exactly which lines become A, which become B, and where the predict step slots in.

---

Drafted with Dia