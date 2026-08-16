# Consciousness Might Not Look Like Anything

## Why the interesting system may appear calm — and what that does to a volatility weight

If you are looking for consciousness in behavior, you may be looking at the wrong thing. The system that is most in contact with the world need not flinch, chatter, or otherwise announce that something happened. It may look, from the outside, as if nothing did.

Start with the opposite claim, which is the one that feels obvious: the most conscious being is the one whose internal state is most disturbed by the world. Raw sensitivity. Volatility of the insides as a measure of contact with the outside.

A thermostat kills it. Turn the heat on and its state flips hard. Nobody thinks that makes it more conscious than a rock. Disturbability is cheap. What matters is the *structure* of the response, not its magnitude or how often it happens.

Integrated Information Theory names two joint properties. **Differentiation** is a large repertoire of distinguishable internal states. **Integration** is that repertoire coming from one system you cannot carve into independent parts without losing information. Neither is enough on its own. High differentiation without integration is a bundle of switches — many parts, running in parallel, no unity. High integration without differentiation is a single wire — total unity, two possible states.

Even that refinement still aims at the wrong target. The goal was never a system that gets pushed around a lot. It is a system that can absorb and correct for a wide range of perturbations while remaining coherent. Internal state change is a side effect of regulation, not the point. Consciousness, if it tracks anything here, tracks regulatory competence rather than disturbability.

The Free Energy Principle is a formal home for that move. A system that persists as a distinct entity keeps its internal states inside a viable range, against a perturbing environment, through two channels: **perception** (update the model to match the world) and **action** (act on the world to match the model). Depth, on this view, is how much of a perturbation the system can anticipate and neutralize *before* it registers as a raw internal jolt. Paradoxically, a more competent system may look calmer, not more volatile. It is absorbing disturbance predictively.

---

## Homeostasis is not the interesting pole

Two kinds of regulation sit under that picture.

**Homeostasis** waits. Deviation is detected, then corrected toward a *fixed* setpoint. Reactive, after the fact.

**Allostasis** (Sterling) moves the setpoint first. The target is anticipated, context-dependent, allowed to shift. Proactive.

The hypothesis that follows is not “more allostasis is more consciousness” as a fixed amount. It is that the measure is the *range* of allostatic versus homeostatic regulation a system can cover — how much of its self-regulation is anticipatory, context-sensitive, and setpoint-shifting rather than fixed and reactive.

That range decomposes into three pieces, and they multiply rather than add, for the same reason differentiation and integration do: each is near-useless without the others.

- **Horizon** — how far ahead the system anticipates and acts before a perturbation hits. A reflex after deviation is low. Perception-driven action taken well before impact is high.
- **Retargetability** — whether the setpoint itself can move, across how many variables, independently. A fixed setpoint is low. Many setpoints shifting independently is high.
- **Context-sensitivity** — how much of the space of possible contexts actually drives the retargeting. A setpoint that shifts on a clock is low. A setpoint that shifts from a rich combinatorial read of context is high.

Horizon without retargetability is just a longer reflex. Retargetability without context is a clock. Independent parallel regulators, summed, inflate the score without unification — the bundle of switches again. System-level range has to be integration-weighted, or you are back to counting parts.

Where, then, does experience itself come from?

Three readings present themselves. Experience as the *sum* of both regulatory systems’ activity collapses back into dynamism, which was already rejected. Experience as the *seam* where the system is negotiating between homeostatic and allostatic control is possible, but thinner in the literature. The strongest candidate, with the best support (Friston, Clark, Seth), is the residual: experience as leftover mismatch between what the allostatic model anticipated and what actually occurred.

That predicts something ordinary. Experience should be most vivid at surprise and violation, and thin out as prediction improves. Habituation is the residual shrinking.

It is a trigger condition, a correlate — not an explanation of why any information processing should feel like anything. The hard problem stays open. The regulatory claim does not need it to close.

Given an internal model producing an expectation \(\hat{E}_t\), and an observed outcome \(O_t\):

\[
r_t = \mathrm{distance}(\hat{E}_t,\, O_t)
\]

That is the leftover: the part of the perturbation the model failed to anticipate. Moment to moment, the “conscious-like” signal is not raw disturbance. It is residual-after-anticipation. In the trait sense, capacity is allostatic range. A highly allostatic system should show large range and *small typical leftover*. It is built to absorb most perturbation predictively. Residual appears mainly at the edges, where the model is under-fit.

---

## What a volatility weight was standing in for

This is where the volatility problem lives.

A common way to stop a memory — or a continually learning model — from treating every fact as equally overwriteable is to attach a volatility to each *kind* of thing. Personality is stubborn. Mood is cheap. A job sits in between. In continual learning, the extreme homeostatic version is a fixed elastic penalty on old weights: one setpoint, reactive, no context. A domain volatility scalar is already a step toward allostasis. It is context-informed. It is still a slow prior, computed retrospectively from history, not from live surprise. An escalation rule driven by that scalar is a coarse retargeting: one context signal, no online residual.

The philosophical model treats two signals as distinct. A single volatility number conflates them.

1. A slow, domain-level **prior** about how noisy this channel usually is — allostatic, but not reactive to the current state.
2. A fast, per-step **residual** — live mismatch between expectation and observation.

A domain can be historically volatile while currently well-predicted. Under one scalar, that case is indistinguishable from a domain that is both historically *and* currently volatile. The model says they should be treated very differently. High prior, low leftover: the channel is usually messy, but right now the model is absorbing it. High prior, high leftover: the channel is messy *and* this step was not priced in.

That is the whole gap. Volatility-as-prior answers how readily this *kind* of memory should move in general. Leftover surprise answers whether this step was a regime change or the noise already budgeted for. One number cannot do both jobs.

The natural repair is not “a better volatility.” It is to let the trigger see both, and — more structurally — to let a channel *move between modes* rather than carry one weight for its lifetime. Sustained low residual, and the channel settles into homeostatic protection. Rising residual, and it reopens into allostatic plasticity. That is range, not a fixed allostatic amount: homeostasis and allostasis as a dynamic property of a domain, which is what Part 1.6 actually claimed.

The diagnostic that would show this is doing something real, rather than adding tunable capacity, is forgetting-after-recency: a domain that was stable for a long time and then suddenly shifted character. A static volatility cannot see the shift. A residual, or a slow readout of recent residual, can. If the win appears *specifically* there, and not uniformly across the board, the leftover term is earning its keep.

Two further distinctions follow from the same definitions, and they are easy to miss if residual is implemented as “how different is the new observation from the stored one.”

Surprise, in the model, is leftover after anticipation. Ordinary noise on a volatile channel is *expected*; the prior should widen what counts as normal, not license every difference as a jolt. Same words after a stretch of similar asides are not surprising. The expectation has moved. The residual goes to zero. Catching a pile of weak evidence is a different job from asking whether *this step* was unexpected. The first is horizon — accumulation over time, with decay, so that identical evidence at different spacing does not count the same. Daily mentions can reopen a belief; the same mentions spread thin can decay instead of accumulating. A counter that only ever rises has no route back to settled. Range requires a way home.

And dropping the prior entirely, so that only residual remains, is not free. The prior was doing work as insurance when the *kind* of thing was guessed wrong: a very-stable fact, misread as something more changeable, then contradicted by weak evidence. Residual-only opens that fact. The double use of volatility — in the evidence and in the bar — looks like a defect when a career actually changes, and like a safeguard when a trait is misfiled. It is a genuine trade. Blending the two formulas does not inherit both wins. Partial prior in the trigger is still prior in the trigger; the recency-shift case is a cliff, not a gradient. The remaining honest move is a switch: take the residual path when the world is telling you it moved — an explicit correction, or leftover that is actually unexpected — and keep the prior otherwise.

None of that requires a story about inner life. It is the same regulatory split, applied to the problem of things that are allowed to change at different rates, and of changes that arrive as a jolt versus as weather.

---

## What this is not

The chain is a structural analogy. Homeostasis, allostasis, differentiation, and integration show up anywhere a system must balance responsiveness against stability under a changing environment — thermoregulation, memory consolidation, continual learning. None of it is evidence that a memory system, or a learning rule, possesses experience. The consciousness hypothesis and the engineering improvement are separable. The engineering claim can be tested. The consciousness claim remains philosophical.

What the analogy forbids, if you take it as a picture of the volatility problem, is fairly sharp.

It forbids treating disturbability as the goal. It forbids using one number for a slow prior about a channel and a fast leftover at a step. It forbids calling surprise the distance to what is stored, rather than to what was anticipated. It forbids assigning a domain one plasticity for its lifetime. It forbids blending two control laws that win on different failures and expecting the average to keep both.

A highly allostatic system, on this view, is not the one that moves the most. It is the one with range enough to absorb most of the world in advance, and leftover only at the edges where the model still does not fit.

---

*The argument above follows [the research log](allostatic-consciousness-to-voltmem.md). Sources it rests on: Tononi & Edelman, “Consciousness and Complexity” (1998); Tononi, Boly, Massimini & Koch on IIT (2016); Casali et al. on the Perturbational Complexity Index (2013); Aaronson’s exchange with Tononi (2014) and Doerig, Schurger & Herzog (2021) as counterpoint; Peter Sterling, “Allostasis: a model of predictive regulation” (2012); Friston on the free-energy principle (2010); Andy Clark, *Surfing Uncertainty*; Anil Seth, *Being You*; Barrett & Simmons on interoceptive predictions (2015); Damasio, *Self Comes to Mind*; Kirkpatrick et al. on EWC (2017).*
