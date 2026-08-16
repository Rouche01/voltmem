# Same Words, Different Surprise

## How a theory of consciousness changed VoltMem's overwrite rule — and why that still wasn't the product problem

I didn't set out to write a theory of consciousness.

I set out to stop an agent from treating a career change like noise and a passing mood like a personality transplant. That work became [VoltMem](https://github.com/Rouche01/voltmem): a memory layer that scales protection by how fast each *kind* of fact actually changes. Personality traits lock down. Locations update. Current tasks evaporate. The math is a [control knob on the stability–plasticity tradeoff](https://dev.to/rouche01/i-built-a-memory-layer-for-llm-agents-that-knows-which-facts-go-stale-1mg5), not a free-lunch accuracy booster.

Then a different question showed up, and it refused to stay philosophical.

If a system is more "alive" the more the world can shove it around, a thermostat beats a person. Its state flips hard when you perturb it. Nobody thinks that makes it conscious. What matters is not how violently the internals move. It is whether the system can *absorb* a wide range of perturbations while remaining itself — and, more than that, whether it can anticipate them so the jolt never fully arrives.

That is the move from homeostasis to allostasis. Homeostasis waits for deviation, then corrects toward a fixed setpoint. Allostasis shifts the setpoint in advance, from context. Peter Sterling's version is blunt: waiting for error and then fixing it is inefficient. A body that is any good at staying alive has already moved the target.

On this reading, leftover surprise — the part of the world the model failed to anticipate — is the interesting signal. The same sentence can be shocking after a quiet stretch and boring after two weeks of similar asides. Same words. Different surprise.

I spent a stretch of VoltMem 0.4.0 treating that claim as an engineering assumption, not a metaphysics. The assumption survived contact with the batteries. The product problem it revealed was sitting one layer earlier, and no amount of better surprise math could reach it.

---

## The thermostat, then the career change

VoltMem already had a prior about how stubborn each kind of fact should be. Call it \(V_d\): low for "I am an introvert," high for "I am debugging the payment flow." When a new sentence disagrees with a stored one, the library scores the disagreement and checks it against a bar. Strong evidence from a trusted source scores high. Facts you have confirmed many times raise the bar.

The old rule charged that stubbornness twice. Being a "rarely changes" kind of fact both shrank the evidence *and* raised the bar. For a job — which does change, just not every Tuesday — a clear "I retrained as a nurse" could fail to overwrite "I was a data analyst." It was not being careful. It was double-counting the same caution.

A theory of allostasis says those two uses of volatility are different jobs and must not share a multiplier:

- A slow prior about how noisy this *kind* of channel usually is.
- A fast residual: did *this* observation arrive in a way we were not already expecting?

A domain can be historically messy and currently well-predicted. One scalar cannot see that. Continual-learning instinct says "volatile ⇒ stay plastic." Predictive-processing instinct says the opposite: if you already expected noise of this size, it is not a reason to reopen the memory. Both instincts are valid. They are not the same lever.

So we built an allostatic mode: drop volatility from the evidence score, and let a shakiness meter — recent leftover surprise — temporarily lower the bar for memories that are going through something.

The first ablation was rude. The entire career-change win was taking volatility out of the score. The shakiness meter, at that point, changed no outcome at any setting. And the win sat on a cliff: a little volatility left in the score lost the case entirely. Partial blending was dead on arrival.

That looked like a free improvement until we measured the cost.

---

## Insurance that looked like a bug

The classifier that guesses "what kind of fact is this?" is about 84% accurate. When it files a personality trait as something more changeable, and a weak comment arrives, allostatic mode overwrites. Homeostatic mode — the double charge we had just called a defect — still discounts the evidence, and the trait survives.

At the classifier's real error rate, allostatic lost 1.1 points of accuracy and produced 20% more false updates. Every extra overwrite had the same shape: a very-stable fact, mislabeled, contradicted by weak evidence.

The double charge was insurance. Removing it fixes career changes and breaks mislabeled traits. It is a genuine trade, not a free win. You cannot slide volatility halfway back into the score; we already measured that cliff. The remaining honest compromise is a *switch*, not a blend: stay cautious unless the world is actually telling you it moved.

Before that switch could mean anything, we had to stop lying about surprise.

---

## The trap in detecting surprise

The live shakiness meter was an average of raw contradiction — how different the new sentence was from the stored one. Every method in the consciousness notes treats surprise as leftover mismatch *after anticipation*. We were measuring the thermostat.

If someone has been casually mentioning a new job for two weeks, another casual mention is not surprising. If they have been quiet for months and then say it, that *is* surprising. Same words. Different leftover.

The first test was small. Keep a running "what mismatch size is normal for this memory," widen that normal by how noisy the domain usually is, and score the new observation against the prediction, not against the stored sentence. Evidence still uses how contradictory the sentence is. Surprise uses how unexpected that contradiction was.

Three old batteries, one pass bar:

| What we asked | What happened |
|---|---|
| Career change said clearly | Still caught. It never needed the surprise term. It wins by dropping the double charge. |
| Career change as sixteen weak asides | Stopped catching live. After a few similar asides the system *expects* them, leftover goes to zero, the bar stays shut. |
| Wrong label + weak noise on a stable fact | Unchanged. Those overwrites happen on a *fresh* item, before surprise has a vote. |

The sixteen-asides "win" we had been celebrating was a hairline. The average of raw contradiction had been sitting 0.0002 from the trigger; stretching the half-life from two weeks to a month shoved it over. That is not surprise. That is a constant wearing a costume.

The new definition did the right thing, which is why it could not be the Battery D fix. Same words after two weeks of asides *should not* reopen a memory on the hot path. People do not rewrite "my job" on every aside. They notice the pattern later.

That split of jobs is the whole remaining design:

1. **Live path** — was *this step* unexpected? Quiet is correct.
2. **Clear statement** — drop the double charge, or gate into that path.
3. **Pile of weak asides** — add up whether the *belief* that the old fact still holds has actually moved. Do that overnight. Do not reopen the bar because an average crept up.
4. **Wrong label** — keep the insurance unless the gate says this is a real surprise.

The second test was the switch. Composite mode stays homeostatic unless the statement is explicit and strongly contradictory, or leftover surprise is high against a *learned* normal. A fresh item has no anticipation, so the first weak blip cannot open the easy-update path. That is the insurance.

Composite matched homeostatic's false-update rate, still caught the clear career change, and left the daily asides quiet live. As of 0.4.0 it is the default.

The third test put the pile where it belongs. Sixteen daily weak mentions stay quiet on the write path, then overnight consolidation actually supersedes the stored job. The same sixteen spread monthly do not. A core preference does not yield. Same evidence, same count, only the pace differs — which is the proof it is a capability rather than looser plasticity. An earlier version keyed shakiness to a lifetime mismatch counter that could only ratchet open. A long-lived memory would have grown permanently easier to overwrite with age. Time-decayed leftover, and time-decayed belief movement at sleeptime, were the precondition for trusting any of this.

---

## Then we found the larger problem

None of the overwrite work runs if the new sentence never finds the memory it contradicts.

The matcher and the overwrite rule are sequential, not alternatives. Homeostatic vs allostatic only starts after a match. A miss inserts a duplicate. A false pair *does* consult the overwrite law — on the wrong memory.

Through the default `remember()` path, with no domain handed in, keyword similarity routed 22% of update cases. A real embedder routed 78% with oracle labels, 56% with the shipped classifier. Decision error *once routed* was about six points. We had been arguing about the six-point layer.

Worse: the cases allostatic was built to recover — explicit career change, explicit goal change — were among the pairs that never linked. The observation never found the memory, so the escalation law was never consulted. Batteries that passed `domain=` could see the win. The default path could not.

The obvious repair was to lower the link bar. The next battery exists to forbid that move.

"User is proficient in Python" versus "User is proficient in Japanese" scores as a *stronger* match than "I was a data analyst" versus "I retrained as a nurse." The false merges outrank the true links. No cutoff can work: raise it and you miss the career change; lower it and you destroy a real skill. Bag-of-words and sentence embeddings both encode topic, not entity identity. They cannot tell "two facts about family members" from "one fact restated."

A duplicate is recoverable. Both facts stay retrievable; a later pass can reconcile them. A false merge treats two different facts as one and loses a true memory. That weighting is the whole design. Lowering a bar converts the cheaper error into the expensive one.

Embeddings roughly doubled true-link recall on fresh pairs at no extra false-merge *count* — and then raised the *severity*. Under keywords, bad merges mostly discarded the incoming fact. Under embeddings they superseded the stored one. Four deletions of existing memory is not a clean upgrade over five dropped arrivals.

The architecture that survived was the one everyone else already uses for record linkage, restated for a memory slot: recall wide, decide narrow. Similarity is allowed to fetch candidates. It is not allowed to decide "same fact." A conservative local model looking at the pair — same subject? same *question*, never the same answer? — scored 49/56 on held-out pairs with **zero** false merges. A hosted model scored three pairs higher and paid for it with two irreversible losses. Under the weighting we used everywhere else, that is a bad trade.

Prompt framing moved the hosted model from worse-than-nothing to the top of the table. The local model barely noticed the same paragraph. It was conservative about answering yes. That conservatism *is* the zero-false-merge column. We kept it.

A 14B call is about five seconds. Fine overnight. Too slow inside an interactive `remember()`. The worst case of deferring it is a temporary duplicate. That is what makes deferral acceptable rather than a compromise.

We tried the cheap middle anyway. A cross-encoder still thought Python/Japanese was a stronger match than analyst→nurse. We stopped; ranking inverted means no threshold will save you. File-each-fact-as-a-card-then-join — same person, same question — got the cells that used to fool lexical tricks, then still destroyed four memories when both sides filed the same generic drawer (`current_task`, `current_mood`, `current_manager`). Close, in the useless way: the join is right when the cards are specific, and unsafe when they are not.

So the shipped write path is millisecond and conservative. Heuristic subject/attribute frames that we trust (city, birth year, skill, occupation) may join. Grey frames insert as twins. Overnight, a local verifier reconciles twins. Live `remember()` does not wait for the 14B unless you ask it to.

One more perverse finding, because it is the kind that survives: under a threshold ladder, *improving the classifier makes data loss worse*. Correct labels put two distinct facts in the same slot, where the ladder can merge them. Misclassification was accidentally protecting memories by scattering them. Two-stage linking removes the incentive. The verifier never consults the domain label.

---

## What 0.4.0 actually is

The consciousness writeup was a searchlight, not a control law. It made us split expected noise from unexpected leftover, refuse raw dynamism, and notice that a miss never consults homeostatic vs allostatic. The \(H \times R \times C\) formula did not have to be true for those moves to pay. Neither did "this is what experience is."

What shipped:

- **Surprise means unexpected leftover**, not "how different is this sentence?" A predicted weak stream no longer impersonates a regime change.
- **Composite overwrite** — cautious by default; allostatic only on an explicit correction or a learned unexpected residual. Career changes said clearly get through. Mislabeled traits keep the insurance.
- **Sleeptime for the pile** — daily weak evidence can move a belief overnight; the same evidence dripped monthly does not. Very-stable domains do not erode.
- **Search widely, decide narrowly** — embeddings for recall, a conservative local verifier for precision, deferred to sleeptime. Duplicates are okay. Silent overwrites are not.

The philosophy and the engineering stay separable. The engineering claims were tested. The consciousness claim remains a structural analogy: thermoregulation, memory consolidation, and continual learning instantiate the same shape. None of this is evidence that VoltMem possesses experience. The hard problem is still open. The batteries do not care.

---

## If you're building persistent agent memory

Three distinctions paid rent. I would steal those and leave the rest.

**Contradiction is not surprise.** Distance from the stored fact is an observation. Surprise is distance from what you had already predicted, scaled by how noisy that channel usually is. If you average raw mismatch until it crosses a bar, you will eventually "catch" a slow change by sitting on a hairline. That is not a capability.

**A switch, not a blend.** If two control laws win on different failure modes, mixing their formulas can lose both. Gate on the kind of evidence. Keep the insurance for weak noise on a guessed label. Spend the plasticity on a clear correction or a leftover you did not expect.

**The matcher is in front of the overwrite rule.** Improving when to update a fact does nothing for a sentence that never found it — and does the wrong thing for a sentence that found the wrong one. Rank topic similarity and entity identity separately. Weight errors by whether they can be undone. If your "better embedder" starts deleting stored facts, you did not improve matching. You changed which mistake you make.

And if you notice that the same comment from a friend is alarming in January and invisible in March, you are not being inconsistent. You updated the prediction. The leftover shrank. That is the system working.

---

*[VoltMem on GitHub](https://github.com/Rouche01/voltmem) · [Original VoltMem writeup](https://dev.to/rouche01/i-built-a-memory-layer-for-llm-agents-that-knows-which-facts-go-stale-1mg5) · [Sleeptime compute post](https://github.com/Rouche01/voltmem/blob/main/docs/SLEEPTIME_COMPUTE_v3.md) · Built by [Richard Emate](https://github.com/Rouche01)*
