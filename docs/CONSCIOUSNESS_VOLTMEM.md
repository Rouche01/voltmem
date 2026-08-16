# How a theory of leftover surprise changed a memory layer

## Consciousness Might Not Look Like Anything

If you are looking for consciousness in behavior, you may be looking at the wrong thing.

The system most in contact with the world need not flinch, chatter, or otherwise announce that something happened. It may look, from the outside, as if nothing did.

The obvious claim is the opposite: the most conscious being is the one whose internals get shoved around the hardest. Raw sensitivity. Volatility as contact. A thermostat kills that story. Turn the heat on and its state flips. Nobody thinks it is more conscious than a rock. Disturbability is cheap.

What matters is the structure of the response — a large repertoire of distinguishable states, produced by one system you cannot carve into independent parts without losing something. A bundle of switches is many parts and no unity. A single wire is total unity and two possible states. Neither is interesting.

Even that is not quite the target. The interesting system is not the one that gets pushed around a lot. It is the one that can absorb a wide range of perturbations while remaining itself — and, more than that, can see them coming so the jolt never fully arrives. Internal change is a side effect of regulation, not the point. Paradoxically, a more competent system may look calmer, not more volatile. It is absorbing disturbance predictively. From the outside there may be nothing to watch.

That is the sense in which consciousness might not look like anything.

I did not set out to settle that question. I set out to stop an agent from treating a career change like noise and a passing mood like a personality transplant. The philosophy showed up anyway. Then it earned its keep.

---

## Two kinds of regulation

**Homeostasis** waits. Deviation is detected, then corrected toward a fixed setpoint.

**Allostasis** moves the setpoint first. The target is anticipated, context-dependent, allowed to shift. Blood pressure before you stand up. The coat before the cold. Peter Sterling’s version is blunt: waiting for error and then fixing it is inefficient.

The interesting quantity is not how much allostasis a system has as a fixed amount. It is the *range* it can cover: how far ahead it can act, how many setpoints can move independently, how much of the space of possible contexts actually drives those moves. Each of those is near-useless without the others. Horizon without retargetability is a longer reflex. Retargetability without context is a clock.

Experience, on the strongest reading of the same literature, is not the regulation itself. It is the leftover: the part of the world the model failed to anticipate. Prediction error after the coat was already on. The same stimulus should feel vivid at first and thin out as the prediction improves. Habituation is the residual shrinking.

That is a trigger condition, not a solution to the hard problem. I am not claiming a Python library is conscious. I am claiming the split is load-bearing for any memory that has to stay current without coming apart.

---

## One number was doing two jobs

[VoltMem](https://github.com/Rouche01/voltmem) already had a prior about how stubborn each *kind* of fact should be. Personality locks down. Mood is cheap. A job sits in between. That prior is a slow allostatic signal: *this channel is usually like that.* It is not a reading of what is happening now.

A live residual would ask a different question. Given what we already expected — including that this channel is usually noisy, or usually quiet — did *this* observation arrive in a way we had not already priced in?

A domain can be historically messy and currently well-predicted. Under one scalar those cases are the same. They should not be.

The old overwrite rule charged that stubbornness twice. Being a “rarely changes” kind of fact both shrank the evidence *and* raised the bar. For a job, a clear “I retrained as a nurse” could fail to overwrite “I was a data analyst.” It was not being careful. It was double-counting the same caution.

So we built an allostatic mode: drop volatility from the evidence score, and let recent leftover surprise temporarily lower the bar for memories that are going through something.

The first ablation was rude. The entire career-change win was taking volatility out of the score. It was a cliff, not a blend. A little volatility left in the formula lost the case entirely.

That looked like a free improvement until we measured the cost.

---

## The defect was insurance

The classifier that guesses “what kind of fact is this?” is about 84% accurate. When it files a personality trait as something more changeable, and a weak comment arrives, allostatic mode overwrites. The double charge we had just called a bug still discounts the evidence, and the trait survives.

At the real error rate, allostatic lost 1.1 points of accuracy and produced 20% more false updates. Every extra overwrite had the same shape: a very-stable fact, mislabeled, contradicted by weak evidence.

Removing the double charge fixes career changes and breaks mislabeled traits. It is a genuine trade. You cannot slide volatility halfway back in; we already measured that cliff. The remaining honest move is a *switch*, not a mix.

Most of the time the system stays homeostatic: this is the kind of fact it is; I need a lot of proof. It goes allostatic when the world is actually telling it the setpoint moved — a clear correction, or a leftover it did not already expect.

That switch is now the default in VoltMem 0.4.0. It is called `composite`. It matches the cautious law’s false-update rate and still catches the explicit career change. Allostatic stays available if you pass a trusted domain label and want the residual path all the time. Homeostatic stays available if you want the insurance always on.

A theory of range, not amount, said a channel should be able to move between modes rather than carry one weight for its lifetime. The gate is that claim, implemented as a predicate instead of a personality.

---

## We were measuring the thermostat

The live shakiness meter was an average of raw contradiction — how different the new sentence was from the stored one. Every method in the notes treats surprise as leftover mismatch *after anticipation*. We were scoring the flinch.

If someone has been casually mentioning a new job for two weeks, another casual mention is not surprising. If they have been quiet for months and then say the same words, that *is* surprising. Same sentence. Different leftover.

We shipped that definition. Each memory keeps a running “what mismatch size is normal,” widened by how noisy the domain usually is. Surprise is distance from that prediction, not from the stored sentence. Evidence still uses how contradictory the sentence is. Surprise uses how unexpected that contradiction was.

The career change said clearly still works. It never needed the surprise term. It wins by dropping the double charge.

The sixteen weak asides stopped catching live. After a few similar mentions the system *expects* them. Leftover goes to zero. The bar stays shut. That is the definition working, which is why it could not be the slow-burn fix. People do not rewrite “my job” on every aside. They notice the pattern later.

We had been celebrating a hairline. An average of raw contradiction had been sitting 0.0002 from the trigger; stretching the half-life from two weeks to a month shoved it over. That is not surprise. That is a constant wearing a costume.

The pile belongs overnight. Sixteen daily weak mentions stay quiet on the write path, then consolidation actually supersedes the stored job. The same sixteen spread monthly do not. A core preference does not yield. Same evidence, same count, only the pace differs. That is horizon: identical evidence at different spacing must not count the same. An earlier version keyed shakiness to a lifetime counter that could only ratchet open. A long-lived memory would have grown permanently easier to overwrite with age. Time decay was the route back to settled. Range requires a way home.

So 0.4.0 has three timescales, not one knob:

- **Live leftover** — was *this step* unexpected? Quiet is correct.
- **The gate** — explicit correction, or a leftover we did not already expect, opens the easy-update path. Otherwise keep the insurance.
- **Sleeptime** — has the *belief* that the old fact still holds actually moved, over a window? Daily weather can shift a job overnight. A monthly drip cannot. A deep preference does not erode.

That is the sleeptime compute idea from a [previous post](https://dev.to/rouche01/the-maintenance-window-i-didnt-know-i-was-running-51kk), now with a detector that sees spacing instead of a counter that never forgets.

---

## Then we found the thing sitting in front of all of it

None of the overwrite work runs if the new sentence never finds the memory it contradicts.

The matcher and the overwrite rule are sequential, not alternatives. Homeostatic vs allostatic only starts after a match. A miss inserts a duplicate. A false pair *does* consult the overwrite law — on the wrong memory.

Through the default `remember()` path, keyword similarity routed 22% of update cases. A real embedder routed 78% with perfect labels, 56% with the shipped classifier. Decision error *once routed* was about six points. We had been arguing about the six-point layer. Two of the cases that never linked even with embeddings were the exact career-change and goal-change probes allostatic was built to recover.

The obvious repair is to lower the link bar. That repair is forbidden.

“User is proficient in Python” versus “User is proficient in Japanese” scores as a *stronger* match than “I was a data analyst” versus “I retrained as a nurse.” The false merges outrank the true links. No cutoff can work. Bag-of-words and sentence embeddings both encode topic, not entity identity. They cannot tell two facts about family members from one fact restated.

A duplicate is recoverable. Both facts stay retrievable; a later pass can reconcile them. A false merge treats two different facts as one and loses a true memory. Lowering a bar converts the cheaper error into the expensive one.

Worse: under a threshold ladder, *improving the classifier makes data loss worse*. Correct labels put two distinct facts in the same slot, where the ladder can merge them. Misclassification was accidentally protecting memories by scattering them.

The architecture that survived is recall wide, decide narrow. Similarity is allowed to fetch candidates. It is not allowed to decide “same fact.” A conservative local model looking at the pair — same subject? same *question*, never the same answer? — scored 49/56 on held-out pairs with **zero** false merges. A hosted model scored three pairs higher and paid for it with two irreversible losses. Under the weighting we used everywhere else, that is a bad trade. We kept the conservatism.

A 14B call is about five seconds. Fine overnight. Too slow inside an interactive `remember()`. The worst case of deferring it is a temporary duplicate. That is what makes deferral acceptable rather than a compromise.

So the shipped write path is millisecond and conservative. Heuristic subject/attribute frames we trust may join. Grey frames insert as twins. Overnight, the local verifier reconciles them. Live `remember()` does not wait for the 14B unless you ask it to.

The philosophy said leftover only appears at the edges where the model is under-fit. In the library, leftover only appears at all if the observation finds the slot. The matcher is not a theory of consciousness. It is the condition under which the control law, and therefore any residual, is even computed.

---

## What 0.4.0 actually is

The consciousness writeup was a searchlight, not a control law. It made us split expected noise from unexpected leftover, refuse raw dynamism, refuse a lifetime weight, refuse a blend, and notice that a miss never consults the overwrite rule. The range formula did not have to be true for those moves to pay. Neither did “this is what experience is.”

What shipped:

**Surprise means unexpected leftover**, not how different the sentence is. A predicted weak stream no longer impersonates a regime change.

**Composite overwrite** — cautious by default; allostatic only on an explicit correction or a learned unexpected residual. Career changes said clearly get through. Mislabeled traits keep the insurance.

**Sleeptime for the pile** — daily weak evidence can move a belief overnight; the same evidence dripped monthly does not. Very-stable domains do not erode.

**Search widely, decide narrowly** — embeddings for recall, a conservative local verifier for precision, deferred to sleeptime. Duplicates are okay. Silent overwrites are not.

---

## If you're building persistent agent memory

Three distinctions paid rent. I would steal those and leave the rest.

**Contradiction is not surprise.** Distance from the stored fact is an observation. Surprise is distance from what you had already predicted, scaled by how noisy that channel usually is. If you average raw mismatch until it crosses a bar, you will eventually "catch" a slow change by sitting on a hairline. That is not a capability.

**A switch, not a blend.** If two control laws win on different failure modes, mixing their formulas can lose both. Gate on the kind of evidence. Keep the insurance for weak noise on a guessed label. Spend the plasticity on a clear correction or a leftover you did not expect.

**The matcher is in front of the overwrite rule.** Improving when to update a fact does nothing for a sentence that never found it — and does the wrong thing for a sentence that found the wrong one. Rank topic similarity and entity identity separately. Weight errors by whether they can be undone. If your "better embedder" starts deleting stored facts, you did not improve matching. You changed which mistake you make.

A thermostat looks alive. A system that saw the perturbation coming may look like nothing happened. VoltMem’s write path is now allowed to look like that on purpose: millisecond, quiet, twins instead of guesses. The work happens at the edges, and overnight.

I still do not think that makes it conscious. I think it makes it less of a thermostat.
