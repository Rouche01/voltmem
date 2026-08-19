# Zenodo submission — Prior versus Residual

Companion preprint to [doi:10.5281/zenodo.21962419](https://doi.org/10.5281/zenodo.21962419). Submit as a **new** Zenodo record (second DOI), not a version of the first paper.

## Artifacts

| Item | Path / URL |
|------|------------|
| PDF (rebuild before upload) | `paper/allostatic-memory-control.pdf` |
| Editable source | `paper/build_allostatic_paper.py` |
| Prose source | `paper/allostatic-memory-control.md` |
| Code | https://github.com/Rouche01/voltmem (tag / package 0.4.0) |
| This record | https://doi.org/10.5281/zenodo.22019047 |
| Cites | https://doi.org/10.5281/zenodo.21962419 |

## Rebuild PDF

```bash
.venv/bin/python paper/build_allostatic_paper.py
```

## Suggested metadata

**Resource type:** Preprint

**Title:** Prior versus Residual: Homeostatic and Allostatic Control Laws for Agent Memory Updates

**Creator:** Emate, Richard · Independent

**Related works:** Cites `10.5281/zenodo.21962419`. Is supplemented by `https://github.com/Rouche01/voltmem`.

**License:** CC BY 4.0

**Description (abstract):**

Agent memory systems that scale overwrite protection by a domain volatility prior Vd treat two different quantities as one number: a slow belief about how often a kind of fact changes, and a fast residual about whether this observation was unexpected. We separate them. A homeostatic law charges Vd in both the evidence score and the threshold. An allostatic law drops Vd from the score and scales the threshold by leftover surprise rt = |Mt − Êt| / σt, the distance from predicted mismatch rather than from stored text. A composite gate uses the allostatic law only for an explicit high-mismatch correction or an unexpected residual against a learned Ê; otherwise it keeps the homeostatic insurance.

On scripted probes with oracle domain and mismatch, dropping Vd from the score recovers explicit recency-shift (entrenched career change) as a cliff at exponent p=0, not a blend. The same drop produces 20% more false updates under the classifier's real error structure, because the double Vd charge was insurance against a mislabeled stable trait plus weak evidence. Composite matches homeostatic's false-update rate (94.9%, 0.93 false updates / 18) while keeping the recency-shift win. Defining surprise as leftover after anticipation makes a predicted weak stream go quiet; catching that stream is a sleeptime job on time-decayed belief mass, not a live EMA of raw mismatch. Sixteen daily weak mentions supersede overnight; the same sixteen monthly do not.

The overwrite law is only reached after a match. Through remember(text), decision error once routed is about six points; similarity and linking dominate. Topic similarity is non-separable for must-link versus must-not-link pairs. Two-stage recall-then-verify with a conservative local model takes irreversible errors to zero on a combined update+coexist harness. We do not claim a public-benchmark win. We claim a measured decomposition: prior and residual are different jobs; a switch beats a blend; linking sits in front of both laws. Results use VoltMem 0.4.0. Companion to doi:10.5281/zenodo.21962419.

## After a reserved DOI

Printed on the PDF: `10.5281/zenodo.22019047`. Replace the file on the same draft, then Publish. Do not delete the draft.
