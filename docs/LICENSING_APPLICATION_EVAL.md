# Licensing application eval (VoltMem)

Motivation from sibling repo **licensing-collective-selves**: D3d **anchors**
reduce simulated fragmentation on the antiwork twin; apply a conservative
**anchor bar** on verified generic slots at write time.

Cross-ref: `../licensing-collective-selves/docs/voltmem-application-eval.md`

## Hypothesis

On the structured join battery, requiring explicit `replaces` + change marker
to UPDATE **anchored** generic slots (`occupation`, `employer`, `residence_city`)
reduces false merges without dropping must-link pairs vs conservative baseline.

## Run

```bash
.venv/bin/python experiments/licensing_anchor_eval.py
```

Pass: held-out **0 false merges** and **must-link ≥ conservative baseline**.

### Result (2026-08-27, heuristic extractor held-out)

| Mode | must-link | false merges |
|------|-----------|--------------|
| Conservative baseline | 5/28 | 0 |
| Anchor bar | 5/28 | 0 |

**Overall: PASS** (non-regression). Unit coverage: `test_anchored_attributes_require_replaces_and_change_marker`.

## Scope

- In scope: Battery H / structured_join held-out split  
- Out of scope: Reddit data, live modularity gate (future policy flag)
- Next: LLM `structured_join_eval.py` path; product `remember()` policy flag only after that
