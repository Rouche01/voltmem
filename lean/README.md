# VoltMem Lean spec

Machine-checked control law for [VoltMem](https://github.com/Rouche01/voltmem).
Python `voltmem/scoring.py` stays the runtime. This package is the spec.

```
lean/
  VoltMem/Algebra.lean      Rat identities (double charge, gap, spacing weights)
  VoltMem/ControlLaw.lean   Float kernel matched to scoring.py
  VoltMem/Invariants.lean   observe() actions; miss never audits
  VoltMem/Theorems.lean     paper-facing names
  VoltMem/Oracle.lean       Battery C / E / D #guard cases
```

## Setup

```bash
# once: https://lean-lang.org/install/
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

Toolchain is pinned in `lean-toolchain` (`leanprover/lean4:v4.33.0`). No mathlib.

```bash
cd lean
lake build
lake exe voltmem_oracle
```

`#guard` in `Oracle.lean` fails the build if a named case flips. The same cases
are asserted against Python in `tests/test_lean_oracle.py`.

## What is proved

| Claim | Where |
|---|---|
| Homeostatic double-charges `V` (`E · θ = res · θ₀ · L`) | `Algebra.homeostatic_double_charge` |
| Allostatic charges `V` once, on the bar | `Algebra.allostatic_escalates_iff` |
| Job-band gap: residual can clear allostatic and miss homeostatic | `Algebra.job_gap_instance` |
| Composite is a switch, not an exponent | `ControlLaw.composite_is_switch` |
| Fresh/weak stays closed (Battery E insurance) | `Theorems.composite_insurance` + oracle `#guard` |
| Later evidence weighs less (`(1/2)^n`) | `Algebra.halfPow_antitone` |
| Stable domains need a heavier sleeptime pile | `Algebra.belief_bar_antitone` |
| No candidate ⇒ insert; law not consulted | `Invariants.miss_never_audits` |
| Only audit is irreversible | `Invariants.only_audit_is_irreversible` |

Empirical batteries (classifier accuracy, embedding recall, 14B verifier) stay
in `experiments/` and pytest. Consciousness / allostatic *range* is not a
theorem here.

## After changing `scoring.py`

1. Port the kernel change in `ControlLaw.lean`.
2. `lake build` — proofs and `#guard`s.
3. `python tests/test_lean_oracle.py` — Python still agrees.
