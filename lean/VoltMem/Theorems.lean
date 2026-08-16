/-
Paper-facing names for the machine-checked claims.
-/

import VoltMem.Algebra
import VoltMem.ControlLaw
import VoltMem.Invariants

namespace VoltMem.Theorems

open VoltMem.Algebra
open VoltMem.ControlLaw
open VoltMem.Invariants

/-- Prior and residual are different jobs: `V` cancels in the homeostatic product. -/
theorem prior_is_a_double_charge
    (res V θ0 L : Rat) (hV : V ≠ 0) :
    E_homeostatic res V * theta θ0 V L 1 = res * θ0 * L :=
  homeostatic_double_charge res V θ0 L hV

/-- A residual in the job-band gap is allostatic-only. Blend is not an interpolation. -/
theorem switch_not_blend
    (res : Rat) (hLo : (1 : Rat) / 2 < res) (hHi : res ≤ 5 / 3) :
    (E_allostatic res > theta theta0 V_job 1 1) ∧
      ¬ (E_homeostatic res V_job > theta theta0 V_job 1 1) :=
  job_gap_instance res hLo hHi

/-- Composite opens on explicit high-`M`, independent of leftover. -/
theorem composite_explicit (learnedRes : Bool) :
    compositeOpens true learnedRes = true :=
  composite_explicit_opens learnedRes

/-- Composite stays closed when neither gate fires (fresh item, weak `M`). -/
theorem composite_insurance : compositeOpens false false = false :=
  composite_fresh_stays_closed

/-- Monthly weights strictly decay. Daily-vs-monthly mass is the Float oracle. -/
theorem spacing_weights_decay {n m : Nat} (h : n < m) :
    halfPow m < halfPow n :=
  halfPow_antitone h

/-- Stable domains need a heavier sleeptime pile. -/
theorem stable_needs_heavier_pile
    (k V1 V2 : Rat) (hk : 0 < k) (hV : 0 < V1) (hLt : V1 < V2) :
    k / V2 < k / V1 :=
  belief_bar_antitone k V1 V2 hk hV hLt

/-- No candidate ⇒ insert. The overwrite law is not consulted. -/
theorem matcher_in_front : observeAction .miss ≠ .audit :=
  miss_never_audits

/-- Silent overwrite is the expensive error; duplicates are not. -/
theorem duplicates_are_recoverable (a : Action) :
    irreversible a = true ↔ a = .audit :=
  only_audit_is_irreversible a

end VoltMem.Theorems
