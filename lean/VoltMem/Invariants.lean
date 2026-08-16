/-
`observe()` as a labeled transition. Matching sits in front: a miss never
consults `E > θ`. Confirm and log are recoverable; audit is the irreversible
write.
-/

namespace VoltMem.Invariants

inductive Action where
  | insert
  | confirm
  | log
  | audit
  deriving DecidableEq, Repr

def irreversible : Action → Bool
  | .audit => true
  | _ => false

/-- Python `mismatch_magnitude < 0.15` is a confirmation, not a conflict. -/
def confirmCutoff : Rat := 3 / 20

/--
Inputs to the write path after candidate selection.

* `miss` — no candidate. The control law is not consulted.
* `hit` — a candidate exists. `escalate` is the `E > θ` (plus overrides)
  decision. `force` is `force_update`.
-/
inductive ObserveInput where
  | miss
  | hit (M : Rat) (escalate force : Bool)

def observeAction : ObserveInput → Action
  | .miss => .insert
  | .hit M escalate force =>
    if force then .audit
    else if M < confirmCutoff then .confirm
    else if escalate then .audit
    else .log

theorem miss_never_audits : observeAction .miss ≠ .audit := by
  decide

theorem miss_inserts : observeAction .miss = .insert := rfl

/-- A miss has no escalate bit. The law is not an input. -/
theorem miss_independent_of_law :
    observeAction .miss = .insert := rfl

theorem only_audit_is_irreversible (a : Action) :
    irreversible a = true ↔ a = .audit := by
  cases a <;> simp [irreversible]

theorem confirm_is_recoverable (M : Rat) (esc : Bool)
    (h : M < confirmCutoff) :
    irreversible (observeAction (.hit M esc false)) = false := by
  simp [observeAction, irreversible, h]

theorem log_is_recoverable (M : Rat) (hM : ¬ M < confirmCutoff) :
    irreversible (observeAction (.hit M false false)) = false := by
  simp [observeAction, irreversible, hM]

theorem force_audits (M : Rat) (esc : Bool) :
    observeAction (.hit M esc true) = .audit := by
  simp [observeAction]

theorem force_is_irreversible (M : Rat) (esc : Bool) :
    irreversible (observeAction (.hit M esc true)) = true := by
  simp [observeAction, irreversible]

end VoltMem.Invariants
