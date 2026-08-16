/-
Executable control law, matched to `voltmem/scoring.py`.

Float is the Python kernel. Algebraic identities (double charge, gap, spacing
weights) are proved over `Rat` in `VoltMem.Algebra`. This module is the
oracle: same constants, same composite switch, same θ-cap and cumulative
mismatch override.
-/

namespace VoltMem.ControlLaw

def alpha : Float := 0.6
def theta0 : Float := 0.15
def explicitOverrideM : Float := 0.85
def explicitMinVd : Float := 0.15
def explicitMaxVd : Float := 0.55
def explicitERatio : Float := 0.85
def cumulativeMismatchEscalate : Nat := 3
def sMin : Float := 0.25
def mismatchPrior : Float := 0.05
def sigmaFloor : Float := 0.08
def sigmaVdKappa : Float := 0.35
def residualZScale : Float := 3.0
def residualGate : Float := 0.50
def beliefShiftK : Float := 0.35
def beliefShiftMinVd : Float := 0.05
def surpriseHalflifeDays : Float := 30.0
def vExpHomeostatic : Float := 1.0
def vExpAllostatic : Float := 0.0
def vdFloor : Float := 1e-6

inductive Source where
  | explicitStatement
  | repeatedConfirmation
  | strongInference
  | weakInference
  | systemGenerated
  deriving DecidableEq, Repr, Inhabited

def Source.reliability : Source → Float
  | .explicitStatement => 1.0
  | .repeatedConfirmation => 1.2
  | .strongInference => 0.7
  | .weakInference => 0.4
  | .systemGenerated => 0.3

/-- Python `SOURCE_RELIABILITY["strong_inference"]` — cumulative floor. -/
def strongInferenceR : Float := 0.7

inductive Mode where
  | homeostatic
  | allostatic
  | composite
  deriving DecidableEq, Repr, Inhabited

/-- Resolved law after the composite gate (never `composite`). -/
inductive Law where
  | homeostatic
  | allostatic
  deriving DecidableEq, Repr, Inhabited

structure Slot where
  domainPrior : Float
  effectiveV : Float
  repetitionCount : Nat := 1
  mismatchCount : Nat := 0
  /-- `-1` = unlearned. Python `or -1.0` treats `0.0` as unlearned too. -/
  mismatchEma : Float := -1.0
  mismatchVar : Float := -1.0
  surpriseEma : Float := 0.0
  deriving Repr

def clamp01 (x : Float) : Float := max 0.0 (min 1.0 x)

/-- Sigmoid-ish `G` in `[0.1, 2]`, centred at 0. Matches `_g_factor`. -/
def gFactor (goalDelta : Float) : Float :=
  0.1 + 1.9 / (1.0 + Float.exp (-3.0 * goalDelta))

def residual (M R : Float) (C : Nat) (G : Float) : Float :=
  let c := Float.ofNat (max C 1)
  (M * R / Float.pow c alpha) * G

/-- Python: `(item.mismatch_ema or -1.0) >= 0` after falsy-0. -/
def learnedHatE (slot : Slot) : Bool :=
  decide (0.0 < slot.mismatchEma)

def expectedMismatch (slot : Slot) : Float :=
  if slot.mismatchEma ≤ 0.0 then mismatchPrior else clamp01 slot.mismatchEma

def mismatchSigma (slot : Slot) : Float :=
  let var := if slot.mismatchVar == 0.0 then -1.0 else slot.mismatchVar
  let empirical :=
    if var ≥ 0.0 then Float.sqrt (max var 0.0) else 0.0
  max sigmaFloor empirical + sigmaVdKappa * max slot.domainPrior 0.0

/-- Leftover surprise `r_t = min(1, |M − Ê| / (σ · 3))`. -/
def residualSurprise (slot : Slot) (M : Float) : Float :=
  let Mt := clamp01 M
  let z := (Mt - expectedMismatch slot).abs / mismatchSigma slot
  min 1.0 (z / residualZScale)

/-- Composite is a switch, not an exponent. -/
def resolveLaw (slot : Slot) (M : Float) (src : Source) (mode : Mode) : Law :=
  match mode with
  | .homeostatic => .homeostatic
  | .allostatic => .allostatic
  | .composite =>
    let Mt := clamp01 M
    if decide (Mt ≥ explicitOverrideM) && src == .explicitStatement then
      .allostatic
    else if learnedHatE slot && decide (residualSurprise slot Mt ≥ residualGate) then
      .allostatic
    else
      .homeostatic

def vExp : Law → Float
  | .homeostatic => vExpHomeostatic
  | .allostatic => vExpAllostatic

def usesModeScale : Law → Bool
  | .homeostatic => false
  | .allostatic => true

/-- `s(m)` with no time decay (oracle cases pass `surpriseEma` already decayed). -/
def surpriseModeScale (slot : Slot) : Float :=
  1.0 - (1.0 - sMin) * clamp01 slot.surpriseEma

def explicitThetaCap (Vd : Float) : Option Float :=
  if Vd < explicitMinVd || Vd > explicitMaxVd then none
  else some (Vd * explicitERatio)

def barV (slot : Slot) : Law → Float
  | .homeostatic => slot.effectiveV
  | .allostatic => slot.domainPrior

def escalationScore (slot : Slot) (M : Float) (src : Source)
    (goalDelta load : Float) (law : Law)
    (vExpOverride : Option Float := none)
    (modeScaleOverride : Option Bool := none) : Float × Float :=
  let R := src.reliability
  let C := slot.repetitionCount
  let Vd := slot.effectiveV
  let Mt := clamp01 M
  let G := gFactor goalDelta
  let p := vExpOverride.getD (vExp law)
  let scale := modeScaleOverride.getD (usesModeScale law)
  let E := residual Mt R C G * Float.pow (max Vd vdFloor) p
  let Vbar := max (barV slot law) vdFloor
  let θ := theta0 * (1.0 / Vbar) * load *
    (if scale then surpriseModeScale slot else 1.0)
  (E, θ)

def escalationDecision (slot : Slot) (M : Float) (src : Source)
    (goalDelta load : Float) (mode : Mode)
    (vExpOverride : Option Float := none)
    (modeScaleOverride : Option Bool := none) : Bool × Float × Float × Law :=
  let Mt := clamp01 M
  let law :=
    if vExpOverride.isNone && modeScaleOverride.isNone then
      resolveLaw slot Mt src mode
    else
      match mode with
      | .allostatic => .allostatic
      | _ => .homeostatic
  let (E, θ0raw) :=
    escalationScore slot M src goalDelta load law vExpOverride modeScaleOverride
  let Vd := match law with
    | .allostatic => slot.domainPrior
    | .homeostatic => slot.effectiveV
  let θ :=
    if decide (Mt ≥ explicitOverrideM) && src == .explicitStatement &&
        decide (goalDelta ≥ 0.0) then
      match explicitThetaCap Vd with
      | some cap => min θ0raw cap
      | none => θ0raw
    else θ0raw
  let R := src.reliability
  let escalate :=
    decide (E > θ) ||
      (decide (slot.mismatchCount ≥ cumulativeMismatchEscalate) &&
        decide (Mt ≥ 0.5) && decide (R ≥ strongInferenceR))
  (escalate, E, θ, law)

def beliefShiftBar (Vd : Float) : Float :=
  beliefShiftK / max Vd beliefShiftMinVd

def decayWeight (ageDays : Float) : Float :=
  Float.pow 0.5 (ageDays / surpriseHalflifeDays)

/-- One evidence row: `(ageDays, M, R)`. -/
def beliefShiftMass (rows : List (Float × Float × Float)) : Float :=
  rows.foldl (fun acc ⟨age, M, R⟩ =>
    acc + clamp01 M * R * decayWeight (max age 0.0)) 0.0

/-! ### Composite gate as Bool, so the switch is a lemma not a float. -/

def explicitHighM (M : Float) (src : Source) : Bool :=
  decide (clamp01 M ≥ explicitOverrideM) && src == .explicitStatement

def learnedResidual (slot : Slot) (M : Float) : Bool :=
  learnedHatE slot && decide (residualSurprise slot M ≥ residualGate)

def compositeOpens (explicit learnedRes : Bool) : Bool :=
  explicit || learnedRes

theorem composite_explicit_opens (learnedRes : Bool) :
    compositeOpens true learnedRes = true := by
  simp [compositeOpens]

theorem composite_fresh_stays_closed :
    compositeOpens false false = false := by
  simp [compositeOpens]

theorem composite_is_switch (explicit learnedRes : Bool) :
    compositeOpens explicit learnedRes =
      (if explicit then true else learnedRes) := by
  cases explicit <;> cases learnedRes <;> simp [compositeOpens]

end VoltMem.ControlLaw
