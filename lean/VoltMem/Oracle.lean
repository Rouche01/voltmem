/-
Named oracle cases from the 0.4.0 batteries. `#guard` locks Lean Float to the
same escalate/law bits Python `tests/test_lean_oracle.py` checks on `scoring.py`.
-/

import VoltMem.ControlLaw

namespace VoltMem.Oracle

open VoltMem.ControlLaw

def job (rep : Nat := 1) : Slot where
  domainPrior := 0.30
  effectiveV := 0.30
  repetitionCount := rep
  mismatchEma := 0.05
  mismatchVar := 0.001

def jobFresh : Slot where
  domainPrior := 0.30
  effectiveV := 0.30

def relationshipFresh : Slot where
  domainPrior := 0.35
  effectiveV := 0.35

def corePref (mismatchCount : Nat := 0) : Slot where
  domainPrior := 0.08
  effectiveV := 0.08
  mismatchCount := mismatchCount

def run (slot : Slot) (M : Float) (src : Source) (mode : Mode)
    (vExpOverride : Option Float := none)
    (modeScaleOverride : Option Bool := none) :=
  escalationDecision slot M src 0.0 1.0 mode vExpOverride modeScaleOverride

def careerHomeo := run (job 6) 0.90 .explicitStatement .homeostatic
def careerAllo := run (job 6) 0.90 .explicitStatement .allostatic
def careerComposite := run (job 6) 0.90 .explicitStatement .composite
def careerP0 := run (job 6) 0.90 .explicitStatement .allostatic
  (vExpOverride := some 0.0) (modeScaleOverride := some false)
def careerP25 := run (job 6) 0.90 .explicitStatement .homeostatic
  (vExpOverride := some 0.25) (modeScaleOverride := some false)

def mislabelHomeo := run relationshipFresh 0.75 .strongInference .homeostatic
def mislabelAllo := run relationshipFresh 0.75 .strongInference .allostatic
def mislabelComposite := run relationshipFresh 0.75 .strongInference .composite

def freshWeakComposite := run jobFresh 0.40 .weakInference .composite
def veryStableAllo := run (corePref 2) 0.90 .explicitStatement .allostatic

-- Battery C: explicit career change. Homeostatic (and p=0.25) miss; allostatic hits.
#guard careerHomeo.1 = false
#guard careerAllo.1 = true
#guard careerComposite.1 = true
#guard careerComposite.2.2.2 = Law.allostatic
#guard careerP0.1 = true
#guard careerP25.1 = false

-- Battery E insurance: fresh mislabel, weak-ish evidence.
#guard mislabelHomeo.1 = false
#guard mislabelAllo.1 = true
#guard mislabelComposite.1 = false
#guard mislabelComposite.2.2.2 = Law.homeostatic

#guard freshWeakComposite.1 = false
#guard freshWeakComposite.2.2.2 = Law.homeostatic
#guard veryStableAllo.1 = false

/-- Sixteen weak mentions: daily mass clears the job bar; monthly does not. -/
def weakRow (ageDays : Float) : Float × Float × Float :=
  (ageDays, 0.40, 0.4)

def dailyPile : List (Float × Float × Float) :=
  List.range 16 |>.map (fun i => weakRow (Float.ofNat i))

def monthlyPile : List (Float × Float × Float) :=
  List.range 16 |>.map (fun i => weakRow (Float.ofNat i * 30.0))

def jobBar : Float := beliefShiftBar 0.30
def prefBar : Float := beliefShiftBar 0.08

#guard decide (beliefShiftMass dailyPile > jobBar)
#guard decide (beliefShiftMass monthlyPile < jobBar)
#guard decide (beliefShiftMass dailyPile < prefBar)

end VoltMem.Oracle
