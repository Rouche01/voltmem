import VoltMem.Oracle
import VoltMem.ControlLaw

open VoltMem.ControlLaw
open VoltMem.Oracle

def lawStr : Law → String
  | .homeostatic => "homeostatic"
  | .allostatic => "allostatic"

def boolStr (b : Bool) : String := if b then "true" else "false"

def emit (id : String) (r : Bool × Float × Float × Law) : IO Unit := do
  let (esc, E, θ, law) := r
  IO.println s!"  {id}: escalate={boolStr esc} E={E} theta={θ} law={lawStr law}"

def main : IO Unit := do
  IO.println "VoltMem control-law oracle (Lean Float kernel)"
  emit "career_homeo" careerHomeo
  emit "career_allo" careerAllo
  emit "career_composite" careerComposite
  emit "career_p0" careerP0
  emit "career_p25" careerP25
  emit "mislabel_homeo" mislabelHomeo
  emit "mislabel_allo" mislabelAllo
  emit "mislabel_composite" mislabelComposite
  emit "fresh_weak_composite" freshWeakComposite
  emit "very_stable_allo" veryStableAllo
  IO.println s!"daily_mass={beliefShiftMass dailyPile} job_bar={jobBar} pref_bar={prefBar}"
  IO.println s!"monthly_mass={beliefShiftMass monthlyPile}"
