/-
Algebraic content of the VoltMem control law, over `Rat`.

Python `scoring.py` uses `α = 0.6` and a sigmoid `G`. Here residual `res` is
already `M R / C^α · G`, so the identities are only about how the prior `V`
enters `E` and `θ`. IEEE floats and the composite *switch* live in
`VoltMem.ControlLaw`.
-/

import Init.Data.Rat.Lemmas

namespace VoltMem.Algebra

theorem lt_trans {a b c : Rat} (hab : a < b) (hbc : b < c) : a < c := by
  have hle : a ≤ c := Rat.le_trans (Rat.le_of_lt hab) (Rat.le_of_lt hbc)
  match (Rat.le_iff_eq_or_lt).mp hle with
  | Or.inl h => exact h
  | Or.inr heq =>
    subst heq
    have : a = b := Rat.le_antisymm (Rat.le_of_lt hab) (Rat.le_of_lt hbc)
    subst this
    exact (Rat.lt_irrefl hab).elim

/-- Base threshold `θ₀` (Python `THETA_0 = 0.15`). -/
def theta0 : Rat := 3 / 20

/-- Homeostatic evidence: residual discounted by the prior. -/
def E_homeostatic (res V : Rat) : Rat := res * V

/-- Allostatic evidence: residual only. `V` does not enter the score. -/
def E_allostatic (res : Rat) : Rat := res

/-- Threshold from the prior. `s = 1` is the settled (no-surprise) case. -/
def theta (θ0 V L s : Rat) : Rat := θ0 / V * L * s

/-- Homeostatic double-charges `V`: it cancels in the product `E · θ`. -/
theorem homeostatic_double_charge
    (res V θ0 L : Rat) (hV : V ≠ 0) :
    E_homeostatic res V * theta θ0 V L 1 = res * θ0 * L := by
  have h1 : V * (θ0 / V) = θ0 := by
    rw [Rat.mul_comm]
    exact Rat.div_mul_cancel hV
  calc
    E_homeostatic res V * theta θ0 V L 1
      = (res * V) * (θ0 / V * L * 1) := rfl
    _ = (res * V) * (θ0 / V * L) := by rw [Rat.mul_one]
    _ = res * (V * (θ0 / V * L)) := by rw [Rat.mul_assoc]
    _ = res * (V * (θ0 / V) * L) := by rw [← Rat.mul_assoc V]
    _ = res * (θ0 * L) := by rw [h1]
    _ = res * θ0 * L := by rw [← Rat.mul_assoc]

/-- Allostatic charges `V` once, on the bar. -/
theorem allostatic_single_charge
    (res V θ0 L : Rat) (_hV : V ≠ 0) :
    E_allostatic res * theta θ0 V L 1 = res * (θ0 / V) * L := by
  calc
    E_allostatic res * theta θ0 V L 1
      = res * (θ0 / V * L * 1) := rfl
    _ = res * (θ0 / V * L) := by rw [Rat.mul_one]
    _ = res * (θ0 / V) * L := by rw [← Rat.mul_assoc]

/-- Homeostatic (settled) escalates iff `θ₀ < res · V²`. -/
theorem homeostatic_escalates_iff
    (res V θ0 : Rat) (hV : 0 < V) :
    E_homeostatic res V > theta θ0 V 1 1 ↔ θ0 < res * (V * V) := by
  constructor
  · intro h
    have h' : θ0 / V < res * V := by
      simpa [E_homeostatic, theta, Rat.mul_one] using h
    have : θ0 < (res * V) * V := (Rat.div_lt_iff hV).mp h'
    simpa [Rat.mul_assoc] using this
  · intro h
    have : θ0 < (res * V) * V := by simpa [Rat.mul_assoc] using h
    have h' : θ0 / V < res * V := (Rat.div_lt_iff hV).mpr this
    simpa [E_homeostatic, theta, Rat.mul_one] using h'

/-- Allostatic (settled `s = 1`) escalates iff `θ₀ < res · V`. -/
theorem allostatic_escalates_iff
    (res V θ0 : Rat) (hV : 0 < V) :
    E_allostatic res > theta θ0 V 1 1 ↔ θ0 < res * V := by
  constructor
  · intro h
    have h' : θ0 / V < res := by
      simpa [E_allostatic, theta, Rat.mul_one] using h
    exact (Rat.div_lt_iff hV).mp h'
  · intro h
    have h' : θ0 / V < res := (Rat.div_lt_iff hV).mpr h
    simpa [E_allostatic, theta, Rat.mul_one] using h'

/-- For `0 < V < 1` the homeostatic bar is strictly higher than allostatic. -/
theorem homeostatic_bar_strictly_higher
    (V θ0 : Rat) (hV0 : 0 < V) (hV1 : V < 1) (hθ : 0 < θ0) :
    theta θ0 V 1 1 < theta θ0 (V * V) 1 1 := by
  have hV2 : 0 < V * V := Rat.mul_pos hV0 hV0
  have hcancel : θ0 / V * V = θ0 := Rat.div_mul_cancel (Rat.ne_of_gt hV0)
  simp only [theta, Rat.mul_one]
  rw [Rat.lt_div_iff hV2]
  calc
    θ0 / V * (V * V)
      = (θ0 / V * V) * V := by rw [← Rat.mul_assoc]
    _ = θ0 * V := by rw [hcancel]
    _ < θ0 * 1 := Rat.mul_lt_mul_of_pos_left hV1 hθ
    _ = θ0 := Rat.mul_one θ0

/--
The career-change cell: residual that clears the allostatic bar (`θ₀/V`)
but not the homeostatic bar (`θ₀/V²`). A blend `p ∈ (0,1)` is not an
interpolation of those two cells — `p = 0` vs `p = 1` already opens a gap
whenever `0 < V < 1`.
-/
theorem prior_residual_gap
    (res V θ0 : Rat)
    (hV0 : 0 < V) (hAllo : θ0 < res * V) (hHomeo : res * (V * V) ≤ θ0) :
    (E_allostatic res > theta θ0 V 1 1) ∧
      ¬ (E_homeostatic res V > theta θ0 V 1 1) := by
  constructor
  · exact (allostatic_escalates_iff res V θ0 hV0).mpr hAllo
  · intro h
    have : θ0 < res * (V * V) := (homeostatic_escalates_iff res V θ0 hV0).mp h
    exact (Rat.not_lt.mpr hHomeo) this

/-- Concrete job-band numbers: `V = 3/10`, `θ₀ = 3/20`. -/
def V_job : Rat := 3 / 10

theorem job_allostatic_bar : theta0 / V_job = 1 / 2 := by native_decide
theorem job_homeostatic_bar : theta0 / (V_job * V_job) = 5 / 3 := by native_decide

/-- Uncapped laws disagree on any residual strictly between `1/2` and `5/3`. -/
theorem job_gap_instance
    (res : Rat) (hLo : (1 : Rat) / 2 < res) (hHi : res ≤ 5 / 3) :
    (E_allostatic res > theta theta0 V_job 1 1) ∧
      ¬ (E_homeostatic res V_job > theta theta0 V_job 1 1) := by
  have hV0 : (0 : Rat) < V_job := by native_decide
  have hAllo : theta0 < res * V_job := by
    have : theta0 / V_job < res := by
      rw [job_allostatic_bar]
      exact hLo
    exact (Rat.div_lt_iff hV0).mp this
  have hHomeo : res * (V_job * V_job) ≤ theta0 := by
    have hbar : theta0 / (V_job * V_job) = (5 : Rat) / 3 := job_homeostatic_bar
    have hV2 : (0 : Rat) < V_job * V_job := by native_decide
    have : res ≤ theta0 / (V_job * V_job) := by
      rw [hbar]
      exact hHi
    have h' : res * (V_job * V_job) ≤
        (theta0 / (V_job * V_job)) * (V_job * V_job) :=
      Rat.mul_le_mul_of_nonneg_right this (Rat.le_of_lt hV2)
    have hcancel : theta0 / (V_job * V_job) * (V_job * V_job) = theta0 :=
      Rat.div_mul_cancel (Rat.ne_of_gt hV2)
    rw [hcancel] at h'
    exact h'
  exact prior_residual_gap res V_job theta0 hV0 hAllo hHomeo

/-! ### Time-decayed mass: later evidence weighs less. -/

/-- `(1/2)^n`. Monthly spacing is this weight at `n` months. -/
def halfPow : Nat → Rat
  | 0 => 1
  | n + 1 => halfPow n / 2

theorem halfPow_pos : ∀ n, (0 : Rat) < halfPow n
  | 0 => by native_decide
  | n + 1 => by
    have h : (0 : Rat) < halfPow n := halfPow_pos n
    have h2 : (0 : Rat) < 2 := by native_decide
    simpa [halfPow] using (Rat.lt_div_iff h2).mpr (by simpa using h)

theorem halfPow_succ_lt (n : Nat) : halfPow (n + 1) < halfPow n := by
  have hpos := halfPow_pos n
  have h2 : (0 : Rat) < (2 : Rat) := by native_decide
  simp only [halfPow]
  rw [Rat.div_lt_iff h2]
  have : (1 : Rat) < 2 := by native_decide
  have hmul : halfPow n * 1 < halfPow n * 2 :=
    Rat.mul_lt_mul_of_pos_left this hpos
  simpa [Rat.mul_one] using hmul

theorem halfPow_antitone {n m : Nat} (h : n < m) : halfPow m < halfPow n := by
  induction m with
  | zero => cases h
  | succ m ih =>
    have hle : n ≤ m := Nat.le_of_lt_succ h
    have hsucc := halfPow_succ_lt m
    match Nat.eq_or_lt_of_le hle with
    | Or.inl heq => simpa [heq] using hsucc
    | Or.inr hlt => exact lt_trans hsucc (ih hlt)

/-- Sleeptime bar `k / V` is higher for more stable (smaller `V`) domains. -/
theorem belief_bar_antitone
    (k V1 V2 : Rat) (hk : 0 < k) (hV : 0 < V1) (hLt : V1 < V2) :
    k / V2 < k / V1 := by
  have hV2 : 0 < V2 := lt_trans hV hLt
  rw [Rat.div_lt_iff hV2]
  have hcancel : k / V1 * V1 = k := Rat.div_mul_cancel (Rat.ne_of_gt hV)
  have hpos : 0 < k / V1 := (Rat.lt_div_iff hV).mpr (by simpa using hk)
  have : k / V1 * V1 < k / V1 * V2 := Rat.mul_lt_mul_of_pos_left hLt hpos
  rw [hcancel] at this
  exact this

end VoltMem.Algebra
