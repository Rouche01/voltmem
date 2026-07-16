"""
Escalation calibration table — print E_t vs θ for tuning EXPLICIT_* constants.

Run after changing THETA_0, ALPHA, domain priors, or explicit-override band.

    .venv/bin/python experiments/calibrate_escalation.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voltmem.domains import MemoryItem  # noqa: E402
from voltmem.scoring import (  # noqa: E402
    escalation_decision,
    escalation_score,
    explicit_theta_cap,
    EXPLICIT_MIN_VD,
    EXPLICIT_MAX_VD,
    EXPLICIT_E_RATIO,
    EXPLICIT_OVERRIDE_M,
)

M_EXPLICIT = 0.90
SOURCE = "explicit_statement"


def _item(v_d: float, domain: str = "professional_context") -> MemoryItem:
    import time
    now = time.time()
    return MemoryItem(
        id="cal",
        content="x",
        domain=domain,
        source=SOURCE,
        volatility_ema=v_d,
        created_at=now,
        last_confirmed_at=now,
    )


def main() -> None:
    print("=" * 88)
    print("ESCALATION CALIBRATION — explicit M=0.9, source=explicit_statement")
    print("=" * 88)
    print(
        f"  Band: [{EXPLICIT_MIN_VD}, {EXPLICIT_MAX_VD}]  "
        f"θ_cap = V_d × {EXPLICIT_E_RATIO}  M_override ≥ {EXPLICIT_OVERRIDE_M}"
    )
    print()
    print(
        f"  {'V_d':>6}  {'E_t':>8}  {'θ_raw':>8}  {'θ_cap':>8}  {'θ_eff':>8}  "
        f"{'esc':>5}  note"
    )
    print("  " + "-" * 72)

    for v in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.55, 0.60]:
        item = _item(v)
        esc, e, theta_eff = escalation_decision(
            item, M_EXPLICIT, source=SOURCE)
        _, theta_raw = escalation_score(item, M_EXPLICIT, source=SOURCE)
        cap = explicit_theta_cap(v)
        cap_s = f"{cap:.4f}" if cap is not None else "   —  "
        if v < EXPLICIT_MIN_VD:
            note = "very stable (no cap)"
        elif v > EXPLICIT_MAX_VD:
            note = "volatile (raw math)"
        else:
            note = "medium-stable band"
        print(
            f"  {v:6.2f}  {e:8.4f}  {theta_raw:8.4f}  {cap_s:>8}  "
            f"{theta_eff:8.4f}  {'YES' if esc else 'no':>5}  {note}"
        )

    print()
    print("  Target: medium-stable band should show esc=YES; V_d ≤ 0.10 should show no.")
    print("=" * 88)


if __name__ == "__main__":
    main()
