"""
Licensing anchor-bar eval — conservative join vs anchored generic slots.
=======================================================================

Sibling research: licensing-collective-selves D3d (anchors prevent fragmentation).

Run:
    .venv/bin/python experiments/licensing_anchor_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voltmem.structure import HeuristicStructuredExtractor, join_structured  # noqa: E402
from linking_pairs import SPLITS  # noqa: E402

ANCHOR_SLOTS = frozenset({"occupation", "employer", "residence_city"})


def _eval_held_out(extractor, *, anchored: bool) -> tuple[int, int, int]:
    """Returns (must_link_hits, must_link_total, false_merges)."""
    pos_pairs, neg_pairs = SPLITS["held-out"]
    must = 0
    false_merges = 0
    for _sd, old, _nd, new, _note in pos_pairs:
        stored = extractor.extract(old)
        incoming = extractor.extract(new)
        kwargs = dict(
            new_text=new,
            stored_text=old,
            conservative=True,
        )
        if anchored:
            kwargs["anchored_attributes"] = ANCHOR_SLOTS
        if join_structured(stored, incoming, **kwargs):
            must += 1
    for _sd, old, _nd, new, _note in neg_pairs:
        stored = extractor.extract(old)
        incoming = extractor.extract(new)
        kwargs = dict(
            new_text=new,
            stored_text=old,
            conservative=True,
        )
        if anchored:
            kwargs["anchored_attributes"] = ANCHOR_SLOTS
        if join_structured(stored, incoming, **kwargs):
            false_merges += 1
    return must, len(pos_pairs), false_merges


def main() -> None:
    extractor = HeuristicStructuredExtractor()
    base_must, base_total, base_fm = _eval_held_out(extractor, anchored=False)
    anc_must, _, anc_fm = _eval_held_out(extractor, anchored=True)

    print("Licensing anchor-bar eval (structured join, held-out)")
    print(f"  anchor slots: {sorted(ANCHOR_SLOTS)}")
    print(
        f"  conservative: must-link {base_must}/{base_total} "
        f"false_merges {base_fm}"
    )
    print(
        f"  anchor bar:   must-link {anc_must}/{base_total} "
        f"false_merges {anc_fm}"
    )
    pass_fm = anc_fm == 0
    pass_ml = anc_must >= base_must
    print(f"  PASS false_merges==0: {pass_fm}")
    print(f"  PASS must_link>=baseline: {pass_ml}")
    if pass_fm and pass_ml:
        print("  overall: PASS")
    else:
        print("  overall: FAIL (tune anchor set or join rule)")


if __name__ == "__main__":
    main()
