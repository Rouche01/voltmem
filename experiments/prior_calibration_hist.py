"""
Prior calibration histogram — Battery A domain_stats footprint.

Replays the selective-updating probes into one MemoryLayer (real priors), then
prints an ASCII histogram of audit_rate by domain and writes optional chart
files for slides / notebooks.

Run:
    python experiments/prior_calibration_hist.py
    python experiments/prior_calibration_hist.py --out experiments/out
    python experiments/prior_calibration_hist.py --png   # needs matplotlib

Outputs (under --out, default experiments/out):
    prior_calibration_stats.json
    prior_calibration_audit_rate.svg
    prior_calibration_audit_rate.png   (only with --png + matplotlib)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.sax.saxutils as xml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.voltmem_eval import run_calibration_footprint  # noqa: E402


def ascii_hist(stats: dict[str, dict], width: int = 40) -> str:
    lines = [
        "Prior calibration — audit_rate by domain (Battery A replay)",
        "(high on stable domains → twitchy; ~0 on volatile despite mismatches → stubborn)",
        "",
    ]
    rows = sorted(stats.items(), key=lambda kv: (-kv[1]["audit_rate"], kv[0]))
    for domain, row in rows:
        rate = row["audit_rate"]
        n = int(round(rate * width))
        bar = "#" * n + "." * (width - n)
        lines.append(
            f"  {domain:<22} {rate:5.2f} |{bar}| "
            f"aud={row['audited']} mm={row['logged_mismatch']} "
            f"conf={row['confirmed']} prior={row['prior']:.2f}"
        )
    return "\n".join(lines)


def write_svg(stats: dict[str, dict], path: str) -> None:
    """Grouped horizontal bar chart (audit_rate) — stdlib only."""
    rows = sorted(stats.items(), key=lambda kv: (-kv[1]["audit_rate"], kv[0]))
    if not rows:
        return
    label_w = 160
    bar_max = 320
    row_h = 22
    pad_top, pad_bot = 48, 36
    height = pad_top + pad_bot + row_h * len(rows)
    width = label_w + bar_max + 80

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        '<text x="16" y="28" font-family="ui-sans-serif,system-ui,sans-serif" '
        'font-size="14" font-weight="600" fill="#111">'
        "VoltMem prior calibration — audit_rate by domain</text>",
        '<text x="16" y="44" font-family="ui-sans-serif,system-ui,sans-serif" '
        'font-size="11" fill="#555">'
        "Battery A replay · real priors · higher = more audits / decisions</text>",
    ]
    for i, (domain, row) in enumerate(rows):
        y = pad_top + i * row_h
        rate = float(row["audit_rate"])
        bw = max(1, int(rate * bar_max)) if row["decisions"] else 0
        # stable-ish priors tint cooler; volatile warmer (by prior value)
        prior = float(row["prior"])
        fill = "#2a6f97" if prior < 0.35 else ("#c45c26" if prior > 0.55 else "#6b7280")
        label = xml.escape(domain)
        parts.append(
            f'<text x="12" y="{y + 14}" font-family="ui-monospace,Menlo,monospace" '
            f'font-size="11" fill="#222">{label}</text>'
        )
        parts.append(
            f'<rect x="{label_w}" y="{y + 4}" width="{bw}" height="14" '
            f'fill="{fill}" rx="2"/>'
        )
        parts.append(
            f'<text x="{label_w + bw + 6}" y="{y + 14}" '
            f'font-family="ui-monospace,Menlo,monospace" font-size="11" fill="#333">'
            f'{rate:.2f}</text>'
        )
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")


def write_png(stats: dict[str, dict], path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    rows = sorted(stats.items(), key=lambda kv: (-kv[1]["audit_rate"], kv[0]))
    domains = [d for d, _ in rows]
    rates = [r["audit_rate"] for _, r in rows]
    colors = [
        "#2a6f97" if r["prior"] < 0.35 else (
            "#c45c26" if r["prior"] > 0.55 else "#6b7280"
        )
        for _, r in rows
    ]
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(domains) + 1)))
    ax.barh(domains[::-1], rates[::-1], color=colors[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("audit_rate (audited / decisions)")
    ax.set_title("VoltMem prior calibration — Battery A replay (real priors)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", default=os.path.join("experiments", "out"),
        help="Directory for json/svg/png (default: experiments/out)",
    )
    ap.add_argument(
        "--png", action="store_true",
        help="Also write PNG (requires matplotlib)",
    )
    ap.add_argument("--width", type=int, default=40, help="ASCII bar width")
    args = ap.parse_args()

    stats = run_calibration_footprint()
    print(ascii_hist(stats, width=args.width))
    print()

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "prior_calibration_stats.json")
    svg_path = os.path.join(args.out, "prior_calibration_audit_rate.svg")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    write_svg(stats, svg_path)
    print(f"  wrote {json_path}")
    print(f"  wrote {svg_path}")

    if args.png:
        png_path = os.path.join(args.out, "prior_calibration_audit_rate.png")
        if write_png(stats, png_path):
            print(f"  wrote {png_path}")
        else:
            print("  skipped PNG — install matplotlib: pip install matplotlib")


if __name__ == "__main__":
    main()
