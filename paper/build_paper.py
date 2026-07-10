"""
Build the reconciled VoltMem paper PDF from source.
=================================================

The original `volatility_ewc_portfolio.pdf` was a compiled binary with no source
in the repo, and it predated the Split-MNIST / capacity / library / multi-turn
results (and their negative controls). This script is the editable source of
record for the PDF. It regenerates `volatility_ewc_portfolio.pdf`, reconciled with
the honest findings (see paper/findings.md and the experiments/ scripts).

The original is archived as `volatility_ewc_portfolio_v1_original.pdf`.

Run:
    uv pip install reportlab --python .venv/bin/python   # once
    .venv/bin/python paper/build_paper.py
"""

import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "volatility_ewc_portfolio.pdf")

# Source of truth for arXiv abstract — keep paper/ARXIV_SUBMISSION.md in sync.
ABSTRACT = (
    "Standard continual-learning methods such as Elastic Weight Consolidation apply "
    "a uniform protection strength to all past knowledge, forcing a "
    "stability&ndash;plasticity tradeoff. We scale protection per domain by measured "
    "volatility <i>before</i> any update, and validate claims with negative controls: "
    "shuffling or inverting the domain&rarr;volatility map must degrade performance "
    "monotonically (REAL &gt; SHUFFLE &gt; SWAP). On a synthetic benchmark with "
    "disjoint task inputs, both retention and adaptation improve simultaneously. On "
    "Split-MNIST the honest result is subtler: volatility weighting is not a "
    "free-lunch Pareto win but a <b>causal control knob</b> that steers the "
    "tradeoff. The same principle powers VoltMem, an open-source Python memory "
    "library for LLM agents: volatility-aware write and retrieval policies beat "
    "flat and inverted controls on scripted multi-turn scenarios (balanced score "
    "0.597, only policy strong on both stable and volatile axes), a noisy retrieval "
    "haystack (0% stale@1 vs 20% cosine-only; separation 0.153 vs &minus;0.003), "
    "and a 3/3 current-truth case study vs Mem0 on mood, preference, and location "
    "updates. On LongMemEval-S (n=60, chunk-calibrated RAG ingest), retrieval "
    "answer@5 is 70.0% for VoltMem&mdash;tying plain cosine and beating the "
    "inverted-volatility control (66.7%)&mdash;without claiming public-benchmark "
    "SOTA (uniform-volatility ablation reaches 71.7%). We report limitations, "
    "embedding-backend variance, failed early wins discarded by sabotage controls, "
    "and open gaps in automatic domain discovery."
)

ABSTRACT_STYLE = ParagraphStyle(
    "abstract", parent=None, fontName="Times-Roman", fontSize=10, leading=14,
    alignment=TA_JUSTIFY, spaceAfter=6, leftIndent=8, rightIndent=8,
)
ABSTRACT_LABEL = ParagraphStyle(
    "abstract_label", parent=None, fontName="Times-Bold", fontSize=11,
    leading=14, alignment=TA_CENTER, spaceBefore=4, spaceAfter=6,
)

# ── styles ──────────────────────────────────────────────────────────────────
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Times-Roman",
                      fontSize=10, leading=14, alignment=TA_JUSTIFY,
                      spaceAfter=6)
REF = ParagraphStyle("ref", parent=BODY, fontName="Times-Roman", fontSize=9,
                     leading=12, alignment=TA_JUSTIFY, spaceAfter=3,
                     leftIndent=12, firstLineIndent=-12)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Times-Bold",
                    fontSize=13, leading=16, spaceBefore=10, spaceAfter=4,
                    textColor=colors.HexColor("#1a1a1a"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold",
                    fontSize=11, leading=14, spaceBefore=6, spaceAfter=2)
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName="Times-Bold",
                       fontSize=19, leading=23, alignment=TA_CENTER,
                       spaceAfter=2)
SUB = ParagraphStyle("sub", parent=ss["Normal"], fontName="Times-Italic",
                     fontSize=10.5, leading=14, alignment=TA_CENTER,
                     textColor=colors.HexColor("#444444"), spaceAfter=2)
NOTE = ParagraphStyle("note", parent=ss["Normal"], fontName="Times-Italic",
                      fontSize=8.5, leading=11, alignment=TA_CENTER,
                      textColor=colors.HexColor("#666666"), spaceAfter=8)
FORMULA = ParagraphStyle("formula", parent=BODY, fontName="Times-Roman",
                         fontSize=11, leading=17, alignment=TA_CENTER,
                         spaceBefore=4, spaceAfter=4,
                         textColor=colors.HexColor("#1a1a1a"))
FORMULA_BLOCK = ParagraphStyle("formula_block", parent=FORMULA,
                               spaceBefore=8, spaceAfter=10)
TLDR = ParagraphStyle("tldr", parent=BODY, backColor=colors.HexColor("#f4f4f2"),
                      borderPadding=8, leftIndent=2, rightIndent=2)
CAP = ParagraphStyle("cap", parent=BODY, fontName="Times-Italic", fontSize=8.5,
                     leading=11, textColor=colors.HexColor("#555555"),
                     spaceBefore=1, spaceAfter=10, alignment=TA_CENTER)


def P(t, style=BODY):
    return Paragraph(t, style)


def F(t, block=False):
    """Centered display equation with HTML sub/superscripts."""
    return Paragraph(t, FORMULA_BLOCK if block else FORMULA)


def bullets(items):
    return [P(f"&bull;&nbsp;&nbsp;{t}") for t in items]


def refs(items):
    return [P(t, REF) for t in items]


REFERENCES = [
    "[1] Kirkpatrick, J., et al. Overcoming catastrophic forgetting in neural "
    "networks. <i>PNAS</i>, 114(13):3521&ndash;3526, 2017.",
    "[2] Lopez-Paz, D., &amp; Ranzato, M. Gradient episodic memory for continual "
    "learning. <i>NeurIPS</i>, 2017.",
    "[3] Zenke, F., et al. Continual learning through synaptic intelligence. "
    "<i>ICML</i>, 2017.",
    "[4] Schwarz, J., et al. Progress &amp; compress: A scalable framework for "
    "continual learning. <i>ICML</i>, 2018.",
    "[5] Wu, Z., et al. LongMemEval: Benchmarking chat assistants on long-term "
    "interactive memory. <i>ICLR</i>, 2025.",
    "[6] Chhikara, P., et al. Mem0: Building production-ready AI agents with "
    "scalable long-term memory. <i>arXiv:2504.19413</i>, 2025.",
    "[7] Parisi, G. I., et al. Continual lifelong learning with neural networks: "
    "A review. <i>Neural Networks</i>, 113:54&ndash;71, 2019.",
    "[8] Reimers, N., &amp; Gurevych, I. Sentence-BERT: Sentence embeddings using "
    "Siamese BERT-networks. <i>EMNLP</i>, 2019.",
]


def make_table(data, col_widths=None, highlight_col=None):
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#33415c")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f2f4f7")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9ced6")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if highlight_col is not None:
        style.append(("BACKGROUND", (highlight_col, 1), (highlight_col, -1),
                      colors.HexColor("#e3ecff")))
        style.append(("FONTNAME", (highlight_col, 1), (highlight_col, -1),
                      "Times-Bold"))
    t.setStyle(TableStyle(style))
    return t


def hr():
    return HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor("#cccccc"), spaceBefore=6,
                      spaceAfter=6)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawCentredString(A4[0] / 2.0, 12 * mm,
                             f"Volatility-Adjusted Memory Protection  ·  "
                             f"page {doc.page}")
    canvas.restoreState()


def build():
    story = []

    # ── header ────────────────────────────────────────────────────────────
    story += [
        P("Volatility-Adjusted Memory Protection", TITLE),
        P("A causal control knob for continual learning and LLM agent memory",
          SUB),
        P("Richard Emate", SUB),
        P("richard@theemate.com &middot; Independent &middot; 2026", SUB),
    ]

    # ── Abstract (synced with paper/ARXIV_SUBMISSION.md) ──────────────────
    story += [P("Abstract", ABSTRACT_LABEL)]
    story += [P(ABSTRACT, ABSTRACT_STYLE)]
    story += [Spacer(1, 6)]

    # ── 1. Introduction ───────────────────────────────────────────────────
    story += [P("1. Introduction", H1)]
    story += [P(
        "Continual-learning systems and LLM agent memory layers both face the "
        "<i>stability&ndash;plasticity dilemma</i>: protect old knowledge too "
        "strongly and the system cannot adapt; too weakly and it forgets or goes "
        "stale. Elastic Weight Consolidation (EWC)&nbsp;[1] penalises parameter "
        "changes in proportion to Fisher information, but applies one global "
        "protection strength uniformly. Agent memory systems such as Mem0&nbsp;[6] "
        "similarly store relevant facts without modelling how fast different kinds "
        "of knowledge age. Not all knowledge changes at the same rate: biographical "
        "facts are stable; mood, location, and current tasks are volatile.")]
    story += [P("This work makes three contributions:")]
    story += bullets([
        "<b>Per-domain volatility scaling</b> of memory protection and retrieval "
        "freshness, with volatility measured <i>before</i> any update to avoid "
        "circularity.",
        "<b>Sabotage negative controls</b> (REAL &gt; SHUFFLE &gt; SWAP) that gate "
        "every non-synthetic empirical claim.",
        "<b>VoltMem</b>, an open-source Python memory library validating the same "
        "principle on agent write policy, retrieval ranking, and scripted "
        "comparisons against Mem0&nbsp;[6] and LongMemEval&nbsp;[5].",
    ])
    story += [P("Section&nbsp;2 states the hypothesis and core equations. "
                "Section&nbsp;3 describes benchmarks and controls. "
                "Section&nbsp;4 records discarded early wins. "
                "Section&nbsp;5 reports results. Section&nbsp;6 lists limitations.")]

    story += [P("Related work", H2)]
    story += [P(
        "Continual learning methods range from regularisation (EWC&nbsp;[1], "
        "synaptic intelligence&nbsp;[3]) to replay and constrained optimisation "
        "(GEM&nbsp;[2], progress &amp; compress&nbsp;[4]). Surveys&nbsp;[7] frame "
        "the stability&ndash;plasticity tradeoff as the central challenge. "
        "LongMemEval&nbsp;[5] benchmarks long-horizon memory retrieval in chat "
        "assistants; Mem0&nbsp;[6] targets production agent memory with vector "
        "storage and consolidation. VoltMem differs by assigning explicit "
        "volatility priors per domain and validating behaviour under inverted "
        "controls, rather than claiming uniform benchmark dominance.")]

    # ── 2. Hypothesis ─────────────────────────────────────────────────────
    story += [P("2. Hypothesis", H1)]
    story += [P(
        "Scale the EWC penalty per stored task by how volatile that task's domain "
        "actually is, measured before any protection is applied. High measured "
        "volatility &rarr; weak protection; low volatility &rarr; strong "
        "protection.")]
    story += [F("Protection weight", block=True)]
    story += [F("w<sub>d</sub> = 1 / V<sub>d</sub><sup>&gamma;</sup>")]
    story += [P(
        "The same volatility prior drives the open-source VoltMem memory layer. "
        "Write path: an escalation score versus a volatility-scaled threshold; "
        "retrieve path: down-rank stale volatile chunks.")]
    story += [F("Escalation score and threshold", block=True)]
    story += [F("E<sub>t</sub> = [(M<sub>t</sub> &middot; R<sub>t</sub>) / C<sup>&alpha;</sup>] &middot; V<sub>d</sub> &middot; G<sub>t</sub>")]
    story += [F("&theta;<sub>t</sub> = &theta;<sub>0</sub> &middot; (1 / V<sub>d</sub>) &middot; L<sub>t</sub> &nbsp;&nbsp;&nbsp; (audit iff E<sub>t</sub> &gt; &theta;<sub>t</sub>)")]
    story += [F("Retrieval freshness", block=True)]
    story += [F("staleness = 1 &minus; exp(&minus;V<sub>d</sub> &middot; age<sub>days</sub>)")]
    story += [F("score = similarity &middot; (1 &minus; V<sub>d</sub> &middot; staleness)")]
    story += [P(
        "The original prediction was that this improves both retention and "
        "adaptability <i>simultaneously</i>. That turned out to be true only in a "
        "specific regime; the general, real-data claim is weaker and more precise "
        "(Section 5).")]

    # ── 3. Method ─────────────────────────────────────────────────────────
    story += [P("3. Method", H1)]
    story += [P("Synthetic benchmark", H2)]
    story += [P(
        "12 sequential binary classification tasks on 2D Gaussian blobs, "
        "alternating between two domain types sharing a single trunk and single "
        "output head — genuine weight competition, no dedicated capacity to hide "
        "in. <b>Stable domain:</b> blob centres jitter &plusmn;0.15 (boundary "
        "barely moves). <b>Volatile domain:</b> blob centres relocate "
        "substantially each task.")]
    story += [P("Real data — Split-MNIST", H2)]
    story += [P(
        "To answer the obvious objection (&ldquo;2D blobs are a toy&rdquo;), the "
        "same idea runs on Split-MNIST. A stable pool of digits keeps a fixed "
        "digit&rarr;class mapping every task; a disjoint volatile pool is randomly "
        "re-partitioned each task so successive volatile tasks contradict each "
        "other. Both feed one shared trunk and one 2-way head, so they genuinely "
        "compete for weights (real forgetting), but because the pools are "
        "disjoint, protecting stable features does not directly fight volatile "
        "inputs.")]
    story += [P("Volatility estimation (pre-update) &amp; penalty", H2)]
    story += [P(
        "Before any gradient step on a new task, measure the model's loss using "
        "only knowledge from prior tasks — a clean mismatch signal uncorrupted by "
        "the EWC penalty — and track a running EMA per domain. Each stored task's "
        "EWC penalty is then weighted by w<sub>d</sub> above: high V<sub>d</sub> "
        "&rarr; small w<sub>d</sub> &rarr; weak protection; low V<sub>d</sub> "
        "&rarr; large w<sub>d</sub> &rarr; strong protection.")]
    story += [P("Negative control (the load-bearing part)", H2)]
    story += [P(
        "Every non-synthetic claim is gated on a <b>sabotage / negative "
        "control</b>: re-run the identical pipeline with the domain&rarr;"
        "volatility mapping (i) shuffled and (ii) inverted. A genuine effect must "
        "degrade monotonically <b>REAL &gt; SHUFFLE &gt; SWAP</b>. If a "
        "&ldquo;win&rdquo; survives shuffling or inversion, it is generic "
        "non-uniform regularisation, not the volatility signal, and it is "
        "discarded. Several early wins failed this test — that filter is why the "
        "surviving claims are trustworthy.")]

    # ── 4. Failures worth keeping ─────────────────────────────────────────
    story += [P("4. Failures Worth Keeping", H1)]
    story += [P("Early attempts produced flat ties or artifacts — kept because "
                "diagnosing them was most of the actual work:")]
    story += bullets([
        "<b>Separate output heads</b> — no genuine weight competition, nothing "
        "to forget in either method.",
        "<b>Volatility measured post-update</b> — contaminated by EWC clamping; "
        "it tracked the intervention, not the domain.",
        "<b>Models trained to 100% confidence</b> — collapses Fisher information "
        "toward zero, making the penalty inert. Fixed by label smoothing and "
        "fewer epochs.",
        "<b>High-capacity MNIST &ldquo;win&rdquo;</b> (hidden=512, +7pp "
        "retention) — <i>failed the negative control</i>: shuffling/inverting the "
        "pairing kept most of the gain, so it was generic regularisation, not "
        "volatility. Discarded as a headline result.",
        "<b>Capacity-saving hypothesis</b> (a small volatility model matching a "
        "larger baseline) — <i>falsified</i> by the sweep (Section 5.3).",
    ])

    # ── 5. Results ────────────────────────────────────────────────────────
    story += [P("5. Results", H1)]

    story += [P("5.1  Synthetic — both axes improve (regime-specific)", H2)]
    story += [make_table([
        ["Metric", "Baseline EWC", "Volatility-Adjusted"],
        ["Early-task stable retention", "0.935", "0.962  (+2.7pp)"],
        ["Late-task volatile adaptation", "0.576", "0.628  (+5.2pp)"],
        ["Avg accuracy — stable tasks", "0.936", "0.963  (+2.7pp)"],
        ["Avg accuracy — volatile tasks", "0.617", "0.633  (+1.6pp)"],
    ], col_widths=[70 * mm, 40 * mm, 45 * mm], highlight_col=2)]
    story += [P("8 runs, 12 tasks each. The model inferred the volatile domain "
                "was ~2.4&times; more unstable than the stable one, purely from "
                "pre-update loss. Holds when tasks use disjoint inputs over a "
                "shared trunk; not a universal Pareto win (see 5.2).", CAP)]

    story += [P("5.2  Split-MNIST — a causal control knob", H2)]
    story += [P(
        "In a genuinely-competing regime (hidden=128, &lambda;&asymp;300, 16 "
        "tasks, 6 runs) the effect <i>passes</i> the negative control. Reported "
        "as a stability index = retention &minus; adaptation:")]
    story += [make_table([
        ["", "baseline", "REAL", "SHUFFLE", "SWAP"],
        ["stability index", "0.317", "0.460", "0.457", "0.405"],
    ], col_widths=[42 * mm, 28 * mm, 28 * mm, 28 * mm, 28 * mm],
        highlight_col=2)]
    story += [P("REAL &minus; SWAP = +0.055 in the predicted direction: inverting "
                "which domains are &ldquo;volatile&rdquo; reliably tilts the model "
                "toward plasticity. The volatility signal causally steers the "
                "tradeoff — but it trades adaptation for retention along a "
                "frontier a well-tuned uniform baseline could also reach.", CAP)]

    story += [P("5.3  Capacity — robustness to under-tuned protection", H2)]
    story += [P(
        "The parameter-savings hypothesis was falsified. The favourable, causal "
        "benefit appears in <b>under-protected large models</b>, where uniform "
        "EWC catastrophically forgets and volatility rescues the stable domain "
        "(hidden=256, &lambda;=300, 4 runs):")]
    story += bullets([
        "retention: baseline 0.877 &rarr; <b>0.954</b> (+7.7pp); adaptation only "
        "&minus;2.9pp,",
        "passes the sabotage control (shuffle/swap keep only +1.3pp).",
    ])
    story += [P("Interpretation: volatility auto-allocates a fixed protection "
                "budget to stable knowledge, reducing the need to hand-tune "
                "&lambda; per capacity. A well-tuned uniform baseline can match "
                "it; volatility mainly reduces tuning burden.", CAP)]

    story += [P("5.4  LLM-memory library — volatility-driven under control", H2)]
    story += [P(
        "The same principle powers a small persistent-memory library. Running it "
        "end-to-end under three volatility profiles — real (priors), flat (all "
        "equal), swap (inverted):")]
    story += [make_table([
        ["Capability", "real", "flat", "swap"],
        ["Selective updating (accuracy)", "100%", "83%", "50%"],
        ["Freshness-aware retrieval (separation)", "+0.589", "+0.202", "-0.267"],
    ], col_widths=[74 * mm, 24 * mm, 24 * mm, 24 * mm], highlight_col=1)]
    story += [P("The swap control degrading <i>below</i> flat is the causal "
                "evidence: flipping which domains are volatile flips behaviour in "
                "the wrong direction.", CAP)]

    story += [P("5.5  Reliability-weighted volatility EMA", H2)]
    story += [P(
        "A multi-turn robustness fix: the volatility EMA's learning rate is scaled "
        "by source reliability:")]
    story += [F("&alpha; = (1 &minus; &beta;) &middot; clamp(R<sub>t</sub>, 0, 1)")]
    story += [F("V &larr; (1 &minus; &alpha;) &middot; V + &alpha; &middot; M<sub>t</sub>")]
    story += [P(
        "A low-trust observation moves a stable memory <b>2.5&times; less</b> than the prior "
        "logic; reliable-source updates are unchanged (backward compatible). A "
        "sustained weak stream crosses V &ge; 0.5 at turn 8 versus the old logic's "
        "turn 2, and stable content is never wrongly overwritten. (Also fixed: "
        "observe() previously updated the EMA twice per call.) Honest limit: "
        "reliability scales the step size, not the EMA's fixed point, so "
        "persistent contradictions still raise volatility over time — arguably "
        "correct, but not a hard &ldquo;never drifts&rdquo; guarantee.")]

    story += [P("5.6  LLM-agent memory &mdash; beating naive policies on both "
                "failure modes", H2)]
    story += [P(
        "As memory for an LLM agent, the same principle is tested over many noisy "
        "sessions where facts drift at domain-appropriate rates. A memory layer "
        "can fail two ways at once: go <i>stale</i> on volatile facts, or get "
        "<i>corrupted</i> by a confident-but-wrong observation on stable facts. "
        "Naive policies are forced onto one side; VoltMem escapes the tradeoff "
        "(24 sessions &times; 20 runs; balanced = harmonic mean of stable &amp; "
        "volatile accuracy):")]
    story += [make_table([
        ["system", "overall", "stable", "volatile", "balanced"],
        ["voltmem (real)", "0.522", "0.578", "0.617", "0.597"],
        ["voltmem (flat)", "0.361", "0.578", "0.188", "0.283"],
        ["voltmem (swap)", "0.347", "0.407", "0.194", "0.263"],
        ["always-overwrite", "0.573", "0.360", "0.767", "0.490"],
        ["never-overwrite", "0.361", "0.578", "0.188", "0.283"],
        ["reliability-threshold", "0.579", "0.437", "0.548", "0.486"],
    ], col_widths=[52 * mm, 24 * mm, 22 * mm, 24 * mm, 26 * mm], highlight_col=4)]
    story += [P("VoltMem-real is the only policy strong on both axes; each naive "
                "policy catastrophically fails one, and a pure reliability rule is "
                "mediocre on both. real &gt; flat &gt; swap is the causal control. "
                "Building the demo also surfaced a real bug: observe() updated the "
                "volatility EMA <i>before</i> the escalation decision, so one "
                "confident blip inflated volatility and lowered its own threshold, "
                "overwriting a stable fact on first contact. Fixed by deciding "
                "against the volatility known <i>before</i> the observation, then "
                "learning from it.", CAP)]

    story += [P("5.7  Slot-aware linking &mdash; write-path wedge vs Mem0", H2)]
    story += [P(
        "Embedding-based <i>remember()</i> must associate a new turn with an "
        "existing memory before <i>observe()</i> can apply the escalation rule. "
        "A single global similarity threshold fails on paraphrases (MiniLM scores "
        "0.44&ndash;0.53 on mood/preference rephrasings). <b>Slot-aware linking</b> "
        "adds a domain-scoped fallback: volatile singleton slots (mood, location, "
        "task) and preference sibling domains link at a volatility-scaled lower "
        "threshold.")]
    story += [make_table([
        ["scenario", "VoltMem", "Mem0 (real)"],
        ["mood update", "1 fact, correct top", "2 facts, stale top"],
        ["preference change", "1 fact, correct top", "2 facts, stale top"],
        ["location change", "1 fact, correct top", "2 facts, stale top"],
    ], col_widths=[50 * mm, 50 * mm, 50 * mm], highlight_col=1)]
    story += [P(
        "Scripted case study (3/3 current-truth wins) vs Mem0&nbsp;[6] "
        "(open-source, gpt-4o-mini, text-embedding-3-small); not a "
        "public-benchmark SOTA claim. Embeddings: MiniLM&nbsp;[8].",
        CAP)]

    story += [P("5.8  Retrieval haystack &amp; LongMemEval-S", H2)]
    story += [P(
        "Noisy haystack (5 slots &times; 20 runs, 6 stale decoys + 3 distractors "
        "per slot, top-5):")]
    story += [make_table([
        ["system", "current@1", "current@5", "stale@1", "separation"],
        ["voltmem_real", "0.600", "1.000", "0.000", "0.153"],
        ["voltmem_flat", "0.800", "1.000", "0.200", "0.133"],
        ["voltmem_swap", "0.650", "1.000", "0.200", "0.111"],
        ["similarity_only", "0.600", "0.800", "0.200", "&minus;0.003"],
    ], col_widths=[38 * mm, 24 * mm, 24 * mm, 24 * mm, 30 * mm], highlight_col=1)]
    story += [P(
        "PASS: real avoids stale volatile traps (0% stale@1 vs 20% cosine), finds "
        "current in top-5 more often (100% vs 80%), separation real &gt; swap.",
        CAP)]
    story += [P("LongMemEval-S (n=60, chunk-calibrated ingest: user &rarr; "
                "<i>stated_preference</i>, assistant &rarr; <i>opinion</i>):", BODY)]
    story += [make_table([
        ["system", "answer@5", "evidence@5"],
        ["voltmem_flat", "0.717", "0.850"],
        ["voltmem_real", "0.700", "0.833"],
        ["similarity_only", "0.700", "0.850"],
        ["voltmem_swap", "0.667", "0.817"],
    ], col_widths=[42 * mm, 32 * mm, 32 * mm], highlight_col=2)]
    story += [make_table([
        ["question type", "real", "flat", "swap", "cosine"],
        ["single-session-preference", "0.700", "0.700", "0.600", "0.700"],
        ["single-session-assistant", "0.900", "0.900", "0.900", "0.900"],
        ["single-session-user", "0.800", "0.900", "0.700", "0.900"],
        ["knowledge-update", "0.600", "0.600", "0.500", "0.600"],
        ["temporal-reasoning", "0.600", "0.600", "0.700", "0.600"],
        ["multi-session", "0.600", "0.600", "0.600", "0.500"],
    ], col_widths=[44 * mm, 22 * mm, 22 * mm, 22 * mm, 24 * mm], highlight_col=1)]
    story += [P(
        "Real ties cosine overall (70.0%); flat leads slightly (71.7%); swap "
        "(66.7%) &lt; real. Preference type recovers to 0.700 (was 0.300 with "
        "heuristic per-turn domains). Not a public-benchmark SOTA claim.",
        CAP)]

    # ── 6. Caveats ────────────────────────────────────────────────────────
    story += [P("6. Caveats", H1)]
    story += bullets([
        "<b>Not a free-lunch accuracy method.</b> On real data it is a control "
        "knob / tuning-robustness tool, not a universal Pareto improvement.",
        "<b>Regime-dependent.</b> The &ldquo;improve both axes&rdquo; result "
        "needs disjoint task inputs over a shared trunk; benefits are strongest "
        "in under-protected large models.",
        "<b>Small-scale benchmarks only.</b> 2D blobs and Split-MNIST; "
        "Split-CIFAR, larger nets, and longer task streams are pending.",
        "<b>EMA fixed-point drift.</b> Reliability weighting slows erosion but "
        "does not anchor the estimate to the domain prior; a prior-anchored "
        "update is the natural next step.",
        "<b>Embedding backend variance.</b> Linking thresholds are calibrated for "
        "sentence-transformers (MiniLM); production should pin or calibrate per "
        "backend.",
        "<b>LongMemEval overall.</b> At n=60 (chunk-calibrated ingest), real ties "
        "cosine (70.0%); flat leads (71.7%); swap (66.7%) &lt; real.",
        "<b>Manual partitioning.</b> Automatic volatility detection from "
        "gradient-conflict signals remains open work.",
        "<b>Vector index (v0.2).</b> SQLite ANN accelerates candidate retrieval; "
        "volatility re-rank is unchanged (engineering, not a separate claim).",
        "<b>No replay-baseline comparison</b> yet (GEM&nbsp;[2], A-GEM, synaptic "
        "intelligence&nbsp;[3]) — volatility-weighting may be additive or redundant.",
    ])

    # ── 7. Conclusion ─────────────────────────────────────────────────────
    story += [P("7. Conclusion", H1)]
    story += [P(
        "Volatility-adjusted memory protection is a validated <b>control knob</b> "
        "on the stability&ndash;plasticity tradeoff, not a free-lunch accuracy "
        "method. Negative controls (REAL &gt; SHUFFLE &gt; SWAP) separate genuine "
        "volatility-driven behaviour from generic non-uniform regularisation. The "
        "same principle transfers to LLM agent memory in VoltMem: selective "
        "updates, freshness-aware retrieval, and current-truth behaviour in "
        "scripted scenarios where uniform memory layers retain stale facts.")]
    story += [P(
        "Future work: prior-anchored volatility estimates, automatic domain "
        "discovery from gradient conflict, replay baselines, and larger-scale "
        "continual-learning benchmarks. Code, reproduction scripts, and extended "
        "tables: github.com/Rouche01/voltmem.")]

    # ── References ────────────────────────────────────────────────────────
    story += [P("References", H1)]
    story += refs(REFERENCES)

    story += [hr()]
    story += [P(
        "Code: <b>github.com/Rouche01/voltmem</b> &middot; "
        "Package: <b>pypi.org/project/voltmem</b> (v0.2.0)",
        CAP)]

    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title="Volatility-Adjusted Memory Protection",
        author="Richard Emate",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
