"""
Build the allostatic control-law companion PDF.

Source of record for the prose is paper/allostatic-memory-control.md.
This script renders a submission-ready PDF in the same style as
paper/build_paper.py.

Run:
    .venv/bin/python paper/build_allostatic_paper.py
"""

import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "allostatic-memory-control.pdf")
PRIOR_DOI = "10.5281/zenodo.21962419"
PRIOR_DOI_URL = f"https://doi.org/{PRIOR_DOI}"

ABSTRACT = (
    "Agent memory systems that scale overwrite protection by a domain volatility "
    "prior V<sub>d</sub> treat two different quantities as one number: a slow "
    "belief about how often a <i>kind</i> of fact changes, and a fast residual "
    "about whether <i>this</i> observation was unexpected. We separate them. A "
    "<b>homeostatic</b> law charges V<sub>d</sub> in both the evidence score and "
    "the threshold. An <b>allostatic</b> law drops V<sub>d</sub> from the score "
    "and scales the threshold by leftover surprise "
    "r<sub>t</sub> = |M<sub>t</sub> &minus; &Ecirc;<sub>t</sub>| / "
    "&sigma;<sub>t</sub>, the distance from predicted mismatch rather than from "
    "stored text. A <b>composite</b> gate uses the allostatic law only for an "
    "explicit high-mismatch correction or an unexpected residual against a "
    "learned &Ecirc;; otherwise it keeps the homeostatic insurance."
    "<br/><br/>"
    "On scripted probes with oracle domain and mismatch, dropping V<sub>d</sub> "
    "from the score recovers explicit recency-shift (entrenched career change) "
    "as a cliff at exponent p=0, not a blend. The same drop produces 20% more "
    "false updates under the classifier's real error structure, because the "
    "double V<sub>d</sub> charge was insurance against a mislabeled stable trait "
    "plus weak evidence. Composite matches homeostatic's false-update rate "
    "(94.9%, 0.93 false updates / 18) while keeping the recency-shift win. "
    "Defining surprise as leftover after anticipation makes a predicted weak "
    "stream go quiet; catching that stream is a sleeptime job on time-decayed "
    "belief mass, not a live EMA of raw mismatch. Sixteen daily weak mentions "
    "supersede overnight; the same sixteen monthly do not."
    "<br/><br/>"
    "The overwrite law is only reached after a match. Through "
    "<i>remember(text)</i>, decision error once routed is about six points; "
    "similarity and linking dominate. Topic similarity is non-separable for "
    "must-link versus must-not-link pairs. Two-stage recall-then-verify with a "
    "conservative local model takes irreversible errors to zero on a combined "
    "update+coexist harness. We do not claim a public-benchmark win. We claim a "
    "measured decomposition: prior and residual are different jobs; a switch "
    "beats a blend; linking sits in front of both laws."
)

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
                       fontSize=18, leading=22, alignment=TA_CENTER,
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
CAP = ParagraphStyle("cap", parent=BODY, fontName="Times-Italic", fontSize=8.5,
                     leading=11, textColor=colors.HexColor("#555555"),
                     spaceBefore=1, spaceAfter=10, alignment=TA_CENTER)
CELL = ParagraphStyle("cell", parent=ss["Normal"], fontName="Times-Roman",
                      fontSize=8, leading=10, alignment=TA_LEFT)
CELL_C = ParagraphStyle("cell_c", parent=CELL, alignment=TA_CENTER)
ABSTRACT_STYLE = ParagraphStyle(
    "abstract", parent=None, fontName="Times-Roman", fontSize=10, leading=14,
    alignment=TA_JUSTIFY, spaceAfter=6, leftIndent=8, rightIndent=8,
)
ABSTRACT_LABEL = ParagraphStyle(
    "abstract_label", parent=None, fontName="Times-Bold", fontSize=11,
    leading=14, alignment=TA_CENTER, spaceBefore=4, spaceAfter=6,
)


def P(t, style=BODY):
    return Paragraph(t, style)


def F(t, block=False):
    return Paragraph(t, FORMULA_BLOCK if block else FORMULA)


def C(t, center=False):
    return Paragraph(t, CELL_C if center else CELL)


def bullets(items):
    return [P(f"&bull;&nbsp;&nbsp;{t}") for t in items]


def refs(items):
    return [P(t, REF) for t in items]


REFERENCES = [
    "[1] Emate, R. Volatility-Adjusted Memory Protection: A causal control "
    "knob for continual learning and LLM agent memory. "
    f"<i>Zenodo</i>, 2026. <link href=\"{PRIOR_DOI_URL}\">{PRIOR_DOI_URL}</link>.",
    "[2] Kirkpatrick, J., et al. Overcoming catastrophic forgetting in neural "
    "networks. <i>PNAS</i>, 114(13):3521&ndash;3526, 2017.",
    "[3] Friston, K. The free-energy principle: a unified brain theory? "
    "<i>Nature Reviews Neuroscience</i>, 2010.",
    "[4] Yu, A. J., &amp; Dayan, P. Uncertainty, neuromodulation, and "
    "attention. <i>Neuron</i>, 2005.",
    "[5] Liakoni, V., et al. Adaptive learning and decision-making under "
    "uncertainty by metaplastic synapses guided by a surprise detection "
    "system. <i>eLife</i>, 2017.",
    "[6] Li, T., et al. SuRe: Surprise-driven prioritised replay for "
    "continual LLM learning. <i>arXiv:2511.22367</i>, 2025.",
    "[7] Farhang, A., et al. Surprise as a signal for plasticity and "
    "metacognition. <i>arXiv:2606.31495</i>, 2026.",
    "[8] Sterling, P. Allostasis: a model of predictive regulation. "
    "<i>Physiology &amp; Behavior</i>, 2012.",
    "[9] Casali, A. G., et al. A theoretically based index of consciousness "
    "independent of sensory processing and behavior. "
    "<i>Science Translational Medicine</i>, 2013.",
    "[10] Chhikara, P., et al. Mem0: Building production-ready AI agents "
    "with scalable long-term memory. <i>arXiv:2504.19413</i>, 2025.",
]


def make_table(data, col_widths=None, highlight_col=None):
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#33415c")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f2f4f7")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9ced6")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
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
    canvas.drawCentredString(
        A4[0] / 2.0, 12 * mm,
        f"Prior versus Residual  ·  page {doc.page}",
    )
    canvas.restoreState()


def build():
    story = []

    story += [
        P("Prior versus Residual", TITLE),
        P("Homeostatic and allostatic control laws for agent memory updates",
          SUB),
        P("Richard Emate", SUB),
        P("richard@theemate.com &middot; Independent &middot; 16 August 2026",
          SUB),
        P("Preprint. Empirical companion to Emate (2026), "
          f"<link href=\"{PRIOR_DOI_URL}\">doi:{PRIOR_DOI}</link>. "
          "Results from VoltMem 0.4.0. Not a theory of consciousness.", NOTE),
    ]

    story += [P("Abstract", ABSTRACT_LABEL)]
    story += [P(ABSTRACT, ABSTRACT_STYLE)]
    story += [Spacer(1, 6)]

    story += [P("1. Introduction", H1)]
    story += [P(
        "A memory layer for LLM agents has to decide, on each write, whether a "
        "new sentence updates a stored fact or sits beside it. Facts do not "
        "share a timescale. Personality traits should resist a single offhand "
        "comment. Locations should move. Jobs sit in between: they change, but "
        "not every Tuesday.")]
    story += [P(
        "VoltMem&nbsp;[1] scales protection by a domain volatility prior "
        "V<sub>d</sub> &isin; (0, 1]. Low V<sub>d</sub> raises the bar for "
        "overwrite; high V<sub>d</sub> lowers it. The same idea, applied to "
        "Elastic Weight Consolidation&nbsp;[2], is a causal control knob on the "
        "stability&ndash;plasticity tradeoff, not a free-lunch accuracy win: "
        "shuffling or inverting the domain&rarr;volatility map must degrade "
        "performance monotonically (REAL &gt; SHUFFLE &gt; SWAP). That preprint "
        "treats V<sub>d</sub> as one sufficient statistic for how readily a "
        "memory should move.")]
    story += [P(
        "It is not. A channel can be historically volatile and currently "
        "well-predicted, or historically stable and currently in a regime "
        "change. One scalar cannot see the difference. Continual-learning "
        "instinct says volatile weights should stay plastic. "
        "Predictive-processing instinct says a residual of expected size is "
        "not a reason to reopen a belief&nbsp;[3,&nbsp;4]. Both uses of "
        "V<sub>d</sub> are valid. They must not share a multiplier.")]
    story += [P("This paper tests that split on the VoltMem write path.")]
    story += [P("Contributions.", H2)]
    story += bullets([
        "We show that the homeostatic law <i>double-charges</i> V<sub>d</sub> "
        "(numerator of E<sub>t</sub> and denominator of &theta;<sub>t</sub>), "
        "and that this is load-bearing for two opposite failures: it blocks "
        "explicit career change after entrenchment, and it insures mislabeled "
        "stable facts against weak evidence.",
        "We define online surprise as leftover mismatch after anticipation, "
        "r<sub>t</sub> = |M<sub>t</sub> &minus; &Ecirc;<sub>t</sub>| / "
        "&sigma;<sub>t</sub>, with V<sub>d</sub> widening &sigma;. An EMA of "
        "raw M<sub>t</sub> is not surprise; it sits on a knife-edge under a "
        "constant weak stream.",
        "We show that blending the V<sub>d</sub> exponent between 0 and 1 "
        "inherits neither win. A discrete <b>composite</b> gate does.",
        "We restore accumulated weak evidence as a sleeptime detector on "
        "time-decayed belief mass. Spacing, not count, is the diagnostic.",
        "We measure that the control law is a small term in end-to-end "
        "<i>remember()</i> error, and that topic similarity cannot separate "
        "same-fact updates from coexisting facts. Two-stage linking, not a "
        "threshold, is the architectural repair.",
    ])
    story += [P(
        "We do not claim that these control laws implement consciousness, "
        "allostasis in Sterling's physiological sense&nbsp;[8], or a new "
        "continual-learning algorithm on Split-MNIST. The four-way EWC "
        "comparison outlined in the research log was not run. The evidence "
        "here is scripted agent-memory probes.")]

    story += [P("2. Related work", H1)]
    story += [P(
        "<b>Uniform and volatility-weighted consolidation.</b> "
        "EWC&nbsp;[2] applies one global elastic penalty, anchored to old "
        "weights. Emate&nbsp;[1] scales that penalty per domain by "
        "V<sub>d</sub> measured before the update. The present paper leaves "
        "retrieval and EWC aside and asks only when a <i>symbolic</i> memory "
        "slot should overwrite.")]
    story += [P(
        "<b>Surprise-gated plasticity.</b> Gating learning on prediction error "
        "is not new. Liakoni et al.&nbsp;[5] accumulate reward-rate mismatch "
        "across timescales and raise synaptic plasticity when unexpected "
        "uncertainty exceeds expected uncertainty. SuRe&nbsp;[6] ranks replay "
        "by negative log-likelihood. Farhang et al.&nbsp;[7] use a predictor "
        "over a frozen encoder both to gate episodic writes and as a "
        "metacognitive signal. Our r<sub>t</sub> is the same family of idea, "
        "applied to a memory <i>slot</i> rather than a synapse or a replay "
        "buffer: surprise is distance from predicted mismatch, not from stored "
        "text, and not from a lifetime counter.")]
    story += [P(
        "<b>Expected versus unexpected uncertainty.</b> Yu and Dayan&nbsp;[4] "
        "distinguish noise the system should already have budgeted for from a "
        "change in the world's contingencies. Domain V<sub>d</sub> is our "
        "expected-uncertainty prior. Residual r<sub>t</sub> is unexpectedness "
        "at a step. Collapsing them is the defect we measure.")]
    story += [P(
        "<b>Agent memory stores.</b> Mem0&nbsp;[10], Zep/Graphiti, and related "
        "systems retrieve by embedding and then add, update, or skip. We do "
        "not compare product quality. We measure a sequential structure they "
        "share with VoltMem: matching happens before the overwrite rule. A "
        "miss never consults the control law.")]

    story += [P("3. Control laws", H1)]
    story += [P(
        "On <i>observe()</i>, an extractor supplies mismatch magnitude "
        "M<sub>t</sub> &isin; [0, 1], source reliability R<sub>t</sub>, and "
        "(optionally) a domain. Let C be confirmation count, G<sub>t</sub> a "
        "goal-delta factor, L<sub>t</sub> load (here 1). Residual evidence is")]
    story += [F("res<sub>t</sub> = (M<sub>t</sub> R<sub>t</sub> / "
                "C<sup>&alpha;</sup>) G<sub>t</sub>, &nbsp;&nbsp; &alpha; = 0.6",
                block=True)]
    story += [P("<b>Homeostatic</b> (the original VoltMem law&nbsp;[1]):")]
    story += [F("E<sub>t</sub> = res<sub>t</sub> &middot; V<sub>d</sub>, "
                "&nbsp;&nbsp; &theta;<sub>t</sub> = &theta;<sub>0</sub> / "
                "V<sub>d</sub> &middot; L<sub>t</sub>", block=True)]
    story += [P(
        "Update iff E<sub>t</sub> &gt; &theta;<sub>t</sub>. V<sub>d</sub> "
        "shrinks the score and raises the bar. For a medium-stable job "
        "(V<sub>d</sub> &asymp; 0.3), evidence is cut to a third while the bar "
        "more than triples.")]
    story += [P("<b>Allostatic:</b>")]
    story += [F("E<sub>t</sub> = res<sub>t</sub>, &nbsp;&nbsp; "
                "&theta;<sub>t</sub> = &theta;<sub>0</sub> / "
                "V<sub>trait</sub> &middot; L<sub>t</sub> &middot; s(m)",
                block=True)]
    story += [P(
        "where V<sub>trait</sub> is the domain prior (not a drifted EMA), and "
        "s(m) &isin; [S<sub>min</sub>, 1] is a decreasing function of recent "
        "leftover surprise. V<sub>d</sub> sets the bar once.")]
    story += [P(
        "<b>Residual surprise.</b> Let &Ecirc; and &sigma; be a per-item "
        "running mean and scale of mismatch, with &sigma; widened by "
        "V<sub>d</sub> so ordinary noise on a volatile channel is expected:")]
    story += [F("r<sub>t</sub> = min(1, |M<sub>t</sub> &minus; "
                "&Ecirc;<sub>t</sub>| / (&sigma;<sub>t</sub> &middot; Z)), "
                "&nbsp;&nbsp; Z = 3", block=True)]
    story += [P(
        "E<sub>t</sub> still uses M<sub>t</sub>. Surprise uses r<sub>t</sub>. "
        "Confirms pull &Ecirc; down. Time decay on a persisted "
        "<i>surprise_at</i> (30-day half-life) is the route back to settled. A "
        "lifetime <i>mismatch_count</i> has no such route; it only ratchets "
        "open.")]
    story += [P(
        "<b>Composite</b> is a switch, not an exponent p &isin; (0, 1):")]
    story += bullets([
        "allostatic if M<sub>t</sub> &ge; 0.85 and the source is an explicit "
        "statement, or if &Ecirc; has been learned and r<sub>t</sub> &ge; 0.5;",
        "otherwise homeostatic.",
    ])
    story += [P(
        "A fresh item has no &Ecirc;, so the first weak blip cannot open the "
        "easy-update path.")]
    story += [P(
        "<b>Sleeptime.</b> Online r<sub>t</sub> asks whether this <i>step</i> "
        "was unexpected. After a few similar asides, &Ecirc; &asymp; M and "
        "r<sub>t</sub> &rarr; 0. Accumulated weak evidence is scored later as "
        "time-decayed mass &sum; M<sub>t</sub> R<sub>t</sub> "
        "&frac12;<super>age/30d</super> against a bar 0.35 / V<sub>d</sub>, "
        "ignoring rows before the last confirm. Lifetime counts are not read.")]

    story += [P("4. Experimental setup", H1)]
    story += [P(
        "Probes are scripted. Unless noted, <i>observe()</i> is called with "
        "<i>domain=</i> and <i>mismatch_magnitude=</i>, so Batteries A&ndash;E "
        "isolate the control law from routing. Negative control on A: REAL "
        "&gt; flat &gt; swap of the V<sub>d</sub> map. Errors on E and J are "
        "typed: a <b>false update</b> or <b>false merge</b> destroys a stored "
        "fact (irreversible); a missed update or duplicate leaves both facts "
        "retrievable.")]
    story += [make_table([
        [C("<b>Battery</b>"), C("<b>What it asks</b>"), C("<b>Path</b>")],
        [C("A"), C("Retain/update labels under real / flat / swap priors"),
         C("<i>observe(domain, M)</i>")],
        [C("C"), C("Explicit recency-shift (career change after quiet; "
                   "preference control)"), C("same")],
        [C("D"), C("Weak slow-burn (sixteen casual mentions; daily vs monthly)"),
         C("same")],
        [C("E"), C("Classifier label noise at the real confusion structure, "
                   "and at 50% mislabel"), C("same")],
        [C("F/J"), C("End-to-end <i>remember(text)</i>: update + coexist"),
         C("matcher + law")],
        [C("G/H"), C("Must-link vs must-not-link pairs, held-out 56"),
         C("linking only")],
    ], col_widths=[22 * mm, 105 * mm, 38 * mm])]
    story += [P(
        "The heuristic classifier is &asymp;84% accurate on a 230-utterance "
        "corpus. Ground truth on E stays tied to the <i>true</i> domain: a "
        "mislabel does not change whether the memory ought to update.", CAP)]

    story += [P("5. Results", H1)]
    story += [P("5.1  Two ingredients, two jobs", H2)]
    story += [P(
        "Sweeping the V<sub>d</sub> exponent in E<sub>t</sub> (p=1 "
        "homeostatic, p=0 allostatic) crossed with s(m) on/off: only p=0 "
        "recovers recency-shift, and it is a cliff. At p=0 the entrenched "
        "career change clears the bar by ~27%; at p=0.25 it misses by ~6%. "
        "s(m) changed no outcome at any p on Battery C, because every C probe "
        "ends in explicit M=0.90, which clears a medium-band &theta;-cap "
        "regardless of surprise.")]
    story += [P(
        "Battery A remains 20/20 under real priors with REAL &gt; flat &gt; "
        "swap intact.")]

    story += [P("5.2  The double charge is insurance", H2)]
    story += [P("At the classifier's real error rate:")]
    story += [make_table([
        ["law", "accuracy", "false updates / 18"],
        ["homeostatic", "94.9%", "0.93"],
        ["allostatic", "93.8%", "1.12"],
        ["composite", "94.9%", "0.93"],
    ], col_widths=[50 * mm, 40 * mm, 50 * mm], highlight_col=1)]
    story += [P(
        "Allostatic loses 1.1 points and produces 20% more false updates. "
        "Under 50% mislabel the gap widens (3.44 vs 2.84 false updates). Every "
        "allostatic-only failure has the same shape: a very-stable fact "
        "(<i>personality_trait</i> V=0.05, <i>biographical</i> V=0.10) misread "
        "into a more volatile band, then contradicted by weak evidence. "
        "Removing V<sub>d</sub> from E<sub>t</sub> fixes career changes and "
        "breaks mislabeled traits. It is a trade. Partial p buys nothing "
        "(cliff at 0). Composite is the remaining option: it matches "
        "homeostatic on E and allostatic on C.")]

    story += [P("5.3  Surprise is leftover, not raw difference", H2)]
    story += [P(
        "An EMA of raw M<sub>t</sub> under a constant weak stream drove "
        "&theta; from 0.500 to 0.2941 against E<sub>t</sub> = 0.2940 &mdash; "
        "2e-4 above the trigger. Nine of ten S<sub>min</sub> &times; half-life "
        "settings caught the change; the shipped 14-day half-life was the "
        "failing corner. That is calibration of a quantity that is not "
        "surprise.")]
    story += [P("After r<sub>t</sub> = |M<sub>t</sub> &minus; "
                "&Ecirc;<sub>t</sub>| / &sigma;<sub>t</sub>:")]
    story += [make_table([
        [C("<b>battery</b>"), C("<b>allostatic</b>")],
        [C("C recency-shift"),
         C("hold (still rides dropping V<sub>d</sub> from E<sub>t</sub>)")],
        [C("D daily weak stream"),
         C("never &mdash; after a few hits &Ecirc; &asymp; M, "
           "r<sub>t</sub> &rarr; 0")],
        [C("E label noise"),
         C("unchanged (fresh item, s(m) unread)")],
    ], col_widths=[50 * mm, 115 * mm])]
    story += [P(
        "Same words after two weeks of asides are not surprising. That is the "
        "definition working. Catching the pile is not a live-EMA job.", CAP)]

    story += [P("5.4  Composite and sleeptime", H2)]
    story += [make_table([
        ["battery", "composite"],
        ["A real priors", "20/20"],
        ["C recency-shift", "hold (career U, preference R)"],
        ["D live weak stream", "never"],
        ["E real label noise", "94.9% / 0.93 FU (identical to homeostatic)"],
        ["E 50% mislabel", "84.2% / 2.84 FU (identical to homeostatic)"],
    ], col_widths=[50 * mm, 115 * mm])]
    story += [P("Sleeptime on the logged pile:")]
    story += [make_table([
        [C("<b>stream</b>"), C("<b>live</b>"),
         C("<b>overnight consolidate()</b>")],
        [C("16 daily weaks, <i>professional_context</i>"), C("never", True),
         C("supersedes", True)],
        [C("same 16, monthly"), C("never", True), C("does not rewrite", True)],
        [C("16 daily weaks, <i>core_preference</i>"), C("never", True),
         C("does not rewrite", True)],
    ], col_widths=[80 * mm, 30 * mm, 55 * mm])]
    story += [P(
        "Identical evidence, identical count, only spacing differs. A counter "
        "that never decays cannot see this. Time-decayed belief mass can.",
        CAP)]

    story += [P("5.5  The law is behind the matcher", H2)]
    story += [P("<i>remember(text)</i> only, 18 update probes:")]
    story += [make_table([
        ["similarity", "labels", "routed", "correct", "correct | routed"],
        ["keyword", "oracle", "22.2%", "22.2%", "100%"],
        ["hashing", "oracle", "27.8%", "27.8%", "100%"],
        ["sentence-transformers", "oracle", "77.8%", "72.2%", "92.9%"],
        ["sentence-transformers", "shipped", "55.6%", "44.4%", "80.0%"],
    ], col_widths=[48 * mm, 28 * mm, 26 * mm, 26 * mm, 37 * mm])]
    story += [P(
        "Error budget: similarity &asymp; 50 points; classifier-via-routing "
        "&asymp; 28; residual routing &asymp; 22; decision once routed "
        "&asymp; 5.6. Batteries A&ndash;E cannot see the first three. Two of "
        "four probes that never route even with embeddings are the explicit "
        "career-change and goal-change cases allostatic was built to recover. "
        "End-to-end allostatic vs homeostatic at the best configuration is "
        "77.8% vs 72.2% &mdash; one probe in 18, not significant.", CAP)]

    story += [P("5.6  No threshold separates topic from identity", H2)]
    story += [P(
        "Held-out 56 pairs (28 must-link, 28 must-not-link). Ranking is "
        "inverted: &ldquo;proficient in Python&rdquo; vs &ldquo;proficient in "
        "Japanese&rdquo; scores 0.80 on the default scorer; the career change "
        "that must link scores 0.25. Non-separability replicates on keyword, "
        "hashing, and sentence-transformers. Embeddings double must-link "
        "recall (12/28 &rarr; 25/28) at almost no change in false-merge "
        "<i>count</i> (13/28 &rarr; 12/28), and raise <i>severity</i>: keyword "
        "false merges mostly discard the incoming fact; embedding false merges "
        "supersede the stored one.")]
    story += [P(
        "Stage-1 recall at bar 0.20 is 27/28 held-out. A perfect verifier "
        "therefore scores 55/56. A cheap lexical verifier (cardinality + "
        "change marker) is 19/24 on the fitted split and 35/56 held-out &mdash; "
        "worse than the embedding ladder (41/56). An LLM verifier asking "
        "&ldquo;same subject?&rdquo; and &ldquo;same attribute?&rdquo; "
        "(attribute = the question a fact answers, never the answer) scores "
        "52/56 hosted (<i>gpt-4o-mini</i>) and 49/56 local "
        "(<i>qwen2.5-coder:14b</i>) with <b>0 false merges</b>. The hosted "
        "model is three pairs higher and two irreversible losses worse. Under "
        "the paper's error taxonomy the local model is preferred. Prompt "
        "framing moved the hosted model from 29/56 to 52/56; the local model "
        "moved 48 &rarr; 49 and never collapsed on the badly posed prompt. A "
        "3B rubber-stamp (24 false merges) and a 14B instruct model that "
        "almost always says KEEP_BOTH (5/28 must-link) show that model choice "
        "does not track &ldquo;bigger is better.&rdquo;")]
    story += [P("On the combined 18 update + 14 coexist harness:")]
    story += [make_table([
        [C("<b>configuration</b>"), C("<b>update</b>", True),
         C("<b>coexist</b>", True), C("<b>overall</b>", True),
         C("<b>irrev.</b>", True)],
        [C("keyword, shipped labels"), C("22.2%", True), C("64.3%", True),
         C("40.6%", True), C("5", True)],
        [C("keyword, oracle labels"), C("22.2%", True), C("35.7%", True),
         C("28.1%", True), C("<b>9</b>", True)],
        [C("embeddings, shipped"), C("44.4%", True), C("78.6%", True),
         C("59.4%", True), C("4", True)],
        [C("embeddings, oracle"), C("72.2%", True), C("71.4%", True),
         C("71.9%", True), C("4", True)],
        [C("embeddings + verify, either labels"), C("55.6%", True),
         C("100%", True), C("75.0%", True), C("<b>0</b>", True)],
    ], col_widths=[62 * mm, 26 * mm, 26 * mm, 26 * mm, 25 * mm],
        highlight_col=4)]
    story += [P(
        "Under the threshold ladder, improving the classifier makes data loss "
        "<i>worse</i>: correct labels put two distinct facts in the same slot. "
        "Two-stage linking removes that incentive (coexist 100% either label "
        "condition). Allostatic vs homeostatic on this combined set is 78.1% "
        "vs 75.0%, zero irreversible errors either way &mdash; still one-probe "
        "scale.", CAP)]

    story += [P("6. Discussion", H1)]
    story += [P(
        "The prior and the residual are different jobs. Charging V<sub>d</sub> "
        "twice is not a bug in general: it is the wrong setting for an explicit "
        "correction of an entrenched medium-stable fact, and the right setting "
        "for weak evidence on a misfiled trait. A blend does not interpolate "
        "those cells. A gate does.")]
    story += [P(
        "Online surprise must not be an average of contradiction. If it is, a "
        "predicted stream impersonates a regime change until a half-life is "
        "nursed across a line. Leftover-after-anticipation habituates, which "
        "is what the definition asked for, and it therefore cannot catch a "
        "slow pile. Horizon belongs to a decayed accumulator, preferably off "
        "the interactive path.")]
    story += [P(
        "None of that matters if the new sentence never finds the slot. The "
        "control-law debate is about six points in an error budget whose first "
        "fifty are the similarity function. We report that not as a reason to "
        "abandon the law, but as a reason not to over-read end-to-end "
        "allostatic-vs-homeostatic tables.")]

    story += [P("7. Limitations", H1)]
    story += bullets([
        "Probes are synthetic and small. Battery F/J rows other than the 50- "
        "and 28-point gaps should not be read individually.",
        "Recency-shift still sits on a hand-tuned &theta;-cap "
        "(<i>EXPLICIT_E_RATIO</i>). Robustness of p=0 to domain "
        "misclassification was the original open risk; composite addresses "
        "the measured cell (weak evidence on a settled item) rather than "
        "proving robustness in general.",
        "The Split-MNIST four-variant design (EWC vs V<sub>d</sub> vs residual "
        "vs residual+mode-switch, with forgetting-after-recency as the key "
        "diagnostic) was not executed.",
        "Related work on surprise-gated <i>weight</i> plasticity is active; we "
        "did not re-run those methods on our probes. The claim is about a "
        "symbolic memory slot.",
        "Linking results depend on a 24/56 split and on prompt wording for "
        "the hosted verifier. Held-out pairs were written from a structural "
        "grid after the first 24; they are not a public benchmark.",
    ])

    story += [P("8. Conclusion", H1)]
    story += [P(
        "A volatility prior answers how readily a <i>kind</i> of memory should "
        "move. Leftover surprise answers whether this step was already priced "
        "in. Using one number for both produces a characteristic pair of "
        "failures, and blending the number does not fix the pair. A switch, a "
        "residual that habituates, and a sleeptime accumulator that sees "
        "spacing, together implement the split. The matcher still sits in "
        "front. Duplicates are recoverable; silent overwrites are not.")]

    story += [P("References", H1)]
    story += refs(REFERENCES)

    story += [hr()]
    story += [P(
        "Code: <b>github.com/Rouche01/voltmem</b> &middot; "
        "Package: <b>voltmem 0.4.0</b> &middot; "
        f"Companion: <b>{PRIOR_DOI_URL}</b>",
        CAP)]

    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title="Prior versus Residual: Homeostatic and Allostatic Control Laws "
              "for Agent Memory Updates",
        author="Richard Emate",
        subject=f"Companion to {PRIOR_DOI_URL}",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
