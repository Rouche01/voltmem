"""
Battery G corpus — must-link / must-not-link pairs, split dev vs held-out.
=========================================================================

Why this file exists separately from the evaluators
---------------------------------------------------
The cheap verifier in ``link_verify_prototype.py`` (single-valued attribute OR
explicit change marker) scored 19/24 against the original 24 pairs, one point
above the best score ANY single similarity threshold can reach. That looked like
evidence the two-stage design works. It is not trustworthy evidence: both of the
verifier's signals were chosen after reading which of those 24 pairs failed, so
19/24 partly measures hindsight rather than generalisation.

So the corpus is split, and the split is load-bearing:

  DEV       the original 24. The verifier was fitted to these. Its score here is
            an upper bound, quotable only as "not worse than before".
  HELD-OUT  56 pairs written afterwards, from a structural grid rather than from
            the verifier's failure list. This is the only honest number.

How the held-out pairs were chosen
----------------------------------
Enumerate the grid that actually drives the two cheap signals and fill every
cell, including the cells where the signals must fail by construction:

  cardinality   is the new statement's domain single-valued (SLOT_DOMAINS)?
  marker        does it carry replacement language ("no longer", "now", ...)?
  overlap       lexical near-copy, or paraphrase with little shared vocabulary?

Two cells are guaranteed losses for the cheap verifier and are populated on
purpose, because they are the cases production will actually hit:

  must-link, multi-valued domain, no marker
      "User was born in 1990" -> "User was born in 1991". A correction to a
      single-valued ATTRIBUTE sitting inside a multi-valued DOMAIN. Cardinality
      is being asked at the wrong granularity.
  must-not-link, single-valued domain, distinct facts
      Two pending to-dos; an appointment and a flight; the user's city and their
      parents' city. Cardinality says "replace" and destroys a true memory.
      The dev set contained only one pair of this shape, which is why the
      signal looked stronger than it is.

The grid is computed from the pairs at evaluation time rather than hand-labelled
here, so a cell cannot be mislabelled to flatter a verifier.

Consumed by ``linking_eval.py`` (Battery G) and ``link_verify_prototype.py``.
"""

# (stored_domain, stored_text, new_domain, new_text, note)

# ── DEV: the original 24. The cheap verifier's signals were fitted to these. ──

DEV_MUST_LINK = [
    ("location", "I live in Berlin",
     "location", "I live in Paris",
     "lexical near-copy (what the passing test relies on)"),
    ("location", "User lives in Berlin",
     "location", "User moved to Paris last month",
     "reworded relocation"),
    ("emotional_context", "User is feeling stressed",
     "emotional_context", "User says they feel calm and relaxed now",
     "mood flip, little shared vocabulary"),
    ("current_task", "User is preparing the Monday slides",
     "current_task", "User finished the slides and is now writing the report",
     "task moved on"),
    ("professional_context", "User works as a data analyst",
     "professional_context",
     "User explicitly said they changed careers and now work as a nurse",
     "Battery F failure: career change, misses slot bar by 0.02"),
    ("long_term_goal", "User wants to become a research scientist",
     "long_term_goal",
     "User explicitly said their goal is now to start a company",
     "Battery F failure: goal change, genuine semantic miss"),
    ("core_preference", "User prefers concise, direct answers",
     "core_preference",
     "User explicitly said they now prefer long, detailed explanations",
     "Battery F failure: preference flip"),
    ("biographical", "User grew up in Lagos",
     "biographical", "User explicitly said they grew up in Nairobi",
     "Battery F failure: biographical correction"),
    ("skill", "User is proficient in Python",
     "skill",
     "User explicitly said they no longer use Python and work in Rust now",
     "skill replacement"),
    ("relationship", "User works closely with Alice",
     "relationship", "User explicitly said they no longer work with Alice",
     "relationship ended"),
    ("core_preference", "I prefer concise, direct answers",
     "core_preference", "I really like short replies",
     "paraphrase the mocked test pretends to score 0.50"),
    ("personality_trait", "User is deeply introverted",
     "personality_trait", "User explicitly said they are now highly extroverted",
     "trait flip"),
]

DEV_MUST_NOT_LINK = [
    ("core_preference", "User prefers dark mode",
     "core_preference", "User prefers concise, direct answers",
     "two distinct preferences, both true"),
    ("skill", "User is proficient in Python",
     "skill", "User is proficient in Japanese",
     "TRAP: near-identical wording, two distinct skills"),
    ("skill", "User is learning Spanish",
     "skill", "User is learning to cook",
     "TRAP: shared 'is learning', unrelated skills"),
    ("relationship", "User's manager is Alice",
     "relationship", "User's mentor is Bob",
     "TRAP: same frame, different people and roles"),
    ("relationship", "User has a sister named Alice",
     "relationship", "User has a brother named Tom",
     "TRAP: same frame, two distinct family members"),
    ("biographical", "User grew up in Lagos",
     "biographical", "User studied at Oxford",
     "two distinct biographical facts"),
    ("location", "User lives in Berlin",
     "transient_fact", "User is travelling to Tokyo next week",
     "residence vs trip: same topic, different fact"),
    ("biographical", "User's father is a doctor",
     "professional_context", "User works as a data analyst",
     "different subject (father vs user)"),
    ("long_term_goal", "User wants to become a research scientist",
     "current_project", "User is job hunting",
     "adjacent topic, distinct facts"),
    ("core_preference", "User prefers dark mode",
     "stated_preference", "User prefers tea over coffee",
     "TRAP: sibling domains are searched together by the slot fallback"),
    # These two carry an explicit change marker ("no longer") but are about a
    # DIFFERENT entity than the stored fact. They exist to stop a verifier from
    # passing just by looking for replacement language.
    ("relationship", "User works closely with Alice",
     "relationship", "User no longer works with Bob",
     "TRAP: change marker, but a different person"),
    ("skill", "User is proficient in Python",
     "skill", "User no longer studies French",
     "TRAP: change marker, but a different skill"),
]

# ── HELD-OUT: written from the grid, after the verifier was already fixed. ────

HELDOUT_MUST_LINK = [
    # single-valued domain, no replacement language
    ("location", "User lives in Austin",
     "location", "User relocated to Seattle in March",
     "slot, no marker: relocation stated as a fact"),
    ("location", "User lives in Berlin",
     "location", "User lives in Munich",
     "slot, no marker, near-copy: bare city correction"),
    ("emotional_context", "User is overwhelmed",
     "emotional_context", "User is calm",
     "slot, no marker, minimal overlap: hardest recall case"),
    ("emotional_context", "User is in a great mood",
     "emotional_context", "User is frustrated today",
     "slot, no marker: mood flip"),
    ("transient_fact", "User is on a call until 3pm",
     "transient_fact", "The call ended early",
     "slot, no marker: transient fact resolved"),
    # single-valued domain, with replacement language
    ("location", "User is based in Toronto",
     "location", "User is now based in Vancouver",
     "slot + marker"),
    ("current_task", "User is debugging the payment flow",
     "current_task", "User is now writing the migration script",
     "slot + marker: task handoff"),
    ("current_task", "User is reviewing the design doc",
     "current_task", "User wrapped that up and moved on to interviews",
     "slot + marker, low overlap"),
    ("emotional_context", "User is anxious about the deadline",
     "emotional_context", "User feels much better since the deadline passed",
     "slot + marker, low overlap"),
    # multi-valued domain, with replacement language
    ("professional_context", "User works at Stripe",
     "professional_context", "User left Stripe and joined Figma",
     "non-slot + marker: employer change"),
    ("relationship", "User reports to Dana",
     "relationship", "User now reports to Priya",
     "non-slot + marker: reporting line change"),
    ("core_preference", "User prefers video calls",
     "core_preference", "User would rather not do video calls anymore",
     "non-slot + marker: preference withdrawn"),
    ("opinion", "User thinks microservices are overrated",
     "opinion", "User has changed their mind about microservices",
     "non-slot + marker: opinion reversal"),
    ("biographical", "User's native language is Portuguese",
     "biographical", "User clarified their native language is actually Spanish",
     "non-slot + marker: correction of a single-valued attribute"),
    ("long_term_goal", "User plans to move abroad",
     "long_term_goal", "User no longer plans to move abroad",
     "non-slot + marker: goal abandoned"),
    ("core_preference", "User dislikes emoji in responses",
     "core_preference", "User said emoji are fine now",
     "non-slot + marker, low overlap"),
    ("personality_trait", "User is a detail-oriented perfectionist",
     "personality_trait",
     "User describes themselves as a big-picture thinker these days",
     "non-slot + marker: trait shift"),
    ("current_project", "User is working on the mobile app",
     "current_project", "User is now working on the web dashboard",
     "non-slot + marker: project switch"),
    ("skill", "User is learning Go",
     "skill", "User finished the Go course and is comfortable with it now",
     "non-slot + marker: same skill, updated level"),
    # multi-valued domain, NO replacement language — the cheap verifier's
    # structural blind spot. Single-valued ATTRIBUTES inside multi-valued
    # DOMAINS, stated as plain facts.
    ("biographical", "User was born in 1990",
     "biographical", "User was born in 1991",
     "non-slot, no marker: birth year correction"),
    ("professional_context", "User is a backend engineer",
     "professional_context", "User was promoted to engineering manager",
     "non-slot, no marker: role change"),
    ("professional_context", "User works remotely",
     "professional_context", "User is back in the office five days a week",
     "non-slot, no marker: work arrangement change"),
    ("skill", "User is fluent in French",
     "skill", "User has lost most of their French and can barely hold a conversation",
     "non-slot, no marker: skill decayed"),
    ("long_term_goal", "User wants to write a novel",
     "long_term_goal", "User has given up on the novel and wants to teach",
     "non-slot, no marker: goal replaced"),
    ("relationship", "User is married to Sam",
     "relationship", "User and Sam separated",
     "non-slot, no marker: relationship ended"),
    ("current_project", "User is building a recommendation engine",
     "current_project",
     "User shelved the recommender and started a fraud detection system",
     "non-slot, no marker: project replaced"),
    ("personality_trait", "User is highly risk-averse",
     "personality_trait", "User has become much more willing to take risks",
     "non-slot, no marker: trait shift"),
    ("stated_preference", "User wants weekly summaries",
     "stated_preference", "User asked to switch to daily summaries",
     "non-slot, no marker: cadence change"),
]

HELDOUT_MUST_NOT_LINK = [
    # single-valued domain, but genuinely coexisting facts. Cardinality alone
    # says "replace" for every one of these and destroys a true memory.
    ("transient_fact", "User has a dentist appointment on Tuesday",
     "transient_fact", "User has a flight on Friday",
     "TRAP slot: two unrelated near-term facts"),
    ("transient_fact", "User's laptop is in for repair",
     "transient_fact", "User's package arrives tomorrow",
     "TRAP slot: two unrelated transient facts"),
    ("transient_fact", "User is fasting today",
     "transient_fact", "User is on call this weekend",
     "TRAP slot: two concurrent temporary states"),
    ("current_task", "User is drafting the quarterly report",
     "current_task", "User needs to renew their passport",
     "TRAP slot: two concurrent to-dos"),
    ("location", "User lives in Berlin",
     "location", "User's parents live in Hamburg",
     "TRAP slot: different subject, same frame"),
    ("location", "User works in the Munich office",
     "location", "User lives in Berlin",
     "TRAP slot: workplace and residence coexist"),
    ("emotional_context", "User is excited about the trip",
     "emotional_context", "User is stressed about the audit",
     "TRAP slot: two feelings about different things"),
    # multi-valued domain WITH replacement language, but about a different
    # entity or attribute. Defeats the marker signal.
    ("relationship", "User reports to Dana",
     "relationship", "User no longer reports to Miguel",
     "TRAP marker: different person"),
    ("skill", "User is fluent in French",
     "skill", "User has changed their mind about learning Korean",
     "TRAP marker: different skill"),
    ("core_preference", "User prefers dark mode",
     "core_preference", "User no longer wants desktop notifications",
     "TRAP marker: different preference"),
    ("long_term_goal", "User wants to write a novel",
     "long_term_goal", "User switched their marathon goal to a half marathon",
     "TRAP marker: different goal"),
    ("opinion", "User thinks microservices are overrated",
     "opinion", "User changed their view on remote work",
     "TRAP marker: different topic"),
    ("professional_context", "User works at Stripe",
     "professional_context", "User's spouse left their job at Google",
     "TRAP marker: different subject"),
    # multi-valued domain, no marker, distinct coexisting facts. The cell the
    # cheap verifier handles correctly by default.
    ("skill", "User is proficient in Rust",
     "skill", "User is proficient in Kubernetes",
     "two distinct skills"),
    ("skill", "User plays the piano",
     "skill", "User plays tennis",
     "shared frame, unrelated skills"),
    ("relationship", "User has a daughter named Mia",
     "relationship", "User has a son named Leo",
     "two distinct children"),
    ("relationship", "User's cofounder is Ravi",
     "relationship", "User's investor is Chen",
     "same frame, different people and roles"),
    ("biographical", "User studied physics at MIT",
     "biographical", "User served in the navy",
     "two distinct biographical facts"),
    ("biographical", "User is left-handed",
     "biographical", "User is colour-blind",
     "two distinct stable attributes"),
    ("core_preference", "User prefers bullet points",
     "core_preference", "User prefers British spelling",
     "two distinct formatting preferences"),
    ("personality_trait", "User is highly competitive",
     "personality_trait", "User is unusually patient",
     "two distinct traits"),
    ("long_term_goal", "User wants to buy a house",
     "long_term_goal", "User wants to learn to sail",
     "two distinct goals"),
    ("current_project", "User is building a recommendation engine",
     "current_project", "User is renovating their kitchen",
     "two distinct projects"),
    ("professional_context", "User manages a team of six",
     "professional_context", "User is on the hiring committee",
     "two distinct facts about the same job"),
    ("opinion", "User thinks Python typing is worth it",
     "opinion", "User thinks standups are a waste of time",
     "two distinct opinions"),
    # cross-domain and sibling-domain traps
    ("core_preference", "User prefers dark mode",
     "stated_preference", "User wants shorter meetings",
     "TRAP: sibling domains searched together"),
    ("biographical", "User's mother is a lawyer",
     "professional_context", "User works as a paralegal",
     "TRAP: high topical overlap, different subject"),
    ("long_term_goal", "User wants to become a CTO",
     "current_project", "User is mentoring two juniors",
     "adjacent topic, distinct facts"),
]

MUST_LINK = DEV_MUST_LINK + HELDOUT_MUST_LINK
MUST_NOT_LINK = DEV_MUST_NOT_LINK + HELDOUT_MUST_NOT_LINK

SPLITS = {
    "dev (fitted)": (DEV_MUST_LINK, DEV_MUST_NOT_LINK),
    "held-out": (HELDOUT_MUST_LINK, HELDOUT_MUST_NOT_LINK),
    "all": (MUST_LINK, MUST_NOT_LINK),
}

# text -> domain, for the OracleExtractor. Every statement appears with a single
# domain across the whole corpus; a clash here would mean a pair is mislabelled.
TRUTH: dict[str, str] = {}
_CLASHES: list[tuple[str, str, str]] = []
for _p in MUST_LINK + MUST_NOT_LINK:
    for _dom, _txt in ((_p[0], _p[1]), (_p[2], _p[3])):
        _prev = TRUTH.setdefault(_txt, _dom)
        if _prev != _dom:
            _CLASHES.append((_txt, _prev, _dom))
if _CLASHES:
    raise AssertionError(f"same text labelled two ways: {_CLASHES}")
