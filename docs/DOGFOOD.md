# Dogfood — Quick Question

Primary real-app test for VoltMem. Sibling repo: `../quick-question`
(same `Projects/` folder). Product spec lives there:
`docs/quick-question-mvp-spec.md`. Cursor plan (what to build next):
`../quick-question/.cursor/plans/current-truth-vault.plan.md`.

This file is the **library-side** plan: what we are trying to prove, how
the product should work, what to ship in 1–2 weeks, and high-value ideas
that are easy to miss.

Keep language plain. Newest thinking belongs in the sections below and
in [`STATUS_LOG.md`](STATUS_LOG.md).

---

## What this product is

Quick Question is a local, user-confirmed picture of *you* that AI agents
can load on a cold start. It does not try to be a diary, a second brain,
or a dump of every chat and calendar you have ever had. It asks a few
high-stakes questions (and later, light daily checks), turns only what
you confirm into durable facts, and keeps that vault on your machine so
other tools can read current truth without a cloud profile of your life.

The aim is that an agent already knows the load-bearing rules — what you
are protecting, what is not allowed to be sacrificed, when you do your
best work, what is no longer true — instead of starting empty or
guessing from noisy history. When your life changes, you confirm the
new rule and the old one should stop driving advice. Connected apps and
existing AI chats can *suggest* candidates; they never silently become
memory.

VoltMem is the engine underneath: different kinds of facts are sticky in
different degrees, so a passing preference does not overwrite a value,
and a real change can still land. The product is the loop that makes
that felt — confirm, load, advise, correct — not the questionnaire
itself.

---

## One-line bet

Quick Question should make VoltMem *felt*: an agent that starts empty,
then speaks from a small set of **confirmed current-truth rules**, and
**drops the old answer** when you change your mind — without dumping
chat logs or calendar noise into memory.

---

## Why this app (not stylens / relay first)

| Fit | Why |
|---|---|
| Same job as the library | “Current truth for agents,” not fashion vocabulary |
| Own domains file | `voltmem/domains.json` — value / goal / habit / preference / circumstance |
| Confirm gate | Suggestions never become agent memory |
| Real stack | Browser → `@voltmem/client` → local sidecar → MCP cold-start |
| Built-in tension | Career vs protected evenings — a stable-fact-that-changed story |

Stylens / relay remain useful later for app-specific vocab. They are
not the clearest showcase of the control law.

**Current QQ state:** visual prototype + MVP-V wiring (outbox, domains,
cold-start pack, read-only MCP). The spine is still a short multiple-choice
quiz. That is too thin to “understand” someone. The dogfood is to thicken
the *right* loop, not to add more screens.

---

## How understanding should work

Three inputs. Only one is allowed to write agent-visible memory: **you
confirming**.

### 1. Questions (primary)

Ask **tradeoffs in a real situation**, not identity labels.

Bad: “What are you hoping to achieve this year?” → “Advance my career.”  
Good: “When career work spills into the evening, what should win?”

Onboarding ≈ five **decision rules**:

1. When two good things collide this month, what should an agent protect?
2. What is not allowed to be sacrificed right now?
3. What used to be true that is no longer true?
4. When should the agent push you vs leave you alone?
5. One sacred hour this week.

Daily is not “which evening vibe looks like you?” Better activities:

- Correct a plan (“I’d move the 7pm task — wrong?”)
- **Still true?** on one sealed fact (10 seconds)
- Pick which of two conflicts wins
- Reject a suggestion (the no is as useful as the yes)

### 2. Connected apps (evidence only)

Apps say *what happened*. They do not say *what should win*.

Connect **calendar first** (even mocked) to *propose* suggestions.
User confirms. Disconnect must not wipe confirmed facts.

Skip high-creep sources (full email, location trails) until the confirm
loop is addictive.

### 3. Existing AI chats (richest mine, easiest to poison)

ChatGPT / Claude / Cursor threads already contain goals, hates, constraints.

**Do not** zip-import transcripts into VoltMem. That is noisy RAG, which
is the failure mode this library exists to refuse.

Do this instead:

1. Paste a thread, or harvest at session end.
2. Extract 3–7 **candidates** (`suggested`).
3. User confirms / edits / dismisses.
4. Then write, with an explicit `domain`.

Best near-term path: **QQ’s own MCP in reverse** — after a Cursor/Claude
session, propose facts to confirm. No ChatGPT OAuth required.

### The loop that makes it a product

**Surprise → one question.**

You said protect evenings. Life (calendar, a chat, your own booking)
keeps putting work at 8pm. Do not silent-overwrite. Ask:

> Career is spilling into evenings. Still move work to tomorrow, or is
> this a sprint?

That is VoltMem’s write policy in UI form: sticky facts stay; change
needs a clear, confirmed escalation.

---

## 1–2 week wow (one path)

Possible. Cut everything else.

| Days | Ship |
|---|---|
| 1–3 | Live “Ask your agent” that loads the cold-start pack from VoltMem |
| 3–5 | Scripted before → after confirm → after you change the tension |
| 5–7 | One-command sidecar; Settings `connected · synced`; private excluded |
| 8–10 | You use it daily; fix the 2–3 breaks in that story |

**Success test (say it without notes):**

> I confirmed I protect evenings. The agent stopped suggesting late
> work. I flipped the career tension — it stopped citing the old rule
> on the next ask.

**Out of scope for the wow:** multi-device sync, real Calendar OAuth,
mobile, cloud classifier, pretty graph editing, bulk chat import.

---

## What this dogfood should prove about VoltMem

| Library surface | How QQ tests it |
|---|---|
| Explicit `domain` on write | Category → `goal` / `value` / … skips the heuristic classifier |
| `VOLTMEM_DOMAINS_FILE` | App-owned priors, not library vocab |
| Tenant isolation | `quick-question-local` |
| Search + summary | Cold-start pack / MCP |
| Current-truth update | Tension reopen / “still true?” / correction |
| Confirm vs suggest | Suggestions never in the sidecar |
| Offline outbox | Confirm while sidecar down → flush → pack updates |
| Private exclusion | `qq:…:private` out of agent pack |
| Optional later: TTL | Calendar-like circumstances with an end date |
| Optional later: sleeptime | Twins / tensions as inbox suggestions |

Use the sidecar **memory browser** (`/ui`) as the operator view of the
same tenant.

---

## Insights that are easy to miss

These came out of the 2026-10-21 thread plus reading the QQ vault code.
Worth exploring; none are “build this week” except where noted.

### 1. QQ is currently bypassing the control law on update

Outbox upsert is **delete old VoltMem id → add new**. That makes the UI
look like current truth, but it never runs escalation, twins, or
mismatch evidence. The showcase can look like “a key-value store with
domains.”

**Explore:** upsert via `remember` / observe of the same subject so a
changed evening rule is an *update*, not a client-side delete. That is
the actual library.

### 2. Dogfood the pack, not a search box

VoltMem’s weak spot on public benches is vague queries (similarity
plateaus). QQ already builds a **small ranked pack** (summary, values
first, tensions). Agents should load that pack by default.

Search is for “what’s my constraint about X?”, not for “who is this
person?” A generic search sample
(`goals preferences habits constraints values`) does not show freshness
re-rank. **The product can dodge Problem 3 and call that a feature.**

### 3. Tensions never leave the browser

Insight links live in localStorage. MCP notes this: Cursor agents get
facts + summary, **not** the career-vs-evenings link. The most
VoltMem-shaped object in the app is invisible to the agent plane.

**Explore:** persist confirmed tensions as first-class memories (or
graph edges) in VoltMem so every MCP client sees the boundary rule.

### 4. Put the negative control in the UI

VoltMem’s research honesty is `real > flat > swap`. The wow screen
should toggle:

- empty context
- QQ/VoltMem pack
- last N raw chat turns (the naive competitor)

Same question, three answers. That is a product demo and a science demo.

### 5. Domain priors may be backwards for this life

QQ sets `value` 0.1, `goal` 0.15, `preference` 0.35. A *this-year goal*
often changes faster than “protect evenings.” If updates feel wrong,
tune the domains file — that is exactly `VOLTMEM_DOMAINS_FILE`.

### 6. Calendar is a TTL teacher, not a personality teacher

Recurring holds and “until Friday” map to `expires_at` / `ttl_seconds`,
which QQ does not send yet. One dated circumstance is a cleaner
VoltMem demo than five vibe preferences.

### 7. Sleeptime should appear as an inbox, not a daemon log

`reconcile_twins`, `pattern_audit`, consolidate are built. QQ already
has a suggestions inbox. **Explore:** maintenance proposes “these two
might be the same” / “this may be stale” → same confirm gate. That
connects library sleeptime to the product without a new mental model.

### 8. The highest-volume future source is corrections, not onboarding

After the agent answers, “no, I don’t work nights” is a better fact
than question 1. Build **correction → suggested update → confirm**
before building more connectors.

### 9. Bidirectional MCP is the AI-chat strategy

Today MCP is read-only (`get_cold_start_pack`). The harvest path is
write-propose: agent session ends → 3 candidates in QQ. That beats
ChatGPT export UX and stays local-first.

### 10. Small vault is the feature

Understanding is not a second brain. Cap agent-visible facts (order of
**15–40** load-bearing rules). VoltMem is a policy layer, not a diary.
Hoarding fights current-truth and makes the pack worse.

### 11. “Still true?” is escalation as UX

Sealed onboarding + reopen on gap / tension / stale is already in the
QQ spec. Productize that as the daily. It is the human form of
cumulative mismatch + explicit override.

### 12. Human classifier vs library classifier

Explicit domains on write **skip** VoltMem’s heuristic. That is
correct for QQ and means this dogfood will **not** stress Problem 1
(keyword collisions) unless you add a free-text harvest path without
a domain. If you want to test the classifier, that is a separate slice
(chat distill with model-proposed domains, user can override).

---

## Explicitly do not do (yet)

- Silent LLM → fact writes
- Bulk ChatGPT/Claude history import as memory
- Public web app talking to a laptop sidecar
- API keys in a hosted client
- Stylens vocabulary inside VoltMem (already removed; keep it that way)
- Replacing the confirm gate with “the model is pretty sure”

---

## How to run the pair

From the VoltMem checkout (see QQ README for Node 24):

```bash
pip install -e ".[sidecar]"
export VOLTMEM_API_KEY=dev-secret
export VOLTMEM_DB_PATH=./voltmem_sidecar.db
export VOLTMEM_EMBEDDINGS=0
export VOLTMEM_DOMAINS_FILE=/absolute/path/to/quick-question/voltmem/domains.json
export HOST=127.0.0.1
export PORT=8080
python -m sidecar
```

QQ Vite proxies `/voltmem` → `:8080`. Tenant default:
`quick-question-local`. Browser UI: sidecar `/ui`.

---

## Open questions

- Should updates go through VoltMem match/escalation, or is client
  delete+add acceptable for v1 dogfood UX?
- Are tensions memories, links, or both?
- One tenant per human, or per device/app install?
- When (if ever) does a local LLM propose domains vs always-forced
  categories?

Record decisions in [`STATUS_LOG.md`](STATUS_LOG.md) when they land.
