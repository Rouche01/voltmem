---
name: Tenant semantics + domain profiles
overview: "Clarify VoltMem isolation vs fact-kind: rename public userId → tenantId (path /v1/tenants/…), keep /v1/users/ as alias; ship optional domain on write; load app-specific domains from VOLTMEM_DOMAINS_FILE instead of compiling them into the library."
todos:
  - id: naming-docs
    content: "Glossary in SIDECAR + README — tenant (isolation), domain (kind + volatility), memory item; UI labels Tenant / Domain (kind); stop calling path segment 'user' in new docs"
    status: completed
  - id: api-tenants-alias
    content: "Sidecar — mount /v1/tenants/{tenant_id}/… as canonical routes (same handlers as users); keep /v1/users/{user_id}/… as deprecated alias; OpenAPI descriptions say tenant"
    status: pending
  - id: client-tenantId
    content: "@voltmem/client — prefer tenantId / forTenant(); accept userId as deprecated alias; paths call /v1/tenants/…; bump client minor; changelog"
    status: pending
  - id: domain-on-add
    content: "Optional domain on POST …/memories + Memory.add/remember plumbing + TS AddOptions.domain; skip classifier when set; tests for forced domain"
    status: completed
  - id: relay-profile
    content: "VOLTMEM_DOMAINS_FILE JSON merged at sidecar startup (domains + optional keywords). No app vocabulary compiled into the library. stylens stays the only built-in profile."
    status: completed
  - id: ui-tenant-copy
    content: "Memory browser — Tenant field label; Domain (kind) filter; placeholder example relay-local; optional note that domain is classifier or write-supplied"
    status: completed
  - id: dogfood-handoff
    content: "Note for relay-os / stylens — VOLTMEM_TENANT_ID (+ USER_ID fallback), forward domain from context-engine, ship a domains JSON via VOLTMEM_DOMAINS_FILE (separate consumer PRs)"
    status: pending
isProject: true
---

# Tenant semantics + domain profiles

Canonical plan for **naming clarity** and **app-owned domains**. Open while the `voltmem` workspace is active.

**Related:** memory browser [`.cursor/plans/sidecar_memory_browser.plan.md`](./sidecar_memory_browser.plan.md) · optional `domain` on add also listed under [`.cursor/plans/ts_playground_sibling_11142e17.plan.md`](./ts_playground_sibling_11142e17.plan.md) (`voltmem-domain-api`) — **this plan owns that work**; mark the playground todo done when this lands.

## Problem

1. Public API says `userId` / `/v1/users/{user_id}` but dogfood often uses an **app bucket** (`relay-local`), not a person. Internal store already calls this `namespace`. Operators get thrown by the word “user.”
2. Community Engager stamps `[preference]` / `[outcome]` in text; the sidecar **stylens** profile classifies into `style_preference` / `stated_preference`. The UI domain filter shows classifier output, not Relay’s intent.
3. Library already supports `create_memory(domains=DomainRegistry)` and `remember(..., domain=…)`, but the sidecar HTTP `add` path does not pass `domain`, and only `VOLTMEM_PROFILE=stylens` exists.

## Naming recommendation

| Layer | Prefer | Avoid / deprecate |
|-------|--------|-------------------|
| Public HTTP | `/v1/tenants/{tenant_id}/…` | New docs saying “user” for the path segment |
| TS / docs | `tenantId`, `forTenant()` | `userId` (keep as alias one minor+) |
| Env (consumers) | `VOLTMEM_TENANT_ID` | `VOLTMEM_USER_ID` (fallback) |
| Internal SQLite column | keep `namespace` | Renaming the column (no benefit) |
| Fact kind | `domain` | Calling it a “table” in product copy |

**Why `tenant` over `namespace` or `user`**

- **`tenant`** — matches the mental model (“isolation boundary / database”). Clear for app buckets *and* real end-users (a person can still be a tenant id).
- **`namespace`** — already the store column; using it publicly is accurate but clashes with “Python package namespace” and doubles vocabulary when docs also say tenant. Keep it as: *tenant id is stored as `namespace`.*
- **`user` / `userId`** — wrong default implication for Relay; deprecate.

**Do not** rename the physical column or force a data migration. Path alias + client alias is enough.

```text
tenant_id  ──HTTP/client──►  Memory(user_id=…)  ──►  row.namespace
domain     ──optional on write / else classify──►  row.domain
```

## Mental model (glossary — put in SIDECAR)

| Concept | Role | Example |
|---------|------|---------|
| **Tenant** | Isolation boundary (one logical DB) | `relay-local`, `alice` |
| **Domain** | Fact kind + volatility prior (partition inside tenant) | `community_outcome` |
| **Memory item** | One stored fact | approve/abort outcome text |
| **Profile** | Process-wide registry + classifier installed at sidecar boot | `stylens`, `relay` |

Analogy for docs (soft): tenant ≈ database, domain ≈ typed partition — **not** a second SQLite table.

## Phase 1 — Docs + UI copy (no break)

- [`docs/SIDECAR.md`](../../docs/SIDECAR.md), [`sidecar/README.md`](../../sidecar/README.md), root README multi-tenant blurb: tenant language; note `/users` alias.
- Memory browser: label **Tenant**, **Domain (kind)**; placeholder `relay-local`.

## Phase 2 — HTTP + client rename (compat)

### Sidecar

- Register the same handlers under `/v1/tenants/{tenant_id}/…`.
- Keep `/v1/users/{user_id}/…` indefinitely for one minor (or until 0.6); OpenAPI mark deprecated.
- Path param name in new routes: `tenant_id`.

### `@voltmem/client`

```ts
interface VoltMemClientOptions {
  tenantId?: string;
  /** @deprecated use tenantId */
  userId?: string;
}
```

- Resolve: `tenantId ?? userId`; error text says `tenantId is required`.
- `forTenant(id)` (+ `forUser` deprecated alias).
- Request paths: `/v1/tenants/${encodeURIComponent(tenantId)}/…`.
- Bump client version (minor); CHANGELOG / clients README.

Python in-process `create_memory(..., user_id=)` can gain `tenant_id=` as an alias kwarg in the same release if cheap; not required for sidecar dogfood.

## Phase 3 — Domain on write

Engine already: `remember(text, domain=…)` skips classification when set.

Wire through:

1. Sidecar `AddBody.domain: str | None`
2. `Memory.add` / pool → `remember(..., domain=body.domain)`
3. TS `AddOptions.domain?: string`
4. Tests: POST with `domain=community_outcome` → list/graph show that domain even if text would classify as style

`add_event` facets already carry `domain` — leave as-is; document parity.

## Phase 4 — domains file (no app vocabulary in the library)

`VOLTMEM_DOMAINS_FILE` JSON is merged onto the built-in profile at startup. `stylens` stays the only code profile. Relay (and any later app) ships its own file:

```json
{
  "domains": [
    {"name": "community_preference", "volatility": 0.20},
    {"name": "community_rules", "volatility": 0.15},
    {"name": "community_outcome", "volatility": 0.55}
  ],
  "keywords": {
    "community_outcome": ["[outcome]", "aborted"],
    "community_preference": ["[preference]", "allowlist"]
  }
}
```

Keyword hints run before the profile classifier. One process still has one registry, so two priors means two sidecars, each with its own file.

**Out of scope:** per-tenant domain files, HTTP “create domain” API.

## Phase 5 — Dogfood handoff (not this repo’s merge gate)

Track as notes / issue links; implement in consumer repos after client publish:

| Consumer | Change |
|----------|--------|
| relay-os `@relay/context-engine` | Prefer `tenantId`; env `VOLTMEM_TENANT_ID` with `VOLTMEM_USER_ID` fallback; pass `domain` through to client `add` (stop relying on text prefix alone, or keep prefix as prompt sugar) |
| community-engager | Map `preference` → `community_preference`, `outcome` → `community_outcome` in the domains file |
| Deploy | `VOLTMEM_DOMAINS_FILE` pointing at Relay's JSON on the sidecar Community Engager uses |

## Non-goals

| Out | Why |
|-----|-----|
| Migrating SQLite `namespace` column name | Noise; document mapping instead |
| Dropping `/v1/users` in the same PR as alias | Breaks every current client overnight |
| Domains as real SQL tables | Wrong model |
| Forcing every write to supply domain | Classifier + profile remain the default path |

## Success criteria

- [x] Glossary published; UI says Tenant / Domain (kind)
- [ ] `/v1/tenants/{id}/memories` works; `/v1/users/{id}/…` still works
- [ ] Client accepts `tenantId`; old `userId` still works
- [x] POST memory with `domain` persists that domain
- [x] `VOLTMEM_DOMAINS_FILE` merges app domains and keyword hints; tests cover load + one write/search
- [ ] SIDECAR documents profile + tenant rename + domain-on-write

## Suggested order

1. `naming-docs` + `ui-tenant-copy` (cheap, unblocks mental model)
2. `domain-on-add` + tests (unblocks correct filters even on stylens)
3. `relay-profile` — shipped as `VOLTMEM_DOMAINS_FILE`, not a compiled relay profile
4. `api-tenants-alias` + `client-tenantId` (compat-sensitive; ship together)
5. `dogfood-handoff` checklist for relay-os

## Versioning

- Sidecar / library: **minor** (0.5.x or next minor after current) — additive routes + optional body field + new profile.
- `@voltmem/client`: **minor** with deprecation, not a hard break until a later major drops `userId` / `/users`.
