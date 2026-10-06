# Deploy the VoltMem HTTP sidecar

Anyone integrating VoltMem from TypeScript, Cloudflare Workers, or another
non-Python runtime needs a **reachable sidecar** process. The TypeScript
package `@voltmem/client` only speaks HTTP — it does not embed the Python
engine.

This guide covers how to get that process running and how to point a client at it.

**Related:** [sidecar/README.md](../sidecar/README.md) (API reference) ·
[clients/typescript/README.md](../clients/typescript/README.md) ·
[Dockerfile](../Dockerfile)

---

## Architecture

```text
Your app (Worker / Node / etc.)
        │  @voltmem/client  or  fetch
        ▼
VoltMem sidecar  (Docker or uvicorn)
        │
        ▼
SQLite + embeddings  (persistent volume)
```

You choose who runs the sidecar:

| Model | Who runs it | Typical use |
|---|---|---|
| Self-host | Each team deploys their own container | Product apps (e.g. stylens) |
| Shared service | You host one multi-tenant URL | SaaS / demos |
| Local | Developer laptop | Integration tests |

---

## Concepts

| Concept | Role | Example |
|---|---|---|
| **Tenant** | Isolation boundary: one logical database inside the shared SQLite file. A person or an app bucket. | `alice`, `relay-local` |
| **Domain** | Fact kind plus a volatility prior. A partition inside a tenant. | `style_preference` |
| **Memory item** | One stored fact. | “I prefer darker colors and minimal fits” |
| **Profile** | Built-in registry and classifier loaded when the sidecar process starts (`VOLTMEM_PROFILE`). | `stylens` |

The tenant id is what you pass as Python `user_id` and as the HTTP path segment `/v1/tenants/{tenant_id}`. SQLite stores that id in the `namespace` column. Call it a **tenant** in docs and in the memory browser. `/v1/users/{tenant_id}/…` is the same handlers, marked deprecated in OpenAPI, so older clients keep working.

A domain is a column on the memory row, with its volatility prior coming from the profile. The profile classifier assigns the kind on write unless the caller sets `domain`. Python `remember(..., domain=…)` and `POST /v1/tenants/{tenant_id}/memories` with `{ "domain": "…" }` both skip classification. `POST .../events` already takes `domain` on each facet. The memory browser filter shows that stored kind.

Tenant is the database. Domain is a typed partition inside it. Both live in the same SQLite table.

### Extra domains

`VOLTMEM_PROFILE` still picks the built-in registry. The default is `stylens`. App-specific kinds do not belong in that code. Point `VOLTMEM_DOMAINS_FILE` at a JSON file and the sidecar merges it in before serving traffic:

```json
{
  "domains": [
    {"name": "community_preference", "volatility": 0.20},
    {"name": "community_outcome", "volatility": 0.55}
  ],
  "keywords": {
    "community_outcome": ["[outcome]", "aborted"],
    "community_preference": ["[preference]", "allowlist"]
  }
}
```

`domains` are registered on top of the profile (`slot` is optional). `keywords` are optional and are checked before the profile classifier; earlier keys win. A write can still set `domain` and skip classification. One process still has one registry, so two apps that need different priors run two sidecars, each with its own file.

---

## Option A — Pull a published image (recommended)

A `sidecar-v*` tag publishes two images. `:latest` includes sentence-transformers (MiniLM). `:slim` uses the hashing scorer. Hashing is enough when the query shares tokens with the stored text. The score floor is `0`, so those overlapping tokens still rank. Use the full image when a search must match a fact that shares no tokens with the query.

```bash
# Hashing scorer
docker pull ghcr.io/rouche01/voltmem-sidecar:slim

docker run -d --name voltmem \
  -p 8080:8080 \
  -e VOLTMEM_API_KEY="$(openssl rand -hex 32)" \
  -v voltmem-data:/data \
  ghcr.io/rouche01/voltmem-sidecar:slim

# Full image (MiniLM). Pin a release with :0.6.0 or :0.6.0-slim.
# docker pull ghcr.io/rouche01/voltmem-sidecar:latest
```

Verify:

```bash
curl -s http://127.0.0.1:8080/health
# {"status":"ok"}
```

Save the API key; clients must send it as `X-API-Key`.

> If the package is private on first publish, make it public under the repo
> **Packages** settings, or authenticate: `echo $GITHUB_TOKEN | docker login ghcr.io -u USER --password-stdin`.

---

## Option B — Build from the public Dockerfile

The Dockerfile lives at the root of
[github.com/Rouche01/voltmem](https://github.com/Rouche01/voltmem):

```bash
git clone https://github.com/Rouche01/voltmem.git
cd voltmem

# Full image (default): sentence-transformers, VOLTMEM_EMBEDDINGS=1
docker build -t voltmem-sidecar .

# Slim image: hashing scorer, VOLTMEM_EMBEDDINGS=0
docker build --build-arg EMBEDDINGS=0 -t voltmem-sidecar:slim .

docker run -d --name voltmem \
  -p 8080:8080 \
  -e VOLTMEM_API_KEY=replace-me \
  -v voltmem-data:/data \
  voltmem-sidecar:slim
```

No install of Python on the host is required — only Docker. `EMBEDDINGS` must be `0` or `1`. Omitting it builds the full image.

---

## Option C — Run without Docker

```bash
pip install "voltmem[sidecar,embeddings]"   # or: pip install -e ".[sidecar,embeddings]" from a clone
export VOLTMEM_API_KEY=replace-me
export VOLTMEM_DB_PATH=./voltmem_sidecar.db
export VOLTMEM_EMBEDDINGS=1
python -m sidecar
# listens on 0.0.0.0:8080 by default
```

---

## Production checklist

1. **Persist `/data`** — map a volume so SQLite survives restarts (`VOLTMEM_DB_PATH=/data/voltmem.db` in the image).
2. **Set `VOLTMEM_API_KEY`** — required in production; `/health` stays open, all `/v1/*` routes require `X-API-Key`.
3. **TLS + public URL** — put the container behind Fly.io, Railway, Render, Cloud Run, or your reverse proxy; Workers cannot call `localhost` in production.
4. **Embeddings** — image builds with `.[sidecar,embeddings]`; first start can take a minute while models load.
5. **Multi-tenant** — pass a stable tenant id per person or app bucket (`alice`, `relay-local`); one sidecar / one DB can serve many tenants. The HTTP segment is `{tenant_id}` on `/v1/tenants/…` (`/v1/users/…` is the deprecated alias).

### Example: Fly.io

```bash
fly launch --name voltmem-sidecar --region ams --no-deploy
fly volumes create voltmem_data --size 3 --region ams
fly secrets set VOLTMEM_API_KEY="$(openssl rand -hex 32)"
fly deploy   # uses the repo Dockerfile
```

Mount the volume at `/data` in `fly.toml` (`destination = "/data"`).

---

## Connect `@voltmem/client`

```bash
npm install @voltmem/client
# until published: "file:../voltmem/clients/typescript" after npm run build there
```

```ts
import { VoltMemClient } from "@voltmem/client";

const mem = new VoltMemClient({
  baseUrl: process.env.VOLTMEM_URL!,      // https://voltmem.example.com
  apiKey: process.env.VOLTMEM_API_KEY!,
  tenantId: "alice",                       // person or app bucket; userId still accepted
});

await mem.add("I prefer darker colors and minimal fits");
const hits = await mem.search("style preferences", { limit: 5 });
const stats = await mem.domainStats();
```

Cloudflare Worker secrets: `VOLTMEM_URL`, `VOLTMEM_API_KEY` — never expose the key to browsers.

---

## Memory browser

Operators can open `http://127.0.0.1:8080/ui` on a running sidecar and paste `VOLTMEM_API_KEY` plus a tenant id such as `relay-local`. The page lists active memories and draws replacement and shared-event edges from `GET /v1/tenants/{tenant_id}/graph`. **Clear tenant** wipes that tenant’s memories via `DELETE .../memories` after a typed confirm; **Clear domain** (with a domain filter set) clears one kind via `?domain=`. `/ui` itself is unauthenticated; memory calls still send `X-API-Key`. Bind with `HOST=127.0.0.1` when the UI should stay on the machine. Details: [sidecar/README.md](../sidecar/README.md#memory-browser).

---

## Smoke test (curl)

```bash
export BASE=http://127.0.0.1:8080
export KEY=replace-me

curl -s "$BASE/health"

curl -s -X POST "$BASE/v1/tenants/alice/memories" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"data":"I prefer darker colors and minimal fits"}'

curl -s "$BASE/v1/tenants/alice/memories/search?q=style%20preferences&limit=3" \
  -H "X-API-Key: $KEY"

curl -s "$BASE/v1/tenants/relay-local/graph" \
  -H "X-API-Key: $KEY" | jq '.nodes | length'

# browser: open "$BASE/ui" and paste KEY + tenant id (relay-local, alice, …)
```

Full route table: [sidecar/README.md](../sidecar/README.md).

---

## Publishing the image (maintainers)

CI workflow [`.github/workflows/publish-sidecar.yml`](../.github/workflows/publish-sidecar.yml)
builds both Dockerfile variants and pushes to `ghcr.io/rouche01/voltmem-sidecar`.

Triggers: tags matching `sidecar-v*` (e.g. `sidecar-v0.6.0`) and manual
`workflow_dispatch`. A version tag pushes `:VERSION` and `:latest` (embeddings)
plus `:VERSION-slim` and `:slim` (hashing). A manual run pushes `:sha-<short>`
and `:sha-<short>-slim`.

```bash
git tag sidecar-v0.6.0
git push origin sidecar-v0.6.0
```

After the first successful push, set the GHCR package visibility to **Public**
so anonymous `docker pull` works.
