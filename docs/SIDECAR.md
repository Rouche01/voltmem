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

## Option A — Pull a published image (recommended)

When releases publish to GitHub Container Registry:

```bash
docker pull ghcr.io/rouche01/voltmem-sidecar:latest

docker run -d --name voltmem \
  -p 8080:8080 \
  -e VOLTMEM_API_KEY="$(openssl rand -hex 32)" \
  -v voltmem-data:/data \
  ghcr.io/rouche01/voltmem-sidecar:latest
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

docker build -t voltmem-sidecar .
docker run -d --name voltmem \
  -p 8080:8080 \
  -e VOLTMEM_API_KEY=replace-me \
  -v voltmem-data:/data \
  voltmem-sidecar
```

No install of Python on the host is required — only Docker.

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
5. **Multi-tenant** — pass a stable `user_id` per end-user; one sidecar / one DB can serve many tenants.

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
  userId: "alice",
});

await mem.add("I prefer darker colors and minimal fits");
const hits = await mem.search("style preferences", { limit: 5 });
const stats = await mem.domainStats();
```

Cloudflare Worker secrets: `VOLTMEM_URL`, `VOLTMEM_API_KEY` — never expose the key to browsers.

---

## Smoke test (curl)

```bash
export BASE=http://127.0.0.1:8080
export KEY=replace-me

curl -s "$BASE/health"

curl -s -X POST "$BASE/v1/users/alice/memories" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"data":"I prefer darker colors and minimal fits"}'

curl -s "$BASE/v1/users/alice/memories/search?q=style%20preferences&limit=3" \
  -H "X-API-Key: $KEY"
```

Full route table: [sidecar/README.md](../sidecar/README.md).

---

## Publishing the image (maintainers)

CI workflow [`.github/workflows/publish-sidecar.yml`](../.github/workflows/publish-sidecar.yml)
builds the Dockerfile and pushes to:

`ghcr.io/rouche01/voltmem-sidecar`

Triggers: tags matching `sidecar-v*` (e.g. `sidecar-v0.1.0`) and manual
`workflow_dispatch`.

```bash
git tag sidecar-v0.1.0
git push origin sidecar-v0.1.0
```

After the first successful push, set the GHCR package visibility to **Public**
so anonymous `docker pull` works.
