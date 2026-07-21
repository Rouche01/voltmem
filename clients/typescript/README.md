# `@voltmem/client`

Workers-safe TypeScript client for the [VoltMem HTTP sidecar](../../sidecar/README.md).
Zero runtime dependencies — uses global `fetch`.

## Install

From this repo (path / file dependency):

```bash
# in your TS app
npm install ../path/to/voltmem/clients/typescript
# or after build:
# "dependencies": { "@voltmem/client": "file:../voltmem/clients/typescript" }
```

```bash
cd clients/typescript && npm install && npm run build
```

Not published to npm yet — use a local `file:` / workspace path.

## Usage

```ts
import { VoltMemClient } from "@voltmem/client";

const mem = new VoltMemClient({
  baseUrl: env.VOLTMEM_URL,   // e.g. https://voltmem.example.com
  apiKey: env.VOLTMEM_API_KEY,
  userId: user.id,
});

await mem.add("I prefer darker colors and minimal fits");
const hits = await mem.search("what colors does this user like?", { limit: 5 });
const stats = await mem.domainStats(); // prior calibration telemetry
```

### Cloudflare Worker

```ts
import { VoltMemClient } from "@voltmem/client";

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const mem = new VoltMemClient({
      baseUrl: env.VOLTMEM_URL,
      apiKey: env.VOLTMEM_API_KEY,
      userId: "alice",
    });
    const hits = await mem.search("style preferences");
    return Response.json(hits);
  },
};
```

Keep `VOLTMEM_API_KEY` in Worker secrets — never ship it to the browser.

### Multi-tenant

```ts
const bob = mem.forUser("bob");
await bob.add("I prefer neon colors");
```

## API

| Method | Sidecar |
|---|---|
| `health()` | `GET /health` |
| `add(data, opts?)` | `POST /v1/users/{userId}/memories` |
| `search(q, opts?)` | `GET .../memories/search` |
| `getAll()` | `GET .../memories` |
| `get(id)` | `GET .../memories/{id}` |
| `delete(id)` | `DELETE .../memories/{id}` |
| `clear()` | `DELETE .../memories` |
| `summary()` | `GET .../summary` |
| `domainStats()` | `GET .../domain_stats` |

Errors throw `VoltMemError` with `.status` and `.body`.

## Develop

```bash
npm install
npm test    # tsc + node:test against mocked fetch
```
