# Canvases

Live Cursor canvases are stored outside this repo. Cursor only loads them from its managed project folder.

## VoltMem direction

Roadmap canvas: reframe (including sleeptime / maintenance-window), open-problem aims, competitors, synthetic stress tests, and **dogfood projects** (stylens-lite-api → relay-os → Room Scout).

- **Open:** [voltmem-direction](/Users/richardemate/.cursor/projects/Users-richardemate-Projects-voltmem/canvases/voltmem-direction.canvas.tsx)
- Or: Command Palette → **Open Canvas**

## VoltMem next up

Executable to-do board: dogfood, measurement, sleeptime hardening, open-problem backlog, and deferred items. Click a row to cycle status (persists). Use this when you want “what do I do now?” — not brainstorming.

- **Open:** [voltmem-next-up](/Users/richardemate/.cursor/projects/Users-richardemate-Projects-voltmem/canvases/voltmem-next-up.canvas.tsx)

### Dogfood integration path (shipped in-repo)

TypeScript / Cloudflare Workers apps talk to VoltMem over HTTP — do not port the engine:

| Piece | Location |
|---|---|
| FastAPI sidecar | [sidecar/](../../sidecar/) · [sidecar/README.md](../../sidecar/README.md) |
| `@voltmem/client` | [clients/typescript/](../../clients/typescript/) |
| Docker image | [Dockerfile](../../Dockerfile) |

Flow: **Worker → `@voltmem/client` → sidecar → `create_memory`** (`add` / `search` / `domain_stats` / events / maintenance). First dogfood target: **stylens-lite-api**.

Related: [OPEN_PROBLEMS.md](../OPEN_PROBLEMS.md) · [SCHEDULE.md](../SCHEDULE.md) (includes sleeptime is/isn't) · [SLEEPTIME_COMPUTE_v3.md](../SLEEPTIME_COMPUTE_v3.md) · [RESEARCH.md](../RESEARCH.md) · [SIDECAR.md](../SIDECAR.md)
