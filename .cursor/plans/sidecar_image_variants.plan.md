---
name: Sidecar image variants
overview: "Publish two sidecar images from one Dockerfile: the existing embeddings image, and a slim image without PyTorch. :latest stays the full image."
todos:
  - id: dockerfile-arg
    content: "Dockerfile ARG EMBEDDINGS=1. 1 installs .[sidecar,embeddings] and sets VOLTMEM_EMBEDDINGS=1. 0 installs .[sidecar] only, skips build-essential, sets VOLTMEM_EMBEDDINGS=0. Default docker build stays the full image."
    status: completed
  - id: publish-both-tags
    content: "publish-sidecar.yml builds both on sidecar-v* and workflow_dispatch. Full tags :VERSION and :latest. Slim tags :VERSION-slim and :slim. A manual run also pushes :sha-<short> and :sha-<short>-slim."
    status: completed
  - id: docs-variants
    content: "SIDECAR.md and sidecar/README.md document both pulls and the build-arg. Slim is the hashing image; full is MiniLM. No app names."
    status: completed
isProject: true
---

# Sidecar image variants

Canonical plan for the two `voltmem-sidecar` images. Open while the `voltmem` workspace is active.

**Related:** tenant and domains work [`.cursor/plans/tenant_domains_semantics.plan.md`](./tenant_domains_semantics.plan.md). The slim tag is for hosts that search with shared tokens. The full tag stays for MiniLM paraphrase search.

## Problem

The root `Dockerfile` always installs `.[sidecar,embeddings]`. That pulls `sentence-transformers`, PyTorch, and SymPy. Exporting that layer failed on this Mac because the disk was full (`input/output error` while writing a `sympy` file).

A caller that writes an explicit `domain` and searches with tokens that already appear in the stored text does not need that model. The hashing scorer returns those hits. `min_score` defaults to `0`.

## Tags

One Dockerfile. `ARG EMBEDDINGS=1` so a bare `docker build` stays the image operators already expect.

| Image | Build | Pip extra | `VOLTMEM_EMBEDDINGS` | Tags on `sidecar-v0.6.0` |
|---|---|---|---|---|
| Full | `EMBEDDINGS=1` (default) | `sidecar,embeddings` | `1` | `:0.6.0`, `:latest` |
| Slim | `EMBEDDINGS=0` | `sidecar` | `0` | `:0.6.0-slim`, `:slim` |

`:latest` is the full image. The slim tag is explicit.

Local slim build, including on a full disk:

```bash
docker build --build-arg EMBEDDINGS=0 -t voltmem-sidecar:0.6.0-slim .
```

## Publish

[`.github/workflows/publish-sidecar.yml`](../../.github/workflows/publish-sidecar.yml) already pushes on `sidecar-v*` and `workflow_dispatch`. Extend that job to build both variants. GitHub Actions has the disk for the full image, so the release does not depend on this Mac.

```bash
git tag sidecar-v0.6.0
git push origin sidecar-v0.6.0
```

Result:

- `ghcr.io/rouche01/voltmem-sidecar:0.6.0`
- `ghcr.io/rouche01/voltmem-sidecar:latest`
- `ghcr.io/rouche01/voltmem-sidecar:0.6.0-slim`
- `ghcr.io/rouche01/voltmem-sidecar:slim`

## Docs

- [docs/SIDECAR.md](../../docs/SIDECAR.md) Option A shows both pulls. Option B shows the build-arg.
- [sidecar/README.md](../../sidecar/README.md) Docker section matches.
- Say which tag to use: `:slim` when queries share tokens with stored text. The full tag when search must match facts that share no tokens with the query. Do not name a consumer app.

## Out of scope

| Out | Why |
|---|---|
| PyPI / `pyproject.toml` version bump | The image installs this repo. Dogfood does not `pip install voltmem`. |
| `@voltmem/client` publish | Separate tag `client-v0.5.0`. |
| Changing the hashing scorer | Slim is a packaging choice. Search behavior for `VOLTMEM_EMBEDDINGS=0` already exists. |
| A second Dockerfile | The variants would drift. |

## Success criteria

- [ ] `docker build` with no args still installs `sentence-transformers` and leaves `VOLTMEM_EMBEDDINGS=1`
- [ ] `docker build --build-arg EMBEDDINGS=0` does not install `sentence-transformers` or PyTorch, and sets `VOLTMEM_EMBEDDINGS=0`
- [ ] `sidecar-v*` pushes `:VERSION`, `:latest`, `:VERSION-slim`, and `:slim`
- [x] Docs name `:slim` as the hashing image and `:latest` as MiniLM, with no consumer app names
