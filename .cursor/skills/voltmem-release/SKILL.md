---
name: voltmem-release
description: >-
  Release VoltMem surfaces (PyPI voltmem, npm @voltmem/client, GHCR sidecar)
  selectively or all together. Use when the user asks to release, publish,
  bump versions, tag v*/client-v*/sidecar-v*, or ship PyPI/npm/sidecar.
---

# VoltMem release

Three independent version lines. Never assume “all” unless the user says so.

## 1. Confirm scope

Ask (or parse) which surfaces:

| Key | Surface | Version source | Tag | Workflow |
|-----|---------|----------------|-----|----------|
| `python` / `pypi` | `voltmem` | `pyproject.toml` → `[project].version` | `vX.Y.Z` | `publish-pypi.yml` |
| `npm` / `client` | `@voltmem/client` | `clients/typescript/package.json` | `client-vX.Y.Z` | `publish-npm.yml` |
| `sidecar` | GHCR images | **tag only** (no version file) | `sidecar-vX.Y.Z` | `publish-sidecar.yml` |

Examples: `release all`, `release npm + sidecar`, `bump python to 0.5.1 and tag`.

## 2. Pick versions

Before bumping, read current versions and last tags:

```bash
python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"
node -p "require('./clients/typescript/package.json').version"
git tag -l 'v*' | sort -V | tail -3
git tag -l 'client-v*' | sort -V | tail -3
git tag -l 'sidecar-v*' | sort -V | tail -3
```

Semver:

- **patch** — republish / fix-only / tag-before-main mistake recovery
- **minor** — new API or user-facing behavior (e.g. `clear({ domain })`)
- **major** — breaking public API

Surfaces can diverge (library `0.5.0`, client `0.6.2`, sidecar `0.7.0` is fine).

Propose versions; wait for user OK before tagging or pushing.

## 3. Hard order (every surface)

1. Bump version files + CHANGELOG when applicable  
2. Commit (only if user asked to commit)  
3. **Push `main` first**  
4. Create and push the version tag  

Never tag a commit that is not on remote `main`. Tag-before-push caused empty/wrong publishes — if that happens and the version never landed on the registry, delete/retag or bump patch; if it landed, bump patch and new tag.

## 4. Per-surface checklist

### Python (`pypi`)

- [ ] Bump `pyproject.toml` `[project].version`
- [ ] Update README badge / quickstart version only if public quickstart table needs it
- [ ] Commit + push `main`
- [ ] `git tag vX.Y.Z && git push origin vX.Y.Z`
- [ ] Tag must match `pyproject.toml` or workflow fails
- [ ] OIDC trusted publisher required (see `AGENTS.md` Releases). On auth failure: configure PyPI publisher, then re-run job / workflow_dispatch — no new tag if version never uploaded

### npm (`client`)

- [ ] Bump `clients/typescript/package.json` `version`
- [ ] Add `clients/typescript/CHANGELOG.md` entry
- [ ] Commit + push `main`
- [ ] `git tag client-vX.Y.Z && git push origin client-vX.Y.Z`
- [ ] Tag must match `package.json` or workflow fails
- [ ] Trusted publisher: workflow `publish-npm.yml`, env `release`

### Sidecar

- [ ] No version file — ensure desired code is on `main`
- [ ] Commit + push `main`
- [ ] `git tag sidecar-vX.Y.Z && git push origin sidecar-vX.Y.Z`
- [ ] Pushes `ghcr.io/rouche01/voltmem-sidecar:X.Y.Z` + `:latest` and `:X.Y.Z-slim` + `:slim`
- [ ] Auth is `GITHUB_TOKEN` (no PyPI/npm publisher)
- [ ] Optional dry run: Actions → publish-sidecar → Run workflow (sha tags only)

## 5. Release-all recipe

When user says release all (after versions agreed):

```bash
# After bumps committed and main pushed:
git tag vX.Y.Z
git tag client-vA.B.C
git tag sidecar-vP.Q.R
git push origin vX.Y.Z client-vA.B.C sidecar-vP.Q.R
```

Push tags in any order; workflows are independent. Prefer one commit that bumps python + client together if both change in the same release PR.

## 6. Agent rules

- Do **not** push tags or `git push` unless the user explicitly asks.
- Do **not** force-delete remote tags unless the user asks and the version is confirmed unpublished.
- Do **not** amend published package versions — bump instead.
- Link failed Actions runs; suggest trusted-publisher fix before inventing token auth.
- Point at `AGENTS.md` Releases for publisher field tables; keep this skill as the runbook.

## 7. Verify

```bash
# PyPI
pip index versions voltmem 2>/dev/null || curl -s https://pypi.org/pypi/voltmem/json | python -c "import sys,json; print(json.load(sys.stdin)['info']['version'])"

# npm
npm view @voltmem/client version

# GHCR (needs auth for private; public packages list via gh)
gh run list --workflow=publish-sidecar.yml --limit 3
```
