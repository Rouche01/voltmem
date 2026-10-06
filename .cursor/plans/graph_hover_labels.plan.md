---
name: Graph hover labels
overview: "Make graph mode readable: circle-sized hit targets, short id badges on dots, full fact text in a viewport-fixed HTML tooltip on hover/focus/selection, then wheel zoom and drag pan."
todos:
  - id: hide-labels
    content: "Hide #graph .node text by default. Show it on :hover, :focus-within, and .is-selected. Keep the dot, color, and click-to-inspect behavior."
    status: completed
  - id: circle-hits-short-id
    content: "Replace the wide label hit rect with a circle-sized hit target. Always show a 2–5 letter badge (last 4 of memory id). Full fact text only on hover/focus/selection."
    status: completed
  - id: html-tooltip
    content: "Move hover fact text out of SVG into a body-level #graph-tooltip (position:fixed, high z-index). Position from the node getBoundingClientRect, clamp to the viewport, show on hover/focus/selection. Remove SVG .node-fact so text is never under other dots or clipped by the panel."
    status: completed
  - id: zoom-pan
    content: "Add wheel zoom (toward pointer) and drag pan on the graph SVG via a transform group or viewBox. Background drag pans; node click still selects. Optional Reset view control. Bounds/min/max zoom so the graph cannot disappear."
    status: completed
  - id: collision-separation
    content: "Stop stacked dots: min-distance collision in layoutGraph, wider domain-home spiral, weaker home pull, soft walls + post-clamp separation passes. Clear inspect selection when filters hide the selected node."
    status: completed
  - id: legend-stable-badges
    content: "Domain color legend + shown/domain counts; cache layout positions so selection does not reshuffle; short ids hidden until zoomed (k≥1.35) or Show ids toggle."
    status: completed
  - id: verify-graph
    content: "In the browser, open Graph for a tenant with many memories. Confirm legend, stable selection, auto/toggle ids, tooltip, zoom, pan, Reset, no occlusion."
    status: completed
isProject: false
---

# Graph hover labels + zoom/pan

Canonical plan for readable graph mode. Open while the `voltmem` workspace is active.

**Related:** memory browser [`.cursor/plans/sidecar_memory_browser.plan.md`](./sidecar_memory_browser.plan.md).

## Problem

`renderGraph` in [sidecar/static/index.html](../../sidecar/static/index.html) drew a truncated sentence under every dot (`text.slice(0, 41)`). The layout is clamped inside a 460px-tall SVG, so with dozens of memories the labels sat on top of each other. Even with labels hidden, a transparent hit rect sized for that sentence (up to 180×44) made neighboring dots hard to click. SVG text on hover was still painted under later nodes and could clip at the panel edge. Domain clusters can still leave dots too close to pick apart without zoom.

## Phase 1 — Hover labels

Done. Full fact text is no longer drawn under every node by default.

## Phase 1b — Circle hits + short id

Same file.

- Drop the wide hit `<rect>`. Use a transparent hit circle around the visible dot (visible r ≈ 8–11, hit r ≈ 14) so clicks stay on that node.
- Always show a short badge under the dot: last 4 characters of `node.id` (2–5 letters/digits). Class `node-badge`, always visible.
- `aria-label` still carries the full memory text for screen readers.
- Legend: note that the badge is the short memory id.

## Phase 1c — HTML tooltip (not SVG text)

Done. Same file.

- `#graph-tooltip` lives on `document.body` (after `</main>`), `position: fixed`, `z-index: 100`, `pointer-events: none`.
- On hover / focus / selection, `placeGraphTooltip(anchorDot, fact)` sets the text, measures size, places it above the dot (below if near the top), and clamps to the viewport with a 12px pad.
- No SVG `.node-fact` — facts never stack under other dots or clip inside the graph panel.
- Hide when leaving a non-selected node, when leaving graph mode, and at the start of each `renderGraph()`.

## Phase 2 — Zoom and pan

Done. Same file.

- Edges and nodes live in `#graph-scene` with `transform="translate(tx,ty) scale(k)"`. A fixed transparent `.graph-bg` rect (viewBox-sized, outside the scene) receives empty-space pans.
- **Wheel** zooms toward the pointer. Scale clamped to 0.5–4.
- **Drag on empty SVG space** pans. Pointer down on a `.node` does not start a pan; node click still selects.
- **Reset view** button in `#graph-toolbar` restores `k=1`, `tx=0`, `ty=0`.
- Camera resets on load, connect, refresh, filter change, resize, and opening Graph; **preserved** across selection redraws so zoom stays useful while inspecting. (Documented in a one-line comment next to the camera state.)
- After zoom/pan, an open tooltip is repositioned from the anchor node's screen rect.

## Phase 2b — Collision separation

Done. Same file (`layoutGraph`).

- Minimum node distance (~42px) with hard separation passes during and after the force ticks.
- Initial placement uses a golden-angle spiral per domain instead of a tight 16px ring.
- Weaker domain-home pull; soft edge walls so clamp does not restack piles in the corners.
- Changing text/domain/source filters clears the inspect panel when the selected memory no longer matches (avoids stale domain in the detail pane).

## Phase 2c — Legend, stable selection, badge toggle

Done. Same file.

- Legend shows `N shown · D domains` plus a color swatch list mapping each kind.
- Layout positions are cached by node/edge set + canvas size; selecting a node only updates highlight/tooltip (`applyGraphSelection`), no force re-run.
- Short id badges are hidden by default; they appear when zoomed to ≥1.35× or when **Show ids** is pressed.

Order: Phase 1 → 1b → 1c → 2 → 2b → 2c.

## Out of scope

| Out | Why |
|---|---|
| Reworking the force layout or the 460px clamp | Hover, tight hits, and camera fix readability without retuning the physics. |
| Listing domains from the domains file | The menu lists domains stored on rows. Unrelated. |
| A graph library (d3-zoom, Cytoscape) | Vanilla SVG + a few listeners match the current browser. |

## Success criteria

- [x] Graph mode shows colored dots and edges, with no sentence under each dot by default
- [x] Each node shows a short id badge (last 4 of memory id); hit target is circle-sized
- [x] Hovering / selecting / focusing a node shows the full fact in a fixed HTML tooltip above other dots and outside panel clip
- [x] Clicking a clustered dot selects that node, not a neighbor
- [x] Wheel zooms toward the pointer; drag on empty space pans; node click still selects
- [x] Reset view restores the default camera
- [x] Same-domain nodes stay separated under All kinds (no full stacks); zoom still available for dense tenants
- [x] Domain color legend and shown/domain counts
- [x] Selection does not reshuffle node positions
- [x] Short ids appear on zoom or Show ids
- [x] Browser smoke (`relay-local`, 80 nodes): legend + counts, stable selection, tooltip, Show ids / zoom badges, pan, Reset; no hard-overlapping dots
