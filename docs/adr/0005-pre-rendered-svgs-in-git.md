# Pre-rendered DFG SVGs are committed, not rendered on read

The demo serves each DFG as an SVG. There were two ways to get one: render it **on read** from the
committed DFG JSON via graphviz (`dot`) at request time, or **pre-render** at precompute time and
commit the SVG beside the JSON. The S1 tracer bullet (PR #71) and PRD #65's wording chose render-on-
read ("Rendered on read from committed DFG JSON — no SVG blobs in git"). The full-matrix slice (PR
#81) switched to **pre-rendered, committed SVGs**: `precompute_one` writes
`forecast.svg`/`actual.svg`/`diff.svg` next to the JSON, and the served path reads those.

This is the right call and it realigns with **ADR-0002**, which always specified that the precompute
job must "produce and commit/bundle the forecast DFG, comparison DFG, **rendered images**, and
time-series data". The interim "render on read / no SVG blobs" phrasing in PR #71 / PRD #65 was a
transient S1 implementation detail, now superseded. The decisive reason is the **HF Spaces deploy**:
committing the SVGs means the served runtime needs **no graphviz `dot` binary** — serve-time deps
stay **gradio-only**, so the Space works offline and is trivial to build. The DFG **JSON remains the
regenerable source of truth**; the SVGs are a derived, committed artifact.

## Consequences
- The 12 bundled combinations commit 3 SVGs each (36 blobs) under `demo/assets/`. Repo size grows
  modestly; in exchange the serve path drops the `dot` system dependency entirely.
- Regenerating assets (adding a model/dataset, or a deliberate re-render) **does** need graphviz at
  precompute time — a dev/HPC concern, not a serve-time one. See also #91 (precompute also needs the
  processed XES logs for `truth_er`).
- The renderer stays swappable: because the JSON is canonical, a future renderer (e.g. Cytoscape)
  can replace the committed SVGs without touching the asset contract.
- This supersedes the "no SVG blobs in git" wording in PRD #65; that PRD is closed, so the decision
  lives here.
