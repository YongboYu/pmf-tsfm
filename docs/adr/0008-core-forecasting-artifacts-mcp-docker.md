# The core forecasting capability ships as dedicated MCP + Docker artifacts, separate from the demo Space

**Status:** accepted (2026-06-08)

The core `src/pmf_tsfm/` pipeline (multi-model zero-shot DF-relation forecasting + MAE/RMSE +
Entropic Relevance) is delivered to developers and agents as **two dedicated artifacts built on a
shared Gradio-free seam** — a headless **FastMCP server** (`mcp/`) and a self-host **Docker image**
(`docker/`) — so others can run it on **their own** logs (a raw `.xes`, auto-converted, or a prepared
DF-relation `.parquet`), with no caps and numbers that match the paper/CLI. The seam is
`src/pmf_tsfm/api.py` (`forecast_backtest` / `forecast_only` / `list_models`): agent-clean, typed,
Gradio-free, composing the existing Hydra configs and calling the real cores (`run_inference`,
`evaluate_single`, `run_er_evaluation`) — it does **not** reimplement forecasting/metrics/ER. As a
consequence the demo (`demo/`) reverts to a **GUI-only** HF Space (#136).

## Context

ADR-0003 ("one Gradio app, three surfaces") was implemented for the **demo**: the hosted Space, its
auto-generated REST/MCP via `gr.api(...)` + `launch(mcp_server=True)` (#116), and a GUI-only Docker
image (#129/#130). But the demo path is deliberately `pmf_tsfm`-free and Chronos-2-only, and serves
GUI-shaped **drift visualizations**, never the core pipeline. So those surfaces could not deliver the
real capability — multi-model forecasting + genuine accuracy on arbitrary user data. We want that
capability usable by humans (Docker CLI), scripts (REST/CLI), and agents (MCP), on their own data.

## Considered options

1. **Dedicated `mcp/` (FastMCP) + `docker/` over a shared `pmf_tsfm.api` seam** *(chosen)* — two
   small folders wrapping one tested core seam; the Docker image also runs the Hydra CLIs and bundles
   the MCP server. Clean separation from the demo; numbers match the CLI/paper because the seam reuses
   the real cores.
2. **Extend the demo's Gradio `mcp_server=True` to call the core** — rejected: it couples the core
   service to Gradio and keeps `demo/` from being GUI-only; a headless service has no need for a web
   framework.
3. **One combined artifact** — rejected: the ADR-0003 amendment keeps the hosted Space a Gradio-SDK
   Space (for ZeroGPU), and the user wants the dev-ready core artifacts in their own dedicated folders,
   distinct from the visualization demo.

## Amendment to ADR-0003

ADR-0003 rejected a standalone FastMCP server "once Gradio provides built-in MCP." That rejection
**still holds for the demo Space** — a GUI-first app where `mcp_server=True` yields MCP for free
alongside the GUI it already needs. It is **amended for the headless core service**: a no-GUI service
pulling in all of Gradio just to expose a few functions reintroduces exactly the coupling the
demo/core split removes. For the core, a **dedicated FastMCP server** is correct. ADR-0003's "one
codebase, every surface" intent stands; the surfaces are simply realized by separate packagings of the
same core, not by one Gradio app.

## Consequences

- `src/pmf_tsfm/api.py` is the single, unit-tested seam both artifacts depend on (Phase 0, #132).
- The XES→DF-series bridge is lifted into the core (`src/pmf_tsfm/data/log_to_series.py`) so `.xes`
  input works; a deliberate end-trim deviation from upstream is documented there (faithful, not
  byte-exact, reproduction — fidelity to the bundled parquets is not a correctness gate).
- **Scope:** zero-shot **holdout backtest** only (ADR-0004) — no fine-tuning in these artifacts.
  All three families are supported; TimesFM is opt-in in the Docker image (`--build-arg
  INSTALL_TIMESFM=1`).
- **Parquet-only input cannot produce Entropic Relevance** (ER needs the XES log to build
  truth/training DFGs) → `metrics.er = null`; documented, not a bug.
- **Docker must build with `uv`, not pip** — the `[tool.uv] override-dependencies` resolution of the
  `uni2ts`↔`chronos` numpy/torch clash only holds under uv.
- The demo (`demo/`) becomes GUI-only: the `gr.api`/`mcp_server`/`forecast_from_source` wiring and
  `demo/Dockerfile` are removed (#136). ADR-0003 continues to govern the demo Space itself.
- A future reader who finds both a demo MCP/Docker history (#116/#129) and the core `mcp/`+`docker/`
  artifacts should read this ADR: the demo ones packaged the visualization; the core ones package the
  real pipeline, and the demo ones were retired.
