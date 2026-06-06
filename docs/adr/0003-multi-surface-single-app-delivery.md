# One Gradio app delivers GUI + REST + MCP; one codebase, two packagings

> **Amended (PRD #112, HF-deployment grilling).** The original decision called the hosted Space, the
> talk-recording build, and the self-host image the *identical* Docker artifact. That is softened to
> **"one codebase, two packagings"** — see the Amendment section below. The "one Gradio app, three
> surfaces (GUI + REST + MCP)" core stands unchanged.

We want the work usable by humans, scripts, *and* modern agents without building and maintaining
separate services. A Gradio app launched with `mcp_server=True` simultaneously exposes a **GUI**, an
auto-generated **REST API** (`gradio_client`), and an **MCP server** (each function → an MCP tool,
schema derived from its type hints + docstring) — and on HF Spaces the MCP endpoint is hosted for
free. We package that same app as **one Docker image** whose default entrypoint is the GUI but which
also runs the Hydra CLI, so the *identical* artifact is our talk-recording build, our Space deploy,
and a tool others self-host on their own logs/hardware (no caps). A Claude **Skill** and a
`uvx`/PyPI **CLI** wrap the same code for agent and terminal use.

## Considered Options
- **Standalone FastMCP server** — rejected as redundant once Gradio provides built-in MCP.
- **Separate GUI / API / repro artifacts** — rejected: triple the maintenance for one codebase.

## Consequences
- Forecast functions in `demo/forecast.py` must be written **agent-clean** (clear type hints +
  docstrings) because those *are* the MCP/REST tool schemas.
- The agent-facing forecast tool takes a **bundled dataset name or a log URL**, not a browser file
  picker — MCP passes files as URL/base64, so the GUI upload widget has no agent equivalent.
- **Refinement (slice design):** the agent-facing tool exposes only the *live forecasting mechanism*
  (a log → forecast DFG); it **never serves precomputed metrics**. Precomputed bundled results +
  accuracy metrics (ER/MAE/RMSE) are a **GUI-only** reproducibility concern. `mcp_server=True` is
  flipped on only *after* the GUI tracer-bullet slice — building the GUI does **not** block on the
  MCP contract, as long as the forecast functions are written agent-clean from the start.

## Amendment — one codebase, two packagings (PRD #112)

The "*identical* Docker artifact for Space + talk + self-host" turned out to over-constrain the
hosted path. The **HF Space is a Gradio-SDK Space** (`app.py` + `requirements.txt` at the Space root,
serve-time deps gradio-only per ADR-0005) — *not* a hand-built Docker image. That is the simplest
gradio-only host **and** the native home for ZeroGPU (ADR-0001), which the live-upload slice needs;
ZeroGPU is far more fragile on Docker-SDK Spaces.

So the same `demo/` codebase ships as **two packagings**, not one:

1. the **Gradio-SDK HF Space** — the hosted runtime (GUI now; REST/MCP via `mcp_server=True`; ZeroGPU
   for the live path);
2. a **self-host Docker image** (GUI default entrypoint + Hydra CLI) — built from the same code for
   talk-recording and others self-hosting with no caps. **Future work**, not built in PRD #112.

"One codebase, every surface" holds; "one *identical artifact*" does not. The CLI / PyPI / Claude
Skill remain the recorded destination and remain future work.
