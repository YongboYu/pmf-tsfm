# One Gradio app delivers GUI + REST + MCP; packaged as a single self-hostable GUI+CLI image

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
