# pmf-tsfm MCP server

A dedicated **headless [FastMCP](https://modelcontextprotocol.io) server** that exposes the core
forecast+evaluate capability of `pmf-tsfm` as MCP tools for agents and scripts — **no GUI**. It is
a thin plumbing layer over [`pmf_tsfm.api`](../src/pmf_tsfm/api.py): each tool wraps one
agent-clean entry point that composes the existing Hydra configs and calls the real cores, so the
numbers match the paper/CLI. FastMCP auto-derives every tool's JSON schema from its type hints +
docstring — there is **no hand-written schema**.

This is *not* the Gradio-headless MCP surface in `demo/` (`gr.api(...)` + `launch(mcp_server=True)`,
which serves GUI-shaped drift bundles); it is a no-GUI service for agents and scripts.

## Quick start

This server is **repo-local** — not an `npx`/`uvx` registry drop-in. It wraps the `pmf_tsfm`
package (the model cores + Hydra configs, run against *your* data), so there is no published
one-liner: clone the repo, install the `mcp` extra, then point your agent at it.

```bash
git clone https://github.com/YongboYu/pmf-tsfm.git && cd pmf-tsfm
uv sync --extra mcp
# wire it into Claude Code (writes .mcp.json in the repo):
claude mcp add pmf-tsfm --scope project \
  -- uv run --directory "$PWD" --extra mcp python mcp/server.py
```

No-clone option: the [self-host Docker image](#run-via-the-self-host-docker-image) bundles the
server (`docker run -i <image> mcp`). For Claude Desktop / custom-client configs and example
prompts, see [Connect an agent / MCP client](#connect-an-agent--mcp-client).

## Tools

| Tool | Wraps | Returns |
| --- | --- | --- |
| `list_models` | `api.list_models()` | list of model config groups, e.g. `"chronos/chronos2"` |
| `forecast_backtest` | `api.forecast_backtest(...)` | `{predictions_path, quantiles_path, metrics, feature_names, n_windows, model, horizon}` where `metrics = {mae, mae_std, rmse, rmse_std, er}` |
| `forecast_only` | `api.forecast_only(...)` | same bundle with `metrics = null` (forecast, no scoring) |

`forecast_backtest` / `forecast_only` accept a raw `.xes`/`.xes.gz` log (auto-converted to the
daily DF-relation series) or a prepared DF-relation `.parquet`. Scope is the locked zero-shot
holdout backtest: the natural 60/20/20 split holds out the tail and the existing
expanding-window pipeline scores it (MAE/RMSE, plus Entropic Relevance when an XES log is given).

## Run locally

```bash
uv sync --extra mcp          # core deps + the `mcp` SDK
python mcp/server.py         # stdio transport
```

Then connect with any MCP client — the `mcp` Python SDK, or the
[MCP Inspector](https://github.com/modelcontextprotocol/inspector)
(`npx @modelcontextprotocol/inspector python mcp/server.py`). Example client round-trip:

```python
import asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

async def main():
    params = StdioServerParameters(command="python", args=["mcp/server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print((await session.list_tools()).tools)
            out = await session.call_tool(
                "forecast_backtest", {"input_path": "data/processed_logs/sepsis.xes"}
            )
            print(out.content[0].text)   # JSON bundle: metrics has finite MAE/RMSE + ER

asyncio.run(main())
```

## Connect an agent / MCP client

The point of this server is to let an **agent** call forecasting as a tool. Any MCP host works —
[Claude Code](https://docs.claude.com/en/docs/claude-code), [Codex CLI](https://github.com/openai/codex),
Claude Desktop, Cursor, and others, or a custom client over the `mcp` SDK. You register the launch
command once; the three tools above then appear to the model with schemas auto-derived from their
signatures.

Pick a launch command that (a) runs in an environment where the `mcp` extra is installed and (b) has
the **repo root as its working directory** — so `python mcp/server.py` keeps `sys.path[0]` at `mcp/`
(see [Note on the folder name](#note-on-the-folder-name)) and relative `input_path`s resolve against
your data. `uv run --directory` does both:

**Claude Code** — add it as a project-scoped server (writes `.mcp.json`):

```bash
claude mcp add pmf-tsfm --scope project \
  -- uv run --directory "$PWD" --extra mcp python mcp/server.py
```

or write `.mcp.json` at the repo root by hand:

```json
{
  "mcpServers": {
    "pmf-tsfm": {
      "command": "uv",
      "args": ["run", "--directory", "/abs/path/to/pmf-tsfm", "--extra", "mcp", "python", "mcp/server.py"]
    }
  }
}
```

**Claude Desktop / Cursor / Cline / Windsurf** — the same `mcpServers` block in that host's config
(`claude_desktop_config.json`, `.cursor/mcp.json`, …). Use the absolute `--directory` path when the
host has no project working directory.

**Codex CLI** — the same command/args, expressed as TOML in `~/.codex/config.toml`:

```toml
[mcp_servers.pmf-tsfm]
command = "uv"
args = ["run", "--directory", "/abs/path/to/pmf-tsfm", "--extra", "mcp", "python", "mcp/server.py"]
```

Then just ask in natural language — the agent chooses the tools and fills the arguments:

> "List the pmf-tsfm models, then backtest `data/processed_logs/sepsis.xes` with `chronos/chronos2`
> and report the MAE and Entropic Relevance."

A typical call sequence and what comes back:

1. `list_models()` → `["chronos/chronos2", "moirai/2_0_small", ...]`
2. `forecast_backtest(input_path="data/processed_logs/sepsis.xes", model="chronos/chronos2")`
   → `{"metrics": {"mae": ..., "rmse": ..., "er": ...}, "predictions_path": ..., "n_windows": ..., ...}`

Tips for agent use:

- Pass **absolute `input_path`s** (or paths relative to the `--directory` root) — the server resolves
  them against its own working directory, not the agent's.
- Entropic Relevance (`er`) is only computed for `.xes`/`.xes.gz` input; for a `.parquet` series it
  comes back `null` (set `compute_er=false` to skip it on XES too).
- Use `forecast_only` when you want predictions without scoring — e.g. forecasting a genuine future
  that has no held-out ground truth — and `device="cuda"`/`"mps"` to move off the CPU default.

## Run via the self-host Docker image

The self-host image (#134) bundles this server; the `mcp` subcommand launches it over stdio:

```bash
docker run -i <image> mcp
```

(The image installs the package with the `mcp` extra — `pip install .[mcp]` — and the entrypoint's
`mcp` dispatch is owned by the Docker track.)

## Note on the folder name

This folder is literally `mcp/`, which could shadow the installed `mcp` SDK package. It is kept a
**plain script directory (no `__init__.py`)** and launched as `python mcp/server.py`, so
`sys.path[0]` is the `mcp/` directory (which has no nested `mcp/`) and `from mcp.server.fastmcp
import FastMCP` still resolves to the site-packages SDK. Tests add the `mcp/` directory (not the
repo root) to pytest `pythonpath` for the same reason — never put the repo root on `sys.path`.

## Tests

```bash
uv run pytest mcp/tests -m "not slow"     # fast, CI-safe: schema + plumbing, no model load
uv run pytest mcp/tests -m slow           # real backtest on the bundled sepsis XES (if present)
```

The slow acceptance test runs a real `forecast_backtest` on `data/processed_logs/sepsis.xes`
through an in-process MCP client and asserts finite MAE/RMSE + an ER block. That XES is untracked,
so the test auto-skips wherever it is absent (e.g. CI).
