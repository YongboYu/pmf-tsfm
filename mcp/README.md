# pmf-tsfm MCP server

A dedicated **headless [FastMCP](https://modelcontextprotocol.io) server** that exposes the core
forecast+evaluate capability of `pmf-tsfm` as MCP tools for agents and scripts — **no GUI**. It is
a thin plumbing layer over [`pmf_tsfm.api`](../src/pmf_tsfm/api.py): each tool wraps one
agent-clean entry point that composes the existing Hydra configs and calls the real cores, so the
numbers match the paper/CLI. FastMCP auto-derives every tool's JSON schema from its type hints +
docstring — there is **no hand-written schema**.

This is *not* the Gradio-headless MCP surface in `demo/` (`gr.api(...)` + `launch(mcp_server=True)`,
which serves GUI-shaped drift bundles). See **Why a dedicated server** below.

## Tools

| Tool | Wraps | Returns |
| --- | --- | --- |
| `list_models` | `api.list_models()` | list of model config groups, e.g. `"chronos/chronos2"` |
| `forecast_backtest` | `api.forecast_backtest(...)` | `{predictions_path, quantiles_path, metrics, feature_names, n_windows, model, horizon}` where `metrics = {mae, mae_std, rmse, rmse_std, er}` |
| `forecast_only` | `api.forecast_only(...)` | same bundle with `metrics = null` (forecast, no scoring) |

`forecast_backtest` / `forecast_only` accept a raw `.xes`/`.xes.gz` log (auto-converted to the
daily DF-relation series) or a prepared DF-relation `.parquet`. Scope is the locked zero-shot
holdout backtest (ADR-0004): the natural 60/20/20 split holds out the tail and the existing
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

## Run via the self-host Docker image

The self-host image (#134) bundles this server; the `mcp` subcommand launches it over stdio:

```bash
docker run -i <image> mcp
```

(The image installs the package with the `mcp` extra — `pip install .[mcp]` — and the entrypoint's
`mcp` dispatch is owned by the Docker track.)

## Why a dedicated server (not Gradio-headless)

This is a no-GUI service. Pulling all of Gradio in to expose a few functions reintroduces the
coupling the demo/core split removes and contradicts keeping `demo/` GUI-only. ADR-0003 rejected a
standalone FastMCP server *only in the GUI-Space context*; the amendment that carves out this
headless core server is **ADR-0008** (written under #135) — see `docs/adr/`.

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
