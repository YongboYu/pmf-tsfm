"""Dedicated headless FastMCP server over ``pmf_tsfm.api`` (issue #133).

This exposes the *core* forecast+evaluate capability as MCP tools — **not** Gradio-headless.
It is a thin plumbing layer: each tool wraps one of the three agent-clean entry points in
``pmf_tsfm.api`` (``list_models``, ``forecast_only``, ``forecast_backtest``), which compose the
existing Hydra configs and call the real cores, so the numbers match the paper/CLI. FastMCP
auto-derives every tool's JSON schema from its type hints + docstring — no hand-written schema.

Why FastMCP, not Gradio: this is a no-GUI service; pulling all of Gradio in to expose a few
functions reintroduces the coupling the demo/core split removes. See ADR-0003 (which rejected a
standalone FastMCP server only in the GUI-Space context) and its amendment ADR-0008 (#135).

Naming note: this folder is literally ``mcp/`` and could shadow the installed ``mcp`` SDK. It is
kept a plain script directory (no ``__init__.py``) and is launched as ``python mcp/server.py``,
so ``sys.path[0]`` is the ``mcp/`` directory (no nested ``mcp/``) and ``from mcp.server.fastmcp
import FastMCP`` still resolves to the site-packages SDK. Never put the repo root on ``sys.path``.

Run locally::

    uv sync --extra mcp
    python mcp/server.py            # stdio transport

then connect with the ``mcp`` Python SDK client or the MCP Inspector. See ``mcp/README.md``.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

import pmf_tsfm.api as api

mcp = FastMCP("pmf-tsfm")


@mcp.tool()
def list_models() -> list[str]:
    """List the available model config groups, e.g. ``"chronos/chronos2"``.

    These are exactly the values accepted by the ``model`` argument of the forecast tools
    (the Hydra config-group path under ``configs/model/``). Pure and cheap — no model import.
    """
    return api.list_models()


@mcp.tool()
def forecast_backtest(
    input_path: str,
    model: str = "chronos/chronos2",
    horizon: int = 7,
    device: str = "cpu",
    train_end: int | None = None,
    val_end: int | None = None,
    compute_er: bool = True,
) -> dict:
    """Zero-shot holdout backtest on a process log: forecast the held-out tail + score it.

    Args:
        input_path: Path to a raw ``.xes``/``.xes.gz`` log (auto-converted to the daily
                    DF-relation series) or a prepared DF-relation ``.parquet``.
        model:      Model config group, e.g. ``"chronos/chronos2"`` (see ``list_models``).
        horizon:    Forecast/holdout length in days.
        device:     ``"cpu"``, ``"cuda"``, or ``"mps"`` (falls back to CPU if unavailable).
        train_end:  Override the derived 60% train-split index.
        val_end:    Override the derived 80% val-split index.
        compute_er: Compute Entropic Relevance (only possible for ``.xes`` input).

    Returns:
        ``{predictions_path, quantiles_path, metrics, feature_names, n_windows, model,
        horizon}`` where ``metrics = {mae, mae_std, rmse, rmse_std, er}``. ``er`` is ``null``
        for ``.parquet`` input or when ``compute_er`` is false.
    """
    return api.forecast_backtest(
        input_path,
        model=model,
        horizon=horizon,
        device=device,
        train_end=train_end,
        val_end=val_end,
        compute_er=compute_er,
    )


@mcp.tool()
def forecast_only(
    input_path: str,
    model: str = "chronos/chronos2",
    horizon: int = 7,
    device: str = "cpu",
    train_end: int | None = None,
    val_end: int | None = None,
) -> dict:
    """Zero-shot forecast over the held-out tail; return predictions, skip scoring.

    Same arguments as ``forecast_backtest`` (minus ``compute_er``). ``metrics`` is ``null``.

    Args:
        input_path: Path to a raw ``.xes``/``.xes.gz`` log or a DF-relation ``.parquet``.
        model:      Model config group, e.g. ``"chronos/chronos2"`` (see ``list_models``).
        horizon:    Forecast/holdout length in days.
        device:     ``"cpu"``, ``"cuda"``, or ``"mps"`` (falls back to CPU if unavailable).
        train_end:  Override the derived 60% train-split index.
        val_end:    Override the derived 80% val-split index.

    Returns:
        ``{predictions_path, quantiles_path, metrics, feature_names, n_windows, model,
        horizon}`` with ``metrics = null``.
    """
    return api.forecast_only(
        input_path,
        model=model,
        horizon=horizon,
        device=device,
        train_end=train_end,
        val_end=val_end,
    )


def main() -> None:
    """Launch the server over stdio (the default transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
