"""Tests for the dedicated FastMCP server (#133).

Driven through an in-process MCP client (``create_connected_server_and_client_session``) so the
tools are exercised over a real client round-trip, not just as plain functions. The repo has no
``pytest-asyncio``, so each test runs its coroutine with ``asyncio.run``.

The three fast tests are CI-safe (no model import). The real backtest is marked ``slow`` and
skips unless the bundled sepsis XES is present (it is untracked, so absent in CI).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

server = pytest.importorskip("server")
from mcp.shared.memory import (  # noqa: E402  (after importorskip, by design)
    create_connected_server_and_client_session as client_session,
)

# Repo root: this file is mcp/tests/test_server.py → parents[2] is the repo root.
SEPSIS_XES = Path(__file__).resolve().parents[2] / "data" / "processed_logs" / "sepsis.xes"

EXPECTED_TOOLS = {"list_models", "forecast_backtest", "forecast_only"}


async def _list_tools():
    async with client_session(server.mcp._mcp_server) as client:
        return (await client.list_tools()).tools


async def _call(name: str, arguments: dict):
    """Call a tool through the client; return its parsed payload.

    ``dict``-returning tools come back as a JSON text block; ``list``-returning tools as
    structured content under the ``"result"`` key.
    """
    async with client_session(server.mcp._mcp_server) as client:
        result = await client.call_tool(name, arguments)
    assert not result.isError, result
    if result.structuredContent is not None:
        return result.structuredContent.get("result", result.structuredContent)
    return json.loads(result.content[0].text)


def test_exactly_three_tools_with_autoderived_schema():
    """The server exposes the three api tools, each with an auto-derived schema + docstring."""
    tools = {t.name: t for t in asyncio.run(_list_tools())}
    assert set(tools) == EXPECTED_TOOLS
    for tool in tools.values():
        assert (tool.description or "").strip(), f"{tool.name} has no description"
        assert tool.inputSchema.get("type") == "object", f"{tool.name} has no input schema"
    # The forecast tools take the log path; the args came from the type hints, not hand-written.
    for name in ("forecast_backtest", "forecast_only"):
        assert "input_path" in tools[name].inputSchema["properties"]


def test_list_models_roundtrip():
    """``list_models`` returns the same non-empty list as the api, through the client."""
    from pmf_tsfm import api

    models = asyncio.run(_call("list_models", {}))
    assert models == api.list_models()
    assert models, "no models discovered"
    assert "chronos/chronos2" in models


def test_forecast_backtest_plumbing_passes_args_and_bundle(monkeypatch):
    """The tool forwards its arguments to the api and returns the bundle intact (no compute)."""
    captured = {}
    bundle = {
        "predictions_path": "out/p.npy",
        "quantiles_path": "out/q.npy",
        "metrics": {"mae": 1.5, "mae_std": 0.1, "rmse": 2.0, "rmse_std": 0.2, "er": 3.0},
        "feature_names": ["a__b"],
        "n_windows": 4,
        "model": "chronos2",
        "horizon": 7,
    }

    def fake(input_path, **kwargs):
        captured["input_path"] = input_path
        captured.update(kwargs)
        return bundle

    monkeypatch.setattr("pmf_tsfm.api.forecast_backtest", fake)

    out = asyncio.run(
        _call("forecast_backtest", {"input_path": "log.xes", "horizon": 14, "compute_er": False})
    )
    assert out == bundle
    assert captured["input_path"] == "log.xes"
    assert captured["horizon"] == 14
    assert captured["compute_er"] is False


@pytest.mark.slow
@pytest.mark.skipif(not SEPSIS_XES.exists(), reason="bundled sepsis.xes not present")
def test_forecast_backtest_real_sepsis_xes():
    """Acceptance: a real backtest on the bundled sepsis XES yields finite MAE/RMSE + ER."""
    import math

    out = asyncio.run(_call("forecast_backtest", {"input_path": str(SEPSIS_XES)}))
    metrics = out["metrics"]
    assert math.isfinite(metrics["mae"])
    assert math.isfinite(metrics["rmse"])
    assert metrics["er"] is not None and math.isfinite(metrics["er"])
