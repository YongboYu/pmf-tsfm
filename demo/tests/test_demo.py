"""Tests for the bundled forecast explorer demo (slice 1).

Two deep modules are tested here through their public interfaces:

render.py
  - test_dfg_json_to_svg_returns_svg          valid <svg> string
  - (further behaviours added per TDD cycle)

forecast.py
  - (added per TDD cycle)

The Gradio glue (app.py) and the precompute script are smoke-tested only.
"""

from __future__ import annotations

import json

import forecast
import numpy as np
import pytest
from forecast import forecast_bundled
from precompute_demo import frequencies_to_dfg_json, precompute_one
from render import dfg_json_to_svg

# ---------------------------------------------------------------------------
# Shared DFG fixture — a tiny ▶ → A → B → ■ graph
# ---------------------------------------------------------------------------

SMALL_DFG = {
    "nodes": [
        {"label": "▶", "id": 0, "freq": 5},
        {"label": "■", "id": 1, "freq": 5},
        {"label": "A", "id": 2, "freq": 5},
        {"label": "B", "id": 3, "freq": 4},
    ],
    "arcs": [
        {"from": 0, "to": 2, "freq": 5},  # ▶ -> A
        {"from": 2, "to": 3, "freq": 4},  # A -> B
        {"from": 3, "to": 1, "freq": 5},  # B -> ■
    ],
}


# ---------------------------------------------------------------------------
# render.dfg_json_to_svg
# ---------------------------------------------------------------------------


def test_dfg_json_to_svg_returns_svg():
    """A DFG JSON renders to an SVG document string."""
    svg = dfg_json_to_svg(SMALL_DFG)
    assert isinstance(svg, str)
    assert "<svg" in svg


def test_dfg_json_to_svg_node_and_arc_counts():
    """Every node and every arc appears exactly once in the SVG."""
    svg = dfg_json_to_svg(SMALL_DFG)
    assert svg.count('class="node"') == len(SMALL_DFG["nodes"])
    assert svg.count('class="edge"') == len(SMALL_DFG["arcs"])


def test_dfg_json_to_svg_shows_activity_labels():
    """Activity names and the ▶/■ markers are rendered as node text."""
    svg = dfg_json_to_svg(SMALL_DFG)
    for label in ("A", "B", "▶", "■"):
        assert label in svg


def test_dfg_json_to_svg_shows_arc_weights():
    """Each DF arc is labelled with its frequency (the arc weight)."""
    svg = dfg_json_to_svg(SMALL_DFG)
    # The A -> B arc has the distinctive weight 4.
    assert ">4<" in svg


# ---------------------------------------------------------------------------
# forecast.forecast_bundled
# ---------------------------------------------------------------------------

# A distinct actual-future DFG so tests can tell the two panes apart.
ACTUAL_DFG = {
    "nodes": [
        {"label": "▶", "id": 0, "freq": 6},
        {"label": "■", "id": 1, "freq": 6},
        {"label": "A", "id": 2, "freq": 6},
        {"label": "B", "id": 3, "freq": 6},
    ],
    "arcs": [
        {"from": 0, "to": 2, "freq": 6},
        {"from": 2, "to": 3, "freq": 6},
        {"from": 3, "to": 1, "freq": 6},
    ],
}

METRICS = {"er": 1.5127795943816802, "mae": 3.21, "rmse": 4.56}


@pytest.fixture
def asset_dir(tmp_path, monkeypatch):
    """A fixture asset directory for bpi2017 × chronos2 with known contents."""
    base = tmp_path / "bpi2017" / "chronos2"
    base.mkdir(parents=True)
    (base / "forecast_dfg.json").write_text(json.dumps(SMALL_DFG))
    (base / "actual_dfg.json").write_text(json.dumps(ACTUAL_DFG))
    (base / "metrics.json").write_text(json.dumps(METRICS))
    monkeypatch.setattr(forecast, "ASSETS_ROOT", tmp_path)
    return tmp_path


def test_forecast_bundled_returns_triple(asset_dir):
    """forecast_bundled returns the forecast/actual/metrics triple."""
    result = forecast_bundled("bpi2017", "chronos2", 7)
    assert set(result) == {"forecast_dfg", "actual_dfg", "metrics"}


def test_forecast_bundled_reads_committed_assets(asset_dir):
    """The triple's values come straight from the committed assets."""
    result = forecast_bundled("bpi2017", "chronos2", 7)
    assert result["forecast_dfg"] == SMALL_DFG
    assert result["actual_dfg"] == ACTUAL_DFG  # distinct from the forecast pane
    assert result["metrics"] == METRICS


def test_forecast_bundled_metrics_are_accuracy(asset_dir):
    """Bundled metrics expose ER / MAE / RMSE (genuine accuracy, ADR-0004)."""
    metrics = forecast_bundled("bpi2017", "chronos2", 7)["metrics"]
    assert set(metrics) == {"er", "mae", "rmse"}
    assert all(isinstance(v, (int, float)) for v in metrics.values())


def test_forecast_bundled_unknown_dataset_raises(asset_dir):
    """An un-precomputed dataset × model fails clearly rather than silently."""
    with pytest.raises(FileNotFoundError):
        forecast_bundled("nope", "chronos2", 7)


# ---------------------------------------------------------------------------
# demo/precompute_demo.py (glue — smoke + the DFG-builder it relies on)
# ---------------------------------------------------------------------------


def test_frequencies_to_dfg_json_has_single_start_end():
    """The builder keeps one ▶ and one ■ — no duplicate Start/End, no freq-1 arcs."""
    window = np.array([[1, 2, 1], [1, 2, 1], [1, 2, 1]])  # sums to [3, 6, 3]
    dfg = frequencies_to_dfg_json(window, ["▶ -> A", "A -> B", "B -> ■"])
    labels = [n["label"] for n in dfg["nodes"]]
    assert labels.count("▶") == 1 and labels.count("■") == 1
    assert "Start" not in labels and "End" not in labels
    assert {a["freq"] for a in dfg["arcs"]} == {3, 6}  # summed weights, no artificial 1s


def _write_synthetic_outputs(root):
    """A minimal stand-in for outputs/zero_shot + outputs/er for one dataset × model."""
    feature_names = ["▶ -> A", "A -> B", "B -> ■"]
    src = root / "zero_shot" / "testds" / "chronos_2"
    src.mkdir(parents=True)
    (src / "TESTDS_chronos_2_metadata.json").write_text(
        json.dumps({"feature_names": feature_names, "data_metadata": {"dataset_name": "TESTDS"}})
    )
    window = np.array([[1, 2, 1], [1, 2, 1], [1, 2, 1]], dtype=float)
    arr = np.stack([window, window])  # (2 windows, horizon 3, 3 features)
    np.save(src / "TESTDS_chronos_2_predictions.npy", arr)
    np.save(src / "TESTDS_chronos_2_targets.npy", arr)  # preds == targets → MAE/RMSE = 0
    er_dir = root / "er" / "zero_shot" / "TESTDS" / "chronos_2"
    er_dir.mkdir(parents=True)
    (er_dir / "TESTDS_chronos_2_er.json").write_text(
        json.dumps({"windows": [{"pred_er": 0.9}, {"pred_er": 1.23}]})
    )


def test_precompute_one_emits_valid_assets(tmp_path):
    """precompute_one reuses existing outputs (no model run) and writes valid assets."""
    outputs_root = tmp_path / "outputs"
    _write_synthetic_outputs(outputs_root)

    out_dir = precompute_one(
        "testds", "chronos2", outputs_root=outputs_root, assets_root=tmp_path / "assets"
    )

    forecast_dfg = json.loads((out_dir / "forecast_dfg.json").read_text())
    metrics = json.loads((out_dir / "metrics.json").read_text())

    assert {"nodes", "arcs"} <= set(forecast_dfg)
    labels = {n["label"] for n in forecast_dfg["nodes"]}
    assert "▶" in labels and "■" in labels  # ▶/■ markers
    assert all(isinstance(a["freq"], int) and a["freq"] >= 0 for a in forecast_dfg["arcs"])
    assert set(metrics) == {"er", "mae", "rmse"}
    assert metrics["er"] == 1.23  # ER of the final window
    assert metrics["mae"] == 0.0 and metrics["rmse"] == 0.0


# ---------------------------------------------------------------------------
# demo/app.py (glue — smoke only; skipped where gradio is not installed)
# ---------------------------------------------------------------------------


def test_app_load_produces_twin_panes_and_metrics(asset_dir):
    """app.load wires forecast_bundled → twin SVG panes + ER/MAE/RMSE strip."""
    pytest.importorskip("gradio")
    import app

    forecast_html, actual_html, er, mae, rmse = app.load("bpi2017", "chronos2")
    assert "<svg" in forecast_html and "<svg" in actual_html
    # The two panes show distinct DFGs (forecast vs actual-future).
    assert forecast_html != actual_html
    assert (er, mae, rmse) == ("1.513", "3.210", "4.560")


def test_app_builds_blocks(asset_dir):
    """The Blocks app assembles without launching."""
    gr = pytest.importorskip("gradio")
    import app

    assert isinstance(app.build(), gr.Blocks)
