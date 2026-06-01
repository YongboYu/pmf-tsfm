"""Tests for the bundled forecast explorer demo.

The deep modules are tested through their public interfaces:

render.py
  - dfg_json_to_svg — a DFG JSON renders to a valid <svg> string
  - diff_svg        — a forecast/actual pair renders to a colour-coded overlay

dfg_diff.py
  - dfg_diff        — DF relations partition into matched / added / removed

forecast.py
  - forecast_bundled — reads the committed assets into the forecast triple

The Gradio glue (app.py) and the precompute script are smoke-tested only.
"""

from __future__ import annotations

import json
import re

import forecast
import numpy as np
import pytest
from dfg_diff import dfg_diff
from forecast import forecast_bundled
from precompute_demo import frequencies_to_dfg_json, precompute_one
from render import dfg_json_to_svg, diff_svg

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
# dfg_diff.dfg_diff
#
# Classifies DF relations (keyed on (from_label, to_label)) between a forecast
# DFG and the actual-future DFG, with conventional diff(from=forecast, to=actual)
# semantics:
#   matched = in both                          (grey)
#   added   = in actual, not forecast          (amber dashed: happened, missed)
#   removed = in forecast, not actual          (red dashed: predicted, didn't happen)
# ---------------------------------------------------------------------------


def _rels(entries):
    """The set of (from, to) relation keys in a diff class."""
    return {(e["from"], e["to"]) for e in entries}


def test_dfg_diff_identical_all_matched():
    """Diffing a DFG against itself classifies every relation as matched."""
    diff = dfg_diff(SMALL_DFG, SMALL_DFG)
    assert _rels(diff["matched"]) == {("▶", "A"), ("A", "B"), ("B", "■")}
    assert diff["added"] == [] and diff["removed"] == []


def test_dfg_diff_disjoint_all_added_and_removed():
    """No shared relations → forecast's are all removed, actual's all added."""
    forecast = {
        "nodes": [{"label": "▶", "id": 0}, {"label": "X", "id": 1}],
        "arcs": [{"from": 0, "to": 1, "freq": 2}],  # ▶ -> X
    }
    actual = {
        "nodes": [{"label": "▶", "id": 9}, {"label": "Y", "id": 8}],
        "arcs": [{"from": 9, "to": 8, "freq": 3}],  # ▶ -> Y
    }
    diff = dfg_diff(forecast, actual)
    assert diff["matched"] == []
    assert _rels(diff["removed"]) == {("▶", "X")}
    assert _rels(diff["added"]) == {("▶", "Y")}


def test_dfg_diff_partial_overlap_is_label_keyed_with_both_freqs():
    """Relations key on labels (not ids); every entry carries both weights."""
    forecast = {
        "nodes": [{"label": "A", "id": 0}, {"label": "B", "id": 1}, {"label": "C", "id": 2}],
        "arcs": [
            {"from": 0, "to": 1, "freq": 4},  # A -> B   (shared)
            {"from": 1, "to": 2, "freq": 7},  # B -> C   (forecast only -> removed)
        ],
    }
    actual = {
        # Same labels, deliberately different ids, to prove label-keying.
        "nodes": [{"label": "A", "id": 5}, {"label": "B", "id": 6}, {"label": "D", "id": 7}],
        "arcs": [
            {"from": 5, "to": 6, "freq": 9},  # A -> B   (shared, different weight)
            {"from": 6, "to": 7, "freq": 2},  # B -> D   (actual only -> added)
        ],
    }
    diff = dfg_diff(forecast, actual)

    assert _rels(diff["matched"]) == {("A", "B")}
    assert _rels(diff["added"]) == {("B", "D")}
    assert _rels(diff["removed"]) == {("B", "C")}
    # Every entry exposes both weights; the absent side is 0.
    assert diff["matched"][0]["forecast_freq"] == 4
    assert diff["matched"][0]["actual_freq"] == 9
    assert diff["added"][0]["forecast_freq"] == 0  # forecast missed it
    assert diff["added"][0]["actual_freq"] == 2
    assert diff["removed"][0]["forecast_freq"] == 7
    assert diff["removed"][0]["actual_freq"] == 0  # it did not happen


def test_dfg_diff_handles_start_end_marker_relations():
    """Relations touching the ▶/■ markers classify like any other relation."""
    # forecast: ▶ -> A -> ■ ; actual: ▶ -> A (drops the A -> ■ ending).
    forecast = {
        "nodes": [{"label": "▶", "id": 0}, {"label": "A", "id": 1}, {"label": "■", "id": 2}],
        "arcs": [{"from": 0, "to": 1, "freq": 5}, {"from": 1, "to": 2, "freq": 5}],
    }
    actual = {
        "nodes": [{"label": "▶", "id": 0}, {"label": "A", "id": 1}],
        "arcs": [{"from": 0, "to": 1, "freq": 6}],
    }
    diff = dfg_diff(forecast, actual)
    assert _rels(diff["matched"]) == {("▶", "A")}
    assert _rels(diff["removed"]) == {("A", "■")}  # forecast predicted an ending that didn't happen
    assert diff["added"] == []


def test_dfg_diff_empty_inputs_yield_empty_partition():
    """Empty DFGs diff to an empty (but well-formed) partition — no crash."""
    empty = {"nodes": [], "arcs": []}
    diff = dfg_diff(empty, empty)
    assert diff == {"matched": [], "added": [], "removed": []}


# ---------------------------------------------------------------------------
# render.diff_svg
# ---------------------------------------------------------------------------


def test_diff_svg_returns_svg():
    """A pair of DFGs renders to a single overlay SVG document string."""
    svg = diff_svg(SMALL_DFG, ACTUAL_DFG)
    assert isinstance(svg, str)
    assert "<svg" in svg


def test_diff_svg_colour_codes_the_three_diff_classes():
    """A drifting pair emits matched (grey solid), added (amber dashed),
    removed (red dashed) arcs — one of each."""
    # A -> B matched; B -> D added (actual only); B -> C removed (forecast only).
    forecast = {
        "nodes": [{"label": "A", "id": 0}, {"label": "B", "id": 1}, {"label": "C", "id": 2}],
        "arcs": [{"from": 0, "to": 1, "freq": 4}, {"from": 1, "to": 2, "freq": 7}],
    }
    actual = {
        "nodes": [{"label": "A", "id": 5}, {"label": "B", "id": 6}, {"label": "D", "id": 7}],
        "arcs": [{"from": 5, "to": 6, "freq": 9}, {"from": 6, "to": 7, "freq": 2}],
    }
    svg = diff_svg(forecast, actual)

    # All three diff classes are present, by colour.
    grey, amber, red = "#9e9e9e", "#ffb300", "#e53935"
    assert grey in svg and amber in svg and red in svg

    # Added/removed are dashed; matched is solid. Inspect each coloured edge path.
    def stroke_paths(colour):
        return re.findall(rf'stroke="{colour}"[^/]*?d="', svg)

    assert all("stroke-dasharray" in p for p in stroke_paths(amber))  # added dashed
    assert all("stroke-dasharray" in p for p in stroke_paths(red))  # removed dashed
    assert all("stroke-dasharray" not in p for p in stroke_paths(grey))  # matched solid


# A drifting forecast/actual pair: A->B matched (4,9), B->D added (0,2, forecast
# missed it), B->C removed (7,0, predicted but didn't happen).
DRIFT_FORECAST = {
    "nodes": [{"label": "A", "id": 0}, {"label": "B", "id": 1}, {"label": "C", "id": 2}],
    "arcs": [{"from": 0, "to": 1, "freq": 4}, {"from": 1, "to": 2, "freq": 7}],
}
DRIFT_ACTUAL = {
    "nodes": [{"label": "A", "id": 5}, {"label": "B", "id": 6}, {"label": "D", "id": 7}],
    "arcs": [{"from": 5, "to": 6, "freq": 9}, {"from": 6, "to": 7, "freq": 2}],
}

# Edge numbers are colour-coded by side (forecast vs actual), not joined by "→".
FORECAST_COLOUR = "#1e88e5"  # blue
ACTUAL_COLOUR = "#2e7d32"  # green


def test_diff_svg_labels_arcs_with_bicolour_forecast_actual():
    """Each arc shows its forecast weight and actual weight as two distinctly
    coloured numbers (forecast = blue, actual = green) — no "→" join."""
    svg = diff_svg(DRIFT_FORECAST, DRIFT_ACTUAL)
    # matched A->B: forecast 4 (blue), actual 9 (green).
    assert re.search(rf'fill="{FORECAST_COLOUR}"[^>]*>4<', svg)
    assert re.search(rf'fill="{ACTUAL_COLOUR}"[^>]*>9<', svg)
    # added B->D: forecast 0 (blue, it missed it), actual 2 (green).
    assert re.search(rf'fill="{FORECAST_COLOUR}"[^>]*>0<', svg)
    assert re.search(rf'fill="{ACTUAL_COLOUR}"[^>]*>2<', svg)
    # The arrow join is gone — colour carries the forecast/actual distinction.
    assert "→" not in svg


def _stroke_width(svg, colour):
    """The stroke-width of the first ``colour`` edge path (1.0 if unset)."""
    path = re.search(rf'<path[^>]*stroke="{colour}"[^>]*?/>', svg).group(0)
    m = re.search(r'stroke-width="([\d.]+)"', path)
    return float(m.group(1)) if m else 1.0


def test_diff_svg_deemphasizes_matched_arcs():
    """Matched arcs are thinner than the amber/red changed arcs, so drift pops
    against the unchanged backbone."""
    svg = diff_svg(DRIFT_FORECAST, DRIFT_ACTUAL)
    grey, amber, red = "#9e9e9e", "#ffb300", "#e53935"
    matched = _stroke_width(svg, grey)
    assert matched < _stroke_width(svg, amber)  # added stands out
    assert matched < _stroke_width(svg, red)  # removed stands out


def test_diff_svg_embeds_legend():
    """The diff graph carries an embedded legend explaining both the three
    line-style classes and the forecast/actual number colours."""
    svg = diff_svg(DRIFT_FORECAST, DRIFT_ACTUAL)
    assert "Legend" in svg
    # The three diff classes, in forecast-centric wording.
    assert "matched" in svg
    assert "missed it" in svg  # amber: it happened but the forecast missed it
    assert "did not happen" in svg  # red: predicted but it did not happen
    # The forecast/actual number-colour key (so the bicolour edge labels read).
    assert "forecast" in svg and "actual" in svg
    assert FORECAST_COLOUR in svg and ACTUAL_COLOUR in svg


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


# Minimal valid pre-rendered SVGs, distinct per pane so the panes differ.
FORECAST_SVG = '<svg id="forecast"><g class="node"></g></svg>'
ACTUAL_SVG = '<svg id="actual"><g class="node"></g></svg>'
DIFF_SVG = '<svg id="diff">#9e9e9e #ffb300 #e53935</svg>'


@pytest.fixture
def asset_dir(tmp_path, monkeypatch):
    """A fixture asset directory for bpi2017 × chronos2 with known contents."""
    base = tmp_path / "bpi2017" / "chronos2"
    base.mkdir(parents=True)
    (base / "forecast_dfg.json").write_text(json.dumps(SMALL_DFG))
    (base / "actual_dfg.json").write_text(json.dumps(ACTUAL_DFG))
    (base / "metrics.json").write_text(json.dumps(METRICS))
    (base / "forecast.svg").write_text(FORECAST_SVG)
    (base / "actual.svg").write_text(ACTUAL_SVG)
    (base / "diff.svg").write_text(DIFF_SVG)
    monkeypatch.setattr(forecast, "ASSETS_ROOT", tmp_path)
    return tmp_path


def test_forecast_bundled_returns_triple(asset_dir):
    """forecast_bundled returns the forecast/actual/metrics triple plus pre-rendered SVGs."""
    result = forecast_bundled("bpi2017", "chronos2", 7)
    assert set(result) == {
        "forecast_dfg",
        "actual_dfg",
        "metrics",
        "forecast_svg",
        "actual_svg",
        "diff_svg",
    }


def test_forecast_bundled_reads_pre_rendered_svgs(asset_dir):
    """The triple carries the pre-rendered SVG figures straight from the assets."""
    result = forecast_bundled("bpi2017", "chronos2", 7)
    assert result["forecast_svg"] == FORECAST_SVG
    assert result["actual_svg"] == ACTUAL_SVG
    assert result["diff_svg"] == DIFF_SVG


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


def _write_synthetic_outputs(root, *, cap_dir="TESTDS", model_dir="chronos_2"):
    """A minimal stand-in for outputs/zero_shot + outputs/er for one dataset × model.

    ``cap_dir`` is the Capitalized on-disk dataset directory (and ``dataset_name``);
    ``model_dir`` is the on-disk model directory — together they mirror the real
    ``outputs/zero_shot/<CapDir>/<model_dir>/`` layout that ``precompute_one`` reads.
    """
    feature_names = ["▶ -> A", "A -> B", "B -> ■"]
    src = root / "zero_shot" / cap_dir / model_dir
    src.mkdir(parents=True)
    (src / f"{cap_dir}_{model_dir}_metadata.json").write_text(
        json.dumps({"feature_names": feature_names, "data_metadata": {"dataset_name": cap_dir}})
    )
    window = np.array([[1, 2, 1], [1, 2, 1], [1, 2, 1]], dtype=float)
    arr = np.stack([window, window])  # (2 windows, horizon 3, 3 features)
    np.save(src / f"{cap_dir}_{model_dir}_predictions.npy", arr)
    np.save(src / f"{cap_dir}_{model_dir}_targets.npy", arr)  # preds == targets → MAE/RMSE = 0
    er_dir = root / "er" / "zero_shot" / cap_dir / model_dir
    er_dir.mkdir(parents=True)
    (er_dir / f"{cap_dir}_{model_dir}_er.json").write_text(
        json.dumps({"windows": [{"pred_er": 0.9}, {"pred_er": 1.23}]})
    )


def test_precompute_one_emits_valid_assets(tmp_path, monkeypatch):
    """precompute_one reuses existing outputs (no model run) and writes valid assets."""
    import precompute_demo

    monkeypatch.setitem(precompute_demo.DATASET_DIRS, "testds", "TESTDS")
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
    # The pre-rendered SVG figures sit alongside the regenerable JSON source.
    for name in ("forecast.svg", "actual.svg", "diff.svg"):
        assert "<svg" in (out_dir / name).read_text()


def test_precompute_matrix_emits_all_twelve(tmp_path):
    """Precompute over the full BUNDLED matrix writes valid assets for every pair.

    Builds a synthetic ``outputs/`` for each (dataset, model) pair — using the
    DATASET_DIRS / MODEL_DIRS mappings so the src dirs match precompute_one's
    lookup — then asserts the full 4×3 = 12 matrix of asset directories, each
    with valid forecast/actual JSON DFGs (▶/■ markers), an {er,mae,rmse} metrics
    file, and the three pre-rendered SVG figures.
    """
    import precompute_demo

    assert len(precompute_demo.BUNDLED) == 12
    assert len({d for d, _ in precompute_demo.BUNDLED}) == 4  # four datasets
    assert len({m for _, m in precompute_demo.BUNDLED}) == 3  # three models

    outputs_root = tmp_path / "outputs"
    assets_root = tmp_path / "assets"
    for dataset, model in precompute_demo.BUNDLED:
        _write_synthetic_outputs(
            outputs_root,
            cap_dir=precompute_demo.DATASET_DIRS[dataset],
            model_dir=precompute_demo.MODEL_DIRS[model],
        )

    written = set()
    for dataset, model in precompute_demo.BUNDLED:
        out_dir = precompute_one(dataset, model, outputs_root=outputs_root, assets_root=assets_root)
        written.add((dataset, model))

        forecast_dfg = json.loads((out_dir / "forecast_dfg.json").read_text())
        actual_dfg = json.loads((out_dir / "actual_dfg.json").read_text())
        metrics = json.loads((out_dir / "metrics.json").read_text())

        for dfg in (forecast_dfg, actual_dfg):
            assert {"nodes", "arcs"} <= set(dfg)
            labels = {n["label"] for n in dfg["nodes"]}
            assert "▶" in labels and "■" in labels  # ▶/■ markers
        assert set(metrics) == {"er", "mae", "rmse"}
        for name in ("forecast.svg", "actual.svg", "diff.svg"):
            assert "<svg" in (out_dir / name).read_text()

    assert written == set(precompute_demo.BUNDLED)
    assert len(written) == 12


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


def test_fmt_shows_na_for_non_finite_metric():
    """A non-finite metric (e.g. an ER that could not be computed) shows ``n/a``."""
    pytest.importorskip("gradio")
    import app

    assert app._fmt(1.5127) == "1.513"
    assert app._fmt(float("nan")) == "n/a"
    assert app._fmt(float("inf")) == "n/a"


def test_app_head_inlines_vendored_pan_zoom(asset_dir):
    """The built Blocks inline the vendored svg-pan-zoom + the pane initializer."""
    pytest.importorskip("gradio")
    import app

    head = app.build().demo_head
    assert "svg-pan-zoom v3.6.2" in head  # vendored source inlined, not a CDN URL
    assert "svgPanZoom(" in head and ".dfg-pane svg" in head  # pane initializer


def test_app_panes_carry_dfg_pane_class(asset_dir):
    """All three SVG panes carry the `dfg-pane` class so JS can target them."""
    pytest.importorskip("gradio")
    import app

    panes = [
        c
        for c in app.build().blocks.values()
        if getattr(c, "elem_classes", None) and "dfg-pane" in c.elem_classes
    ]
    assert len(panes) == 3  # forecast, actual-future, diff overlay


@pytest.fixture
def drift_asset_dir(tmp_path, monkeypatch):
    """Assets whose forecast and actual genuinely drift (matched + added + removed)."""
    # forecast: A -> B, B -> C ; actual: A -> B, B -> D  → matched A->B, removed B->C, added B->D.
    forecast_dfg = {
        "nodes": [{"label": "A", "id": 0}, {"label": "B", "id": 1}, {"label": "C", "id": 2}],
        "arcs": [{"from": 0, "to": 1, "freq": 4}, {"from": 1, "to": 2, "freq": 7}],
    }
    actual_dfg = {
        "nodes": [{"label": "A", "id": 0}, {"label": "B", "id": 1}, {"label": "D", "id": 2}],
        "arcs": [{"from": 0, "to": 1, "freq": 9}, {"from": 1, "to": 2, "freq": 2}],
    }
    base = tmp_path / "bpi2017" / "chronos2"
    base.mkdir(parents=True)
    (base / "forecast_dfg.json").write_text(json.dumps(forecast_dfg))
    (base / "actual_dfg.json").write_text(json.dumps(actual_dfg))
    (base / "metrics.json").write_text(json.dumps(METRICS))
    (base / "forecast.svg").write_text(FORECAST_SVG)
    (base / "actual.svg").write_text(ACTUAL_SVG)
    (base / "diff.svg").write_text(DIFF_SVG)
    monkeypatch.setattr(forecast, "ASSETS_ROOT", tmp_path)
    return tmp_path


def test_app_load_diff_produces_colour_coded_overlay(drift_asset_dir):
    """app.load_diff renders the forecast/actual pair into one diff overlay pane."""
    pytest.importorskip("gradio")
    import app

    diff_html = app.load_diff("bpi2017", "chronos2")
    assert "<svg" in diff_html
    # The overlay carries the diff colour coding (grey / amber / red).
    assert "#9e9e9e" in diff_html and "#ffb300" in diff_html and "#e53935" in diff_html
