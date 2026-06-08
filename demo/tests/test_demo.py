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
from upload_guard import MAX_UPLOAD_BYTES, SMALL_LOG_BYTES

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


def test_diff_svg_absolute_labels_arcs_with_bicolour_forecast_actual():
    """In the (default) absolute mode each arc shows its forecast weight and actual
    weight as two distinctly coloured numbers (forecast = blue, actual = green) —
    no "→" join and no relative % (that is the relative view's job)."""
    svg = diff_svg(DRIFT_FORECAST, DRIFT_ACTUAL)  # mode defaults to "absolute"
    # matched A->B: forecast 4 (blue), actual 9 (green).
    assert re.search(rf'fill="{FORECAST_COLOUR}"[^>]*>4<', svg)
    assert re.search(rf'fill="{ACTUAL_COLOUR}"[^>]*>9<', svg)
    # added B->D: forecast 0 (blue, it missed it), actual 2 (green).
    assert re.search(rf'fill="{FORECAST_COLOUR}"[^>]*>0<', svg)
    assert re.search(rf'fill="{ACTUAL_COLOUR}"[^>]*>2<', svg)
    # The arrow join is gone — colour carries the forecast/actual distinction.
    assert "→" not in svg
    # No relative change-% rides the absolute view: the over/under direction colours
    # are emitted only by the relative label, so their absence proves no % label is
    # present. (Checking for a bare "%" is too blunt — some graphviz versions emit a
    # "%" in the SVG boilerplate, unrelated to any arc label.)
    assert OVER_COLOUR not in svg and UNDER_COLOUR not in svg


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


def test_diff_svg_defaults_to_accuracy_framing():
    """The default framing is accuracy, so the bundled path's legend is untouched."""
    svg = diff_svg(DRIFT_FORECAST, DRIFT_ACTUAL)
    assert "actual" in svg
    assert "last-known window" not in svg


def test_diff_svg_drift_framing_relabels_legend_off_accuracy():
    """framing='drift' (the live upload) speaks of the last-known window and the recent
    past — never 'actual' or forecast-accuracy wording (ADR-0004: an upload is not scored)."""
    svg = diff_svg(DRIFT_FORECAST, DRIFT_ACTUAL, framing="drift")
    # graphviz serialises the hyphen in "last-known" as the &#45; entity; match either form.
    assert "known window" in svg  # the comparison's true nature (last-known window)
    assert "recent past" in svg  # the diff-class wording is drift-framed
    # the accuracy vocabulary is gone
    assert "actual" not in svg
    assert "missed it" not in svg and "did not happen" not in svg


def test_diff_svg_relative_drift_caption_is_change_from_recent_past():
    """The relative drift view reads '% change from recent past', not over/under-forecast %."""
    svg = diff_svg(DRIFT_FORECAST, DRIFT_ACTUAL, mode="relative", framing="drift")
    assert "recent past" in svg
    assert "forecast %" not in svg  # the accuracy caption "over/under-forecast %" is gone


# A pair exercising every signed-relative-error row on one render. Signed error is
# (forecast - actual) / actual, integer-rounded:
#   X->Y matched over  : 745 / 558 -> +34%
#   Y->Z matched under : 300 / 558 -> -46%
#   Z->W matched equal : 500 / 500 ->   0%
#   W->V added         :   0 / 200 -> -100% (it happened, the forecast missed it)
#   V->X removed       : 400 /   0 ->  —    (predicted but never happened; ratio undefined)
RELERR_FORECAST = {
    "nodes": [
        {"label": "X", "id": 0},
        {"label": "Y", "id": 1},
        {"label": "Z", "id": 2},
        {"label": "W", "id": 3},
        {"label": "V", "id": 4},
    ],
    "arcs": [
        {"from": 0, "to": 1, "freq": 745},
        {"from": 1, "to": 2, "freq": 300},
        {"from": 2, "to": 3, "freq": 500},
        {"from": 4, "to": 0, "freq": 400},
    ],
}
RELERR_ACTUAL = {
    "nodes": [
        {"label": "X", "id": 0},
        {"label": "Y", "id": 1},
        {"label": "Z", "id": 2},
        {"label": "W", "id": 3},
        {"label": "V", "id": 4},
    ],
    "arcs": [
        {"from": 0, "to": 1, "freq": 558},
        {"from": 1, "to": 2, "freq": 558},
        {"from": 2, "to": 3, "freq": 500},
        {"from": 3, "to": 4, "freq": 200},
    ],
}

# The signed-error % is colour-coded by direction so over/under reads at a glance.
# The pair is a warm/cool diverging axis kept off the amber/red arc-class colours.
OVER_COLOUR = "#e8590c"  # orange — forecast over-shot the actual (positive)
UNDER_COLOUR = "#0c8599"  # teal — forecast under-shot the actual (negative)
# graphviz serialises a leading ASCII "-" in HTML-like labels as the numeric entity
# &#45; in the emitted SVG (it renders as a minus); match either form.
MINUS = r"(?:-|&#45;)"


def test_diff_svg_arc_label_shows_over_forecast_percentage():
    """A matched arc whose forecast over-shot the actual shows a signed ``+NN%``
    in the over-forecast colour, after the two raw weights."""
    svg = diff_svg(RELERR_FORECAST, RELERR_ACTUAL, mode="relative")
    # X->Y: forecast 745 vs actual 558 -> +34%, over-forecast (orange).
    assert re.search(rf'fill="{OVER_COLOUR}"[^>]*>\+34%<', svg)


def test_diff_svg_arc_label_shows_under_forecast_percentage():
    """A matched arc whose forecast under-shot the actual shows a negative
    ``-NN%`` in the under-forecast colour."""
    svg = diff_svg(RELERR_FORECAST, RELERR_ACTUAL, mode="relative")
    # Y->Z: forecast 300 vs actual 558 -> -46%, under-forecast (teal).
    assert re.search(rf'fill="{UNDER_COLOUR}"[^>]*>{MINUS}46%<', svg)


def test_diff_svg_arc_label_shows_zero_percent_when_exact():
    """A matched arc whose forecast equals the actual shows a neutral ``0%`` —
    not signed, not in an over/under colour."""
    svg = diff_svg(RELERR_FORECAST, RELERR_ACTUAL, mode="relative")
    # Z->W: forecast 500 vs actual 500 -> 0%.
    assert re.search(r">0%<", svg)
    # 0% is neutral: it is not painted in either direction colour.
    assert not re.search(rf'fill="{OVER_COLOUR}"[^>]*>0%<', svg)
    assert not re.search(rf'fill="{UNDER_COLOUR}"[^>]*>0%<', svg)


def test_diff_svg_added_arc_shows_minus_100_percent():
    """An added arc (it happened, forecast = 0) is a complete under-shot: -100%."""
    svg = diff_svg(RELERR_FORECAST, RELERR_ACTUAL, mode="relative")
    # W->V: forecast 0 vs actual 200 -> -100%, under-forecast (teal).
    assert re.search(rf'fill="{UNDER_COLOUR}"[^>]*>{MINUS}100%<', svg)


def test_diff_svg_removed_arc_shows_dash_when_ratio_undefined():
    """A removed arc (predicted but actual = 0) has an undefined ratio: show an
    em-dash, never a raw inf/nan or a bare ``%``."""
    svg = diff_svg(RELERR_FORECAST, RELERR_ACTUAL, mode="relative")
    # V->X: forecast 400 vs actual 0 -> undefined -> em-dash, in neutral grey.
    assert re.search(r'fill="#555555"[^>]*>—<', svg)
    assert "inf" not in svg.lower() and "nan" not in svg.lower()


def test_diff_svg_legend_names_the_percentage_key():
    """The legend's edge-label key names the over/under signed-error percentage,
    showing the two direction colours so the reader can decode an arc."""
    svg = diff_svg(RELERR_FORECAST, RELERR_ACTUAL, mode="relative")
    assert "over" in svg and "under" in svg
    assert OVER_COLOUR in svg and UNDER_COLOUR in svg


def test_diff_svg_relative_shows_only_the_change_percent():
    """The relative view labels arcs with the bare change % alone — none of the raw
    forecast/actual numbers nor their blue/green colours ride the arcs."""
    svg = diff_svg(RELERR_FORECAST, RELERR_ACTUAL, mode="relative")
    assert "%" in svg  # the change % is shown
    # The blue/green forecast/actual number colours belong to the absolute view only.
    assert FORECAST_COLOUR not in svg and ACTUAL_COLOUR not in svg


def test_diff_svg_relative_colours_do_not_clash_with_arc_classes():
    """The over/under % colours stay off the amber (added) / red (removed) arc
    colours, so a % label can never be mistaken for an arc-class signal."""
    amber, red = "#ffb300", "#e53935"
    assert OVER_COLOUR not in {amber, red}
    assert UNDER_COLOUR not in {amber, red}
    # And the arc classes are still drawn in the relative view (lines kept in both).
    svg = diff_svg(DRIFT_FORECAST, DRIFT_ACTUAL, mode="relative")
    assert "#9e9e9e" in svg and amber in svg and red in svg


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

METRICS = {"er": 1.5127795943816802, "truth_er": 0.8765432109876543, "mae": 3.21, "rmse": 4.56}


# Minimal valid pre-rendered SVGs, distinct per pane so the panes differ.
FORECAST_SVG = '<svg id="forecast"><g class="node"></g></svg>'
ACTUAL_SVG = '<svg id="actual"><g class="node"></g></svg>'
# Two diff overlays — same line coding, distinct so the panes can be told apart.
DIFF_ABS_SVG = '<svg id="diff-abs">#9e9e9e #ffb300 #e53935</svg>'
DIFF_REL_SVG = '<svg id="diff-rel">#9e9e9e #ffb300 #e53935</svg>'


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
    (base / "diff_absolute.svg").write_text(DIFF_ABS_SVG)
    (base / "diff_relative.svg").write_text(DIFF_REL_SVG)
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
        "diff_absolute_svg",
        "diff_relative_svg",
    }


def test_forecast_bundled_reads_pre_rendered_svgs(asset_dir):
    """The triple carries the pre-rendered SVG figures straight from the assets."""
    result = forecast_bundled("bpi2017", "chronos2", 7)
    assert result["forecast_svg"] == FORECAST_SVG
    assert result["actual_svg"] == ACTUAL_SVG
    assert result["diff_absolute_svg"] == DIFF_ABS_SVG
    assert result["diff_relative_svg"] == DIFF_REL_SVG


def test_forecast_bundled_reads_committed_assets(asset_dir):
    """The triple's values come straight from the committed assets."""
    result = forecast_bundled("bpi2017", "chronos2", 7)
    assert result["forecast_dfg"] == SMALL_DFG
    assert result["actual_dfg"] == ACTUAL_DFG  # distinct from the forecast pane
    assert result["metrics"] == METRICS


def test_forecast_bundled_metrics_are_accuracy(asset_dir):
    """Bundled metrics expose forecast ER, truth-ER baseline, MAE, RMSE (ADR-0004)."""
    metrics = forecast_bundled("bpi2017", "chronos2", 7)["metrics"]
    assert set(metrics) == {"er", "truth_er", "mae", "rmse"}
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


def _sublog(rows):
    """A tiny sublog DataFrame in the columns build_truth_dfg / extract_traces expect."""
    import pandas as pd

    return pd.DataFrame(
        rows, columns=["case:concept:name", "concept:name", "time:timestamp"]
    ).astype({"time:timestamp": "datetime64[ns, UTC]"})


def test_truth_er_from_sublog_is_zero_for_a_single_deterministic_trace():
    """The truth DFG replays its own only trace perfectly → zero surprisal (ER 0)."""
    from precompute_demo import truth_er_from_sublog

    sublog = _sublog(
        [
            ("c1", "A", "2020-01-01T00:00:00+00:00"),
            ("c1", "B", "2020-01-01T01:00:00+00:00"),
        ]
    )
    er = truth_er_from_sublog(sublog)
    assert isinstance(er, float)
    assert er == 0.0


def test_truth_er_from_sublog_is_nan_for_an_empty_window():
    """A window with no events yields nan (no crash) — the n/a baseline case."""
    import math

    from precompute_demo import truth_er_from_sublog

    assert math.isnan(truth_er_from_sublog(_sublog([])))


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
    # Real er.json windows carry a date string; the final one drives the truth-ER
    # baseline (precompute extracts that window's sublog from the XES log).
    (er_dir / f"{cap_dir}_{model_dir}_er.json").write_text(
        json.dumps(
            {
                "windows": [
                    {"window": "2019-12-30_2020-01-05", "pred_er": 0.9},
                    {"window": "2020-01-01_2020-01-01", "pred_er": 1.23},
                ]
            }
        )
    )


def _patch_log_loader(monkeypatch):
    """Stub the XES loader so precompute's truth-ER path runs without a real log.

    Real ``data/processed_logs/*.xes`` are gitignored (absent in CI), so the
    precompute smoke tests inject a tiny A→B log whose one event sits in the
    synthetic final window (2020-01-01) → a deterministic truth ER of 0.0.
    """
    import precompute_demo

    precompute_demo._prepared_log.cache_clear()
    monkeypatch.setattr(
        precompute_demo,
        "load_event_log",
        lambda _path: _sublog(
            [
                ("c1", "A", "2020-01-01T00:00:00+00:00"),
                ("c1", "B", "2020-01-01T01:00:00+00:00"),
            ]
        ),
    )


def test_precompute_one_emits_valid_assets(tmp_path, monkeypatch):
    """precompute_one reuses existing outputs (no model run) and writes valid assets."""
    import precompute_demo

    monkeypatch.setitem(precompute_demo.DATASET_DIRS, "testds", "TESTDS")
    _patch_log_loader(monkeypatch)
    outputs_root = tmp_path / "outputs"
    _write_synthetic_outputs(outputs_root)

    out_dir = precompute_one(
        "testds",
        "chronos2",
        outputs_root=outputs_root,
        assets_root=tmp_path / "assets",
        log_dir=tmp_path / "logs",
    )

    forecast_dfg = json.loads((out_dir / "forecast_dfg.json").read_text())
    metrics = json.loads((out_dir / "metrics.json").read_text())

    assert {"nodes", "arcs"} <= set(forecast_dfg)
    labels = {n["label"] for n in forecast_dfg["nodes"]}
    assert "▶" in labels and "■" in labels  # ▶/■ markers
    assert all(isinstance(a["freq"], int) and a["freq"] >= 0 for a in forecast_dfg["arcs"])
    assert set(metrics) == {"er", "truth_er", "mae", "rmse"}
    assert metrics["er"] == 1.23  # forecast ER of the final window
    assert metrics["truth_er"] == 0.0  # truth-DFG baseline on the same final window
    assert metrics["mae"] == 0.0 and metrics["rmse"] == 0.0
    # The pre-rendered SVG figures sit alongside the regenerable JSON source.
    for name in ("forecast.svg", "actual.svg", "diff_absolute.svg", "diff_relative.svg"):
        assert "<svg" in (out_dir / name).read_text()


def test_precompute_matrix_emits_all_twelve(tmp_path, monkeypatch):
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

    _patch_log_loader(monkeypatch)
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
        out_dir = precompute_one(
            dataset,
            model,
            outputs_root=outputs_root,
            assets_root=assets_root,
            log_dir=tmp_path / "logs",
        )
        written.add((dataset, model))

        forecast_dfg = json.loads((out_dir / "forecast_dfg.json").read_text())
        actual_dfg = json.loads((out_dir / "actual_dfg.json").read_text())
        metrics = json.loads((out_dir / "metrics.json").read_text())

        for dfg in (forecast_dfg, actual_dfg):
            assert {"nodes", "arcs"} <= set(dfg)
            labels = {n["label"] for n in dfg["nodes"]}
            assert "▶" in labels and "■" in labels  # ▶/■ markers
        assert set(metrics) == {"er", "truth_er", "mae", "rmse"}
        for name in ("forecast.svg", "actual.svg", "diff_absolute.svg", "diff_relative.svg"):
            assert "<svg" in (out_dir / name).read_text()

    assert written == set(precompute_demo.BUNDLED)
    assert len(written) == 12


# ---------------------------------------------------------------------------
# demo/app.py (glue — smoke only; skipped where gradio is not installed)
# ---------------------------------------------------------------------------


def test_app_load_produces_twin_panes_and_metrics(asset_dir):
    """app.load wires forecast_bundled → twin SVG panes + forecast/truth ER + MAE/RMSE."""
    pytest.importorskip("gradio")
    import app

    forecast_html, actual_html, er, truth_er, mae, rmse = app.load("bpi2017", "chronos2")
    assert "<svg" in forecast_html and "<svg" in actual_html
    # The two panes show distinct DFGs (forecast vs actual-future).
    assert forecast_html != actual_html
    # The truth-ER baseline sits beside the forecast ER in the strip.
    assert (er, truth_er, mae, rmse) == ("1.513", "0.877", "3.210", "4.560")


def test_app_strip_carries_forecast_and_truth_er_boxes(asset_dir):
    """The metrics strip labels the forecast ER and the truth-ER baseline distinctly."""
    pytest.importorskip("gradio")
    import app

    labels = {
        getattr(c, "label", None)
        for c in app.build().blocks.values()
        if type(c).__name__ == "Textbox"
    }
    assert "ER — forecast" in labels
    assert "ER — truth (baseline)" in labels


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
    """Every SVG pane carries the `dfg-pane` class so JS can target them — four on the
    bundled tab and four on the live tab."""
    pytest.importorskip("gradio")
    import app

    panes = [
        c
        for c in app.build().blocks.values()
        if getattr(c, "elem_classes", None) and "dfg-pane" in c.elem_classes
    ]
    # bundled: forecast, actual-future, diff abs, diff rel; live: forecast, last-known,
    # drift abs, drift rel.
    assert len(panes) == 8


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
    (base / "diff_absolute.svg").write_text(DIFF_ABS_SVG)
    (base / "diff_relative.svg").write_text(DIFF_REL_SVG)
    monkeypatch.setattr(forecast, "ASSETS_ROOT", tmp_path)
    return tmp_path


def test_app_load_diff_produces_both_overlays(drift_asset_dir):
    """app.load_diff renders the forecast/actual pair into the absolute and relative
    overlay panes — both carrying the grey/amber/red diff colour coding."""
    pytest.importorskip("gradio")
    import app

    absolute_html, relative_html = app.load_diff("bpi2017", "chronos2")
    for diff_html in (absolute_html, relative_html):
        assert "<svg" in diff_html
        # The overlay carries the diff colour coding (grey / amber / red).
        assert "#9e9e9e" in diff_html and "#ffb300" in diff_html and "#e53935" in diff_html
    # The two panes are distinct documents (absolute vs relative).
    assert absolute_html != relative_html


# ---------------------------------------------------------------------------
# app.py — the live upload tab (slice 3b, #115). The GUI glue over forecast_live;
# the ZeroGPU forecast itself is stubbed (real inference is post-merge, on hardware).
# ---------------------------------------------------------------------------

_LIVE_STUB_BUNDLE = {
    "forecast_svg": "<svg>forecast</svg>",
    "comparison_svg": "<svg>last-known</svg>",
    "diff_absolute_svg": "<svg>diff-abs</svg>",
    "diff_relative_svg": "<svg>diff-rel</svg>",
    "drift": {
        "added": [{"from": "A", "to": "C", "forecast_freq": 3, "recent_freq": 0}],
        "removed": [{"from": "A", "to": "D", "forecast_freq": 0, "recent_freq": 2}],
        "stable": [{"from": "A", "to": "B", "forecast_freq": 5, "recent_freq": 4}],
    },
}


def _drain(gen):
    """Consume a ``run_live`` generator, returning its final yielded output tuple."""
    final = None
    for item in gen:
        final = item
    return final


def _markdown_values(blocks) -> list[str]:
    """Every non-empty gr.Markdown string rendered in a built Blocks app."""
    import gradio as gr

    return [c.value for c in blocks.blocks.values() if isinstance(c, gr.Markdown) and c.value]


def test_app_builds_both_tabs():
    """The app exposes the bundled explorer and the live upload tab side by side."""
    gr = pytest.importorskip("gradio")
    import app

    tabs = [c for c in app.build().blocks.values() if isinstance(c, gr.Tab)]
    labels = {t.label for t in tabs}
    assert {"Bundled explorer", "Live upload (your log)"} <= labels


def test_run_live_renders_panes_and_drift_on_success(monkeypatch):
    """A successful upload yields the twin panes, both drift overlays, and a drift tally."""
    pytest.importorskip("gradio")
    import app

    monkeypatch.setattr(app, "_run_live", lambda log_path, model: _LIVE_STUB_BUNDLE)
    status, forecast, comparison, diff_abs, diff_rel, summary = _drain(
        app.run_live("up.xes", "chronos2")
    )

    assert "✅" in status
    for html in (forecast, comparison, diff_abs, diff_rel):
        assert "<svg" in html
    # Drift, never accuracy: the tally counts added/dropped and disclaims accuracy.
    assert "adds" in summary and "drops" in summary
    assert "not accuracy" in summary.lower()
    assert "mae" not in summary.lower() and "rmse" not in summary.lower()


def test_run_live_yields_interim_progress_before_result(monkeypatch):
    """Forecast streams an immediate 'working' state (blank panes) before the slow GPU
    result, so the live tab shows progress instead of freezing for ~120s."""
    pytest.importorskip("gradio")
    import app

    monkeypatch.setattr(app, "_run_live", lambda log_path, model: _LIVE_STUB_BUNDLE)
    states = list(app.run_live("up.xes", "chronos2"))

    assert len(states) >= 2, "expected an interim progress state, then the result"
    interim_status, *interim_panes = states[0]
    assert "Forecasting" in interim_status
    assert interim_panes == ["", "", "", "", ""]  # no stale/partial graph while working
    assert "✅" in states[-1][0]  # the final state is the completed forecast


def test_run_live_surfaces_upload_rejection_with_blank_panes(monkeypatch):
    """A guard rejection shows a clear message and leaves the panes blank (no stale graph)."""
    pytest.importorskip("gradio")
    import app
    from upload_guard import UploadRejected

    def _reject(log_path, model):
        raise UploadRejected("log is 40.0 MB; the upload cap is 25.0 MB.")

    monkeypatch.setattr(app, "_run_live", _reject)
    status, *panes = _drain(app.run_live("big.xes", "chronos2"))

    assert "rejected" in status.lower() and "cap" in status
    assert panes == ["", "", "", "", ""]


def test_run_live_prompts_when_no_file_uploaded():
    """Pressing Forecast with no upload prompts for a log rather than reaching the GPU."""
    pytest.importorskip("gradio")
    import app

    status, *panes = _drain(app.run_live(None, "chronos2"))
    assert "Upload" in status
    assert panes == ["", "", "", "", ""]


def test_run_live_reports_unexpected_failure_without_crashing(monkeypatch):
    """Any non-guard failure is surfaced as a message, not an uncaught exception."""
    pytest.importorskip("gradio")
    import app

    def _boom(log_path, model):
        raise RuntimeError("graphviz exploded")

    monkeypatch.setattr(app, "_run_live", _boom)
    status, *panes = _drain(app.run_live("up.xes", "chronos2"))
    assert "Could not forecast" in status
    assert panes == ["", "", "", "", ""]


def test_live_tab_offers_example_log(tmp_path):
    """The live tab ships a one-click example: a small committed log wired to the upload."""
    gr = pytest.importorskip("gradio")
    import app

    # The committed sample exists and sits under the small-log gate (a snappy example).
    assert app.EXAMPLE_LOG.exists(), "the live-tab example log must be committed"
    assert app.EXAMPLE_LOG.stat().st_size < SMALL_LOG_BYTES

    # It is wired into the UI as a gr.Examples (a gr.Dataset) referencing that file.
    datasets = [c for c in app.build().blocks.values() if isinstance(c, gr.Dataset)]
    samples = [str(s) for d in datasets for row in d.samples for s in row]
    assert any(app.EXAMPLE_LOG.name in s for s in samples), "example not wired to the upload"


def test_live_example_is_framed_as_a_stand_in_for_upload():
    """The example is labelled as a stand-in for the user's own log (not a duplicate of the
    bundled sepsis entry), keeping the live tab's 'upload your own' framing clear."""
    pytest.importorskip("gradio")
    import app

    note = app._example_note()
    assert "sepsis" in note.lower() and "your own" in note.lower()
    assert any(note in m for m in _markdown_values(app.build())), (
        "the example stand-in note must be shown on the live tab"
    )


def test_live_caps_note_is_derived_from_guard_constants():
    """The caps shown to the user are formatted from the guard constants (single source of
    truth), and the caption is rendered on the live tab — not a hand-typed number."""
    pytest.importorskip("gradio")
    import app

    note = app._live_caps_note()
    # Same formatting the guard uses in its rejection messages, so the up-front cap is the
    # exact number a user sees if their upload is refused (upload_guard.py). The live tab is
    # Chronos-2 only, so the note shows just that one cap (the gated small-log threshold
    # applies only to the bundled families, which the live picker never offers).
    assert f"{MAX_UPLOAD_BYTES / 1e6:.1f}" in note

    markdowns = _markdown_values(app.build())
    assert any(note in m for m in markdowns), "caps note must be shown on the live tab"


def test_tabs_carry_onboarding_accordions():
    """Each tab carries a collapsed 'What am I looking at?' explainer for first-time visitors."""
    gr = pytest.importorskip("gradio")
    import app

    accordions = [c for c in app.build().blocks.values() if isinstance(c, gr.Accordion)]
    explainers = [a for a in accordions if a.label == "What am I looking at?"]
    assert len(explainers) >= 2, "expected an onboarding accordion on each tab"
    assert all(not a.open for a in explainers), "the explainer should start collapsed"
