"""Render a DFG JSON document to an SVG string.

The DFG JSON format is the one produced by ``pmf_tsfm.er.dfg.dfg_to_json``::

    {
        "nodes": [{"label": str, "id": int, "freq": int}, ...],
        "arcs": [{"from": int, "to": int, "freq": int}, ...],
    }

Rendering goes through graphviz (the ``dot`` engine) so the result is a
self-contained ``<svg>`` string that can be embedded directly in ``gr.HTML``.
The renderer is deliberately kept swappable (see ``demo/CLAUDE.md``).
"""

from __future__ import annotations

from typing import Any

import graphviz
from dfg_diff import dfg_diff

# Diff colour coding (PRD #65): grey = matched, amber dashed = it happened but the
# forecast missed it, red dashed = the forecast predicted it but it did not happen.
_MATCHED_COLOUR = "#9e9e9e"
_ADDED_COLOUR = "#ffb300"
_REMOVED_COLOUR = "#e53935"

# Each arc label shows two numbers — the forecast weight and the actual weight —
# distinguished by colour rather than by an arrow, so a reader can tell at a
# glance which side is which (a 0 means the relation is absent on that side).
_FORECAST_COLOUR = "#1e88e5"  # blue
_ACTUAL_COLOUR = "#2e7d32"  # green
_SEP_COLOUR = "#bdbdbd"  # faint separator between the two numbers

# The *relative* diff view labels each arc with the forecast's signed relative
# error vs the actual — (forecast - actual) / actual as an integer percent — so the
# reader sees the *magnitude* of the over/under-shoot. It is colour coded by
# direction along a warm/cool diverging axis: orange = over-forecast (too high),
# teal = under-forecast (too low); 0% and the undefined case (actual = 0) stay
# neutral. The pair is deliberately kept off the edge-class palette (amber added /
# red removed) so a % colour can never be mistaken for an arc colour.
_OVER_COLOUR = "#e8590c"  # orange — forecast over-shot the actual (warm = too much)
_UNDER_COLOUR = "#0c8599"  # teal — forecast under-shot the actual (cool = too little)
_NEUTRAL_COLOUR = "#555555"  # 0% (exact) or — (undefined: actual = 0)

# The diff colours partition arcs the same way in either framing — only what each
# class *means in words* changes with the comparison's nature:
#
# * "accuracy" (bundled, ADR-0004): the comparison is the **actual future**, so a
#   forecast-only arc is an over-prediction and an actual-only arc was missed.
# * "drift" (live upload): the comparison is the **last-known window**, so there is no
#   truth to be right or wrong about — a forecast-only arc is one the forecast
#   *introduces* vs the recent past, an actual-only arc one it *drops*. Never accuracy.
#
# ``dfg_diff(forecast, comparison)`` keys "added" = comparison-only (amber) and
# "removed" = forecast-only (red); the wording below follows that orientation.
_FRAMINGS: dict[str, dict[str, str]] = {
    "accuracy": {
        "comparison": "actual",
        "matched": "matched",
        "added": "happened &#8212; forecast missed it",
        "removed": "forecast predicted it &#8212; did not happen",
        "relative_caption": "-forecast %",
        "over": "over",
        "under": "under",
    },
    "drift": {
        "comparison": "last-known window",
        "matched": "unchanged vs recent past",
        "added": "in recent past, not the forecast",
        "removed": "in the forecast, not recent past",
        "relative_caption": " than recent past, %",
        "over": "higher",
        "under": "lower",
    },
}


def _legend_rows(framing: str) -> list[tuple[str, str]]:
    """Embedded legend rows ``(line-sample HTML, meaning)`` for a framing.

    The sample is a short glyph in that class's colour — a solid rule for matched,
    dashes for the changed classes — so a first-time reader can map line colour/style
    to meaning. The samples are padded to a similar visual width so the column stays
    aligned. The *meanings* are framing-specific (accuracy vs drift).
    """
    words = _FRAMINGS[framing]
    return [
        (
            f'<FONT COLOR="{_MATCHED_COLOUR}">&#9472;&#9472;&#9472;&#9472;&#9472;</FONT>',
            words["matched"],
        ),
        (f'<FONT COLOR="{_ADDED_COLOUR}">&#8211; &#8211; &#8211;</FONT>', words["added"]),
        (f'<FONT COLOR="{_REMOVED_COLOUR}">&#8211; &#8211; &#8211;</FONT>', words["removed"]),
    ]


def _relerr_label(forecast_freq: int, actual_freq: int) -> tuple[str, str]:
    """Signed relative error of the forecast vs the actual, as ``(text, colour)``.

    ``(forecast - actual) / actual`` rounded to an integer percent with an explicit
    sign: positive = the forecast over-shot (orange), negative = under-shot (teal),
    ``0%`` = exact (neutral). When ``actual == 0`` the ratio is undefined (the
    forecast predicted a relation that never happened), so we show an em-dash rather
    than a raw ``inf`` — mirroring the ``n/a`` convention in ``app._fmt``.
    """
    if actual_freq == 0:
        return "&#8212;", _NEUTRAL_COLOUR
    pct = round((forecast_freq - actual_freq) / actual_freq * 100)
    if pct > 0:
        return f"+{pct}%", _OVER_COLOUR
    if pct < 0:
        return f"{pct}%", _UNDER_COLOUR
    return "0%", _NEUTRAL_COLOUR


def _absolute_label(forecast_freq: int, actual_freq: int) -> str:
    """Absolute-view edge label: forecast (blue) | actual (green) — the raw pair."""
    return (
        f'<<FONT COLOR="{_FORECAST_COLOUR}">{forecast_freq}</FONT>'
        f'<FONT COLOR="{_SEP_COLOUR}">&#160;|&#160;</FONT>'
        f'<FONT COLOR="{_ACTUAL_COLOUR}">{actual_freq}</FONT>>'
    )


def _relative_label(forecast_freq: int, actual_freq: int) -> str:
    """Relative-view edge label: the bare signed change % (orange over / teal under)."""
    relerr_text, relerr_colour = _relerr_label(forecast_freq, actual_freq)
    return f'<<FONT COLOR="{relerr_colour}">{relerr_text}</FONT>>'


# Arc-label builder per diff mode — both keep the same grey/amber/red line styling;
# only the text on each arc differs (the raw forecast|actual pair vs the change %).
_LABEL_BUILDERS = {"absolute": _absolute_label, "relative": _relative_label}


def dfg_json_to_svg(dfg: dict[str, Any]) -> str:
    """Render a DFG JSON document to an SVG string.

    Args:
        dfg: DFG in ``{"nodes": [...], "arcs": [...]}`` format.

    Returns:
        An ``<svg>`` document as a string.
    """
    graph = graphviz.Digraph()
    for node in dfg["nodes"]:
        graph.node(str(node["id"]), label=node["label"])
    for arc in dfg["arcs"]:
        graph.edge(str(arc["from"]), str(arc["to"]), label=str(arc["freq"]))
    return graph.pipe(format="svg").decode("utf-8")


def diff_svg(
    forecast_dfg: dict[str, Any],
    actual_dfg: dict[str, Any],
    mode: str = "absolute",
    framing: str = "accuracy",
) -> str:
    """Render the forecast/comparison diff as one colour-coded overlay SVG.

    Arcs are partitioned by :func:`dfg_diff.dfg_diff` and drawn with the same line
    styling regardless of ``mode``/``framing``:

    * **matched** — solid grey,
    * **added** (comparison-only) — amber dashed,
    * **removed** (forecast-only) — red dashed.

    ``mode`` selects only what each arc's *text* shows:

    * ``"absolute"`` — the raw ``forecast (blue) | comparison (green)`` pair,
    * ``"relative"`` — the bare signed change % (warm up / cool down).

    ``framing`` selects only the legend *wording*, so an upload is never read as
    accuracy (ADR-0004):

    * ``"accuracy"`` (bundled) — the comparison is the actual future: ``actual`` /
      ``over/under-forecast %`` / "happened — forecast missed it".
    * ``"drift"`` (live upload) — the comparison is the last-known window:
      ``last-known window`` / ``% change from recent past`` / "in recent past, not
      the forecast". No truth, no accuracy.

    Args:
        forecast_dfg: The forecast DFG (``{"nodes": [...], "arcs": [...]}``).
        actual_dfg:   The comparison DFG (actual-future or last-known window), same shape.
        mode:         ``"absolute"`` (default) or ``"relative"``.
        framing:      ``"accuracy"`` (default) or ``"drift"``.

    Returns:
        An ``<svg>`` document as a string.
    """
    label_for = _LABEL_BUILDERS[mode]
    diff = dfg_diff(forecast_dfg, actual_dfg)
    graph = graphviz.Digraph()

    # Nodes are the union of every relation's endpoints, keyed by label. Each gets
    # an opaque integer id so a label containing a colon (e.g. "SRM: Created") is
    # not parsed by graphviz as node:port syntax (which would merge every such
    # activity into one phantom node and corrupt the diff).
    labels = sorted(
        {
            end
            for entries in diff.values()
            for entry in entries
            for end in (entry["from"], entry["to"])
        }
    )
    node_id = {label: str(i) for i, label in enumerate(labels)}
    for label in labels:
        graph.node(node_id[label], label=label)

    # De-emphasize the unchanged backbone (thin grey) so the amber/red changed
    # arcs (thicker) dominate.
    styling = {
        "matched": {"color": _MATCHED_COLOUR, "penwidth": "1.0"},
        "added": {"color": _ADDED_COLOUR, "style": "dashed", "penwidth": "2.5"},
        "removed": {"color": _REMOVED_COLOUR, "style": "dashed", "penwidth": "2.5"},
    }
    for diff_class, entries in diff.items():
        for entry in entries:
            label = label_for(entry["forecast_freq"], entry["actual_freq"])
            graph.edge(
                node_id[entry["from"]],
                node_id[entry["to"]],
                label=label,
                **styling[diff_class],
            )

    _add_legend(graph, mode, framing)
    return graph.pipe(format="svg").decode("utf-8")


def _add_legend(graph: graphviz.Digraph, mode: str = "absolute", framing: str = "accuracy") -> None:
    """Embed a compact key as a single HTML-table node in its own cluster.

    The table has two parts: an edge-label key explaining what the arc text means in
    this ``mode`` (the forecast/comparison number pair, or the signed change %), and
    one row per diff class pairing a line sample with its meaning. Both the key and the
    rows are ``framing``-specific (accuracy vs drift). A single table keeps the legend
    tight instead of sprawling.
    """
    words = _FRAMINGS[framing]
    # The edge-label key spans the full width on its own row, so it does not widen
    # the line-sample column and throw the grid off. It is mode-specific: the
    # absolute view explains the bicolour number pair, the relative view the % axis.
    if mode == "relative":
        label_key = (
            '<FONT COLOR="#555555">edge label: </FONT>'
            f'<FONT COLOR="{_OVER_COLOUR}">{words["over"]}</FONT>'
            '<FONT COLOR="#555555"> / </FONT>'
            f'<FONT COLOR="{_UNDER_COLOUR}">{words["under"]}</FONT>'
            f'<FONT COLOR="#555555">{words["relative_caption"]}</FONT>'
        )
    else:
        label_key = (
            '<FONT COLOR="#555555">edge numbers: </FONT>'
            f'<FONT COLOR="{_FORECAST_COLOUR}">forecast</FONT>'
            f'<FONT COLOR="{_SEP_COLOUR}">&#160;|&#160;</FONT>'
            f'<FONT COLOR="{_ACTUAL_COLOUR}">{words["comparison"]}</FONT>'
        )
    # The three diff classes form an aligned 2-column grid: line sample | meaning.
    class_rows = "".join(
        f'<TR><TD ALIGN="LEFT">{sample}</TD><TD ALIGN="LEFT">{meaning}</TD></TR>'
        for sample, meaning in _legend_rows(framing)
    )
    table = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="3">'
        '<TR><TD COLSPAN="2" ALIGN="CENTER"><B>Legend</B></TD></TR>'
        f'<TR><TD COLSPAN="2" ALIGN="LEFT">{label_key}</TD></TR>'
        f"{class_rows}</TABLE>>"
    )
    with graph.subgraph(name="cluster_legend") as legend:
        legend.attr(color="#bdbdbd", style="dashed")
        legend.node("_legend", label=table, shape="plaintext")
