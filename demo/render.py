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

# Embedded legend rows: (line-sample HTML, meaning). The sample is a short glyph
# in that class's colour — a solid rule for matched, dashes for the changed
# classes — so a first-time reader can map line colour/style to meaning. The
# samples are padded to a similar visual width so the column stays aligned.
_LEGEND_ROWS = [
    (f'<FONT COLOR="{_MATCHED_COLOUR}">&#9472;&#9472;&#9472;&#9472;&#9472;</FONT>', "matched"),
    (
        f'<FONT COLOR="{_ADDED_COLOUR}">&#8211; &#8211; &#8211;</FONT>',
        "happened &#8212; forecast missed it",
    ),
    (
        f'<FONT COLOR="{_REMOVED_COLOUR}">&#8211; &#8211; &#8211;</FONT>',
        "forecast predicted it &#8212; did not happen",
    ),
]


def _arc_label(forecast_freq: int, actual_freq: int) -> str:
    """HTML-like edge label: forecast weight (blue) · actual weight (green)."""
    return (
        f'<<FONT COLOR="{_FORECAST_COLOUR}">{forecast_freq}</FONT>'
        f'<FONT COLOR="{_SEP_COLOUR}">&#160;|&#160;</FONT>'
        f'<FONT COLOR="{_ACTUAL_COLOUR}">{actual_freq}</FONT>>'
    )


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


def diff_svg(forecast_dfg: dict[str, Any], actual_dfg: dict[str, Any]) -> str:
    """Render the forecast/actual diff as one colour-coded overlay SVG.

    Arcs are partitioned by :func:`dfg_diff.dfg_diff` and drawn as:

    * **matched** — solid grey,
    * **added** (it happened, the forecast missed it) — amber dashed,
    * **removed** (the forecast predicted it, it did not happen) — red dashed.

    Args:
        forecast_dfg: The forecast DFG (``{"nodes": [...], "arcs": [...]}``).
        actual_dfg:   The actual-future DFG, same shape.

    Returns:
        An ``<svg>`` document as a string.
    """
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
            label = _arc_label(entry["forecast_freq"], entry["actual_freq"])
            graph.edge(
                node_id[entry["from"]],
                node_id[entry["to"]],
                label=label,
                **styling[diff_class],
            )

    _add_legend(graph)
    return graph.pipe(format="svg").decode("utf-8")


def _add_legend(graph: graphviz.Digraph) -> None:
    """Embed a compact key as a single HTML-table node in its own cluster.

    The table has two parts: a forecast/actual number-colour key (explaining the
    bicolour edge labels) and one row per diff class pairing a line sample with
    its meaning. A single table keeps the legend tight instead of sprawling.
    """
    # The forecast/actual number-colour key spans the full width on its own row,
    # so it does not widen the line-sample column and throw the grid off.
    number_key = (
        '<FONT COLOR="#555555">edge numbers: </FONT>'
        f'<FONT COLOR="{_FORECAST_COLOUR}">forecast</FONT>'
        f'<FONT COLOR="{_SEP_COLOUR}">&#160;|&#160;</FONT>'
        f'<FONT COLOR="{_ACTUAL_COLOUR}">actual</FONT>'
    )
    # The three diff classes form an aligned 2-column grid: line sample | meaning.
    class_rows = "".join(
        f'<TR><TD ALIGN="LEFT">{sample}</TD><TD ALIGN="LEFT">{meaning}</TD></TR>'
        for sample, meaning in _LEGEND_ROWS
    )
    table = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="3">'
        '<TR><TD COLSPAN="2" ALIGN="CENTER"><B>Legend</B></TD></TR>'
        f'<TR><TD COLSPAN="2" ALIGN="LEFT">{number_key}</TD></TR>'
        f"{class_rows}</TABLE>>"
    )
    with graph.subgraph(name="cluster_legend") as legend:
        legend.attr(color="#bdbdbd", style="dashed")
        legend.node("_legend", label=table, shape="plaintext")
