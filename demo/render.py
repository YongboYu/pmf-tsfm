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

    # Nodes are the union of every relation's endpoints, keyed (and id'd) by label.
    labels = {
        end
        for entries in diff.values()
        for entry in entries
        for end in (entry["from"], entry["to"])
    }
    for label in labels:
        graph.node(label, label=label)

    styling = {
        "matched": {"color": _MATCHED_COLOUR},
        "added": {"color": _ADDED_COLOUR, "style": "dashed"},
        "removed": {"color": _REMOVED_COLOUR, "style": "dashed"},
    }
    for diff_class, entries in diff.items():
        for entry in entries:
            graph.edge(entry["from"], entry["to"], label=str(entry["freq"]), **styling[diff_class])
    return graph.pipe(format="svg").decode("utf-8")
