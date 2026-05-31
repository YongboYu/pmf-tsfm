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
