"""Classify how a forecast DFG drifted from the actual-future DFG.

``dfg_diff`` is a pure deep module: it takes two DFG JSON documents (the
``{"nodes": [...], "arcs": [...]}`` shape from ``pmf_tsfm.er.dfg.dfg_to_json``)
and partitions their directly-follows (DF) relations into three classes.

Relations are keyed on the ``(from_label, to_label)`` activity pair — *not* on
node ids, which number independently in the two DFGs. Semantics follow a
conventional ``diff(from=forecast, to=actual)``:

    matched  relation present in both forecast and actual
    added    present in actual but not forecast  (it happened, the forecast missed it)
    removed  present in forecast but not actual   (the forecast predicted it, it did not happen)

These map to the Diff view's colour coding (grey / amber dashed / red dashed)
and are reused by the upload drift descriptors in a later slice.
"""

from __future__ import annotations

from typing import Any


def _relations(dfg: dict[str, Any]) -> dict[tuple[str, str], int]:
    """Map each DF relation ``(from_label, to_label)`` to its arc weight."""
    label = {node["id"]: node["label"] for node in dfg["nodes"]}
    return {(label[arc["from"]], label[arc["to"]]): arc["freq"] for arc in dfg["arcs"]}


def dfg_diff(
    forecast_dfg: dict[str, Any], actual_dfg: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Partition DF relations of two DFGs into matched / added / removed.

    Args:
        forecast_dfg: The forecast DFG (``{"nodes": [...], "arcs": [...]}``).
        actual_dfg:   The actual-future DFG, same shape.

    Returns:
        ``{"matched": [...], "added": [...], "removed": [...]}`` where each entry
        is ``{"from": label, "to": label, "forecast_freq": int, "actual_freq":
        int}``. Every entry carries *both* weights as a ``forecast → actual``
        transition; the side where the relation is absent is ``0`` (so ``added``
        has ``forecast_freq == 0`` and ``removed`` has ``actual_freq == 0``).
    """
    forecast = _relations(forecast_dfg)
    actual = _relations(actual_dfg)

    def entry(rel: tuple[str, str]) -> dict[str, Any]:
        return {
            "from": rel[0],
            "to": rel[1],
            "forecast_freq": forecast.get(rel, 0),
            "actual_freq": actual.get(rel, 0),
        }

    matched = [entry(rel) for rel in actual if rel in forecast]
    added = [entry(rel) for rel in actual if rel not in forecast]
    removed = [entry(rel) for rel in forecast if rel not in actual]
    return {"matched": matched, "added": added, "removed": removed}
