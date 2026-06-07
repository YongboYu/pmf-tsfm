"""Pure DFG-JSON builders shared by the precompute and live paths.

Both the bundled precompute (``precompute_demo``) and the live upload path
(``forecast_live``) turn a window of DF-relation frequencies into the
``{"nodes": [...], "arcs": [...]}`` JSON the renderer consumes. That builder used to
live in ``precompute_demo`` and reused ``pmf_tsfm.er.dfg.dfg_to_json`` — but the live
path must run in the lean Space serve env **without** ``pmf_tsfm`` (and its heavy
model deps). So the two small, pure functions live here, depending only on numpy.

``dfg_to_json`` is a faithful copy of ``pmf_tsfm.er.dfg.dfg_to_json`` (same node-id
and frequency conventions), so the bundled assets it produces are byte-identical.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def dfg_to_json(dfg: dict[tuple[str, str], int]) -> dict[str, Any]:
    """Serialise a ``{(source, target): freq}`` DFG to the JSON node/arc format.

    Node IDs: ▶ = 0, ■ = 1, other activities in encounter order.
    Node frequency = sum of incoming arc frequencies (or outgoing for ▶).

    A faithful copy of ``pmf_tsfm.er.dfg.dfg_to_json`` (kept here so the live path
    needs no ``pmf_tsfm`` at serve time); changes must stay in lock-step with it.
    """
    reverse_map: dict[str, int] = {"▶": 0, "■": 1}
    node_freq: dict[str, int] = {"▶": 0, "■": 0}

    for (src, tgt), freq in dfg.items():
        for label in (src, tgt):
            if label not in reverse_map:
                reverse_map[label] = len(reverse_map)
                node_freq[label] = 0
        if src == "▶":
            node_freq["▶"] += freq
        else:
            node_freq[tgt] += freq

    arcs = [
        {"from": reverse_map[src], "to": reverse_map[tgt], "freq": freq}
        for (src, tgt), freq in dfg.items()
    ]
    nodes = [
        {"label": label, "id": nid, "freq": node_freq.get(label, 0)}
        for label, nid in reverse_map.items()
    ]
    return {"nodes": nodes, "arcs": arcs}


def frequencies_to_dfg_json(window: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    """Build a clean DFG JSON from one window's DF-relation frequencies.

    Sums the horizon, rounds, drops zero-frequency relations, and keeps the raw
    ▶/■ markers so ``dfg_to_json`` maps them to the canonical start/end nodes —
    no duplicate Start/End nodes and no artificial freq-1 arcs.

    Args:
        window:        shape (horizon, n_features) — one forecast/actual window.
        feature_names: ``"A -> B"`` column names aligned with the last axis.

    Returns:
        DFG in ``{"nodes": [...], "arcs": [...]}`` format.
    """
    freqs = np.clip(np.round(window.sum(axis=0)), 0, None).astype(int)
    dfg: dict[tuple[str, str], int] = {}
    for name, freq in zip(feature_names, freqs, strict=True):
        if freq > 0:
            src, tgt = name.split("->", 1)
            dfg[(src.strip(), tgt.strip())] = int(freq)
    return dfg_to_json(dfg)
