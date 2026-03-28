"""
Entropic Relevance computation.

The BackgroundModel class and DFG-to-automaton conversion are adapted from the
reference implementation (utils_ER.py) and remain algorithmically identical.
The public interface is reduced to two clean functions:

  dfg_to_transitions(nodes, arcs) -> (trans_table, sources)
  compute_er(dfg_json, traces)    -> (er, fitting_ratio, total_traces)

Key design (preserved from reference):
  - Outgoing probabilities are calculated *excluding* arcs to "■" (end).
  - "▶" → X arcs always have probability 1.0.
  - X → "■" arcs always have probability 1.0.
  - When multiple source states exist (disconnected DFG), the minimum ER
    across sources is returned.
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Background model  (trace-level statistics accumulator)
# ---------------------------------------------------------------------------


class BackgroundModel:
    """
    Accumulates trace statistics needed to compute Entropic Relevance.

    Adapted from the reference implementation (BackgroundModel in utils_ER.py).
    """

    def __init__(self) -> None:
        self.number_of_events: int = 0
        self.number_of_traces: int = 0
        self.total_number_non_fitting_traces: int = 0
        self.labels: set[str] = set()
        self.trace_frequency: dict[str, int] = {}

        self._trace_str: str = ""
        self._lprob: float = 0.0
        self._trace_size: dict[str, int] = {}
        self._log2_of_model_probability: dict[str, float] = {}

    def open_trace(self) -> None:
        self._lprob = 0.0
        self._trace_str = ""

    def process_event(self, label: str, log2_prob: float) -> None:
        self._trace_str += label
        self.number_of_events += 1
        self.labels.add(label)
        self._lprob += log2_prob

    def close_trace(self, trace_length: int, fitting: bool) -> None:
        key = self._trace_str
        self._trace_size[key] = trace_length
        self.number_of_traces += 1
        if fitting:
            self._log2_of_model_probability[key] = self._lprob
        else:
            self.total_number_non_fitting_traces += 1
        self.trace_frequency[key] = self.trace_frequency.get(key, 0) + 1

    def _h0(self, rho: int) -> float:
        n = self.number_of_traces
        if rho == 0 or rho == n:
            return 0.0
        p = rho / n
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    def compute_relevance(self) -> float:
        rho = 0
        cost_bits = 0.0

        for key, freq in self.trace_frequency.items():
            if key in self._log2_of_model_probability:
                bits = -self._log2_of_model_probability[key]
                rho += freq
            else:
                bits = (1 + self._trace_size[key]) * math.log2(1 + len(self.labels))
            cost_bits += (bits * freq) / self.number_of_traces

        return self._h0(rho) + cost_bits


# ---------------------------------------------------------------------------
# DFG → automaton conversion
# ---------------------------------------------------------------------------


def dfg_to_transitions(
    nodes: list[dict[str, Any]],
    arcs: list[dict[str, Any]],
) -> tuple[dict[tuple[int, str], tuple[int, float]], list[int]]:
    """
    Convert a DFG in JSON format to an automaton transition table.

    Probabilities exclude transitions to "■" in their denominator (following
    the reference implementation's design: ending a trace is treated as
    a certain event, not a probabilistic choice).

    Returns:
        trans_table: {(from_id, label): (to_id, log2_probability)}
        sources:     list of node IDs with no incoming arcs (initial states)
    """
    node_info: dict[int, str] = {n["id"]: n["label"] for n in nodes}

    sinks: set[int] = set(node_info.keys())
    sources: list[int] = list(node_info.keys())

    # First pass — compute outgoing frequencies excluding arcs to "■"
    out_freq: dict[int, int] = {}
    for arc in arcs:
        if arc["freq"] <= 0:
            continue
        f, t = arc["from"], arc["to"]
        if node_info[t] != "■":
            out_freq[f] = out_freq.get(f, 0) + arc["freq"]
        sinks.discard(f)
        if t in sources:
            sources.remove(t)

    # Second pass — build transition table
    transitions: dict[tuple[int, str], tuple[int, float]] = {}
    for arc in arcs:
        if arc["freq"] <= 0:
            continue
        f, t = arc["from"], arc["to"]
        label = node_info[t]
        src_label = node_info[f]

        if src_label == "▶" or label == "■":
            prob = 1.0
        elif t not in sinks and f in out_freq and out_freq[f] > 0:
            prob = arc["freq"] / out_freq[f]
        else:
            continue

        transitions[(f, label)] = (t, prob)

    # Build log2 transition table
    trans_table: dict[tuple[int, str], tuple[int, float]] = {
        (f, lbl): (t, math.log2(prob)) for (f, lbl), (t, prob) in transitions.items()
    }

    # Remove sources unreachable from any transition
    all_trans_nodes: set[int] = set()
    for (f, _), (t, _) in trans_table.items():
        all_trans_nodes.add(f)
        all_trans_nodes.add(t)
    sources = [s for s in sources if s in all_trans_nodes]

    return trans_table, sources


# ---------------------------------------------------------------------------
# ER computation
# ---------------------------------------------------------------------------


def _run_source(
    source: int,
    trans_table: dict[tuple[int, str], tuple[int, float]],
    traces: list[list[dict[str, str]]],
) -> tuple[float, BackgroundModel]:
    """Run ER accumulation from one initial source state."""
    model = BackgroundModel()

    for trace in traces:
        curr = source
        non_fitting = False
        model.open_trace()
        trace_len = 0
        last_label = ""

        for event in trace:
            label = event["concept:name"]
            if label == "▶":
                continue
            trace_len += 1
            log2_prob = 0.0
            if not non_fitting and (curr, label) in trans_table:
                curr, log2_prob = trans_table[(curr, label)]
            else:
                non_fitting = True
            model.process_event(label, log2_prob)
            last_label = label

        model.close_trace(
            trace_len,
            fitting=not non_fitting and last_label == "■",
        )

    return model.compute_relevance(), model


def compute_er(
    dfg_json: dict[str, Any],
    traces: list[list[dict[str, str]]],
) -> tuple[float, float, int]:
    """
    Compute Entropic Relevance (ER) of a DFG against a set of traces.

    When multiple source states exist (disconnected DFG), the source that
    minimises ER is selected, matching the reference behaviour.

    Args:
        dfg_json: DFG in {"nodes": [...], "arcs": [...]} format
        traces:   list of traces, each a list of {"concept:name": label} dicts
                  with ▶ / ■ markers

    Returns:
        er            — Entropic Relevance value (nan if computation fails)
        fitting_ratio — fraction of traces that fit the model exactly
        total_traces  — number of traces in the sublog
    """
    if not dfg_json.get("nodes") or not dfg_json.get("arcs"):
        return float("nan"), 0.0, len(traces)
    if not traces:
        return float("nan"), 0.0, 0

    try:
        trans_table, sources = dfg_to_transitions(dfg_json["nodes"], dfg_json["arcs"])
    except Exception:
        return float("nan"), 0.0, len(traces)

    if not sources:
        return float("nan"), 0.0, len(traces)

    results = [_run_source(s, trans_table, traces) for s in sources]
    best_er, best_model = min(results, key=lambda x: x[0])

    total = best_model.number_of_traces
    non_fit = best_model.total_number_non_fitting_traces
    fitting_ratio = 1.0 - non_fit / total if total > 0 else 0.0

    return best_er, fitting_ratio, total
