"""
DFG construction utilities for Entropic Relevance evaluation.

All operations on event logs use vectorised pandas — no per-case Python loops
and no pm4py DFG discovery.  pm4py is used only for reading XES files.

DFG internal representation: dict[tuple[str, str], int]
  {(source_activity, target_activity): frequency}

For the automaton / ER computation the DFG is serialised to JSON format:
  {"nodes": [{"label": str, "id": int, "freq": int}, ...],
   "arcs":  [{"from": int, "to": int, "freq": int}, ...]}

Special symbols
---------------
  "▶"  / "■"  — artificial start / end in the raw XES log
  "Start" / "End" — cleaned names used inside the DFG / automaton
  node IDs: ▶ → 0, ■ → 1, other activities in encounter order
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Mapping for special process-mining symbols when they appear as *activities*
_ACTIVITY_CLEAN: dict[str, str] = {"▶": "Start", "■": "End"}


def _clean(name: str) -> str:
    return _ACTIVITY_CLEAN.get(name, name)


# ---------------------------------------------------------------------------
# XES loading
# ---------------------------------------------------------------------------


def load_event_log(xes_path: str | Path) -> pd.DataFrame:
    """
    Load an XES event log into a tidy DataFrame using pm4py.

    Returns a DataFrame with at least the columns:
        case:concept:name  —  case identifier
        concept:name       —  activity label
        time:timestamp     —  UTC-aware datetime
    """
    import pm4py

    path = Path(xes_path)
    if not path.exists():
        raise FileNotFoundError(f"XES event log not found: {path}")

    log = pm4py.read_xes(str(path))
    df: pd.DataFrame = pm4py.convert_to_dataframe(log)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Sublog extraction  (vectorised — pre-compute _next_ts once, then mask)
# ---------------------------------------------------------------------------


def prepare_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by (case, timestamp) and pre-compute the per-case next-event timestamp.

    Must be called once before calling extract_sublog() in a loop.
    The returned DataFrame has an extra "_next_ts" column.
    """
    df = df.sort_values(["case:concept:name", "time:timestamp"]).copy()
    df["_next_ts"] = df.groupby("case:concept:name")["time:timestamp"].shift(-1)
    return df


def extract_sublog(
    df: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Vectorised sublog extraction for one rolling window [ws, we].

    An event is included when either its timestamp OR its direct successor's
    timestamp falls in [ws, we].  This preserves DF relations that straddle
    the window boundary.

    Args:
        df:           prepared DataFrame from prepare_log() (has _next_ts)
        window_start: inclusive start of the window
        window_end:   inclusive end of the window

    Returns:
        Filtered DataFrame without the _next_ts helper column.
    """
    ts = df["time:timestamp"]
    nts = df["_next_ts"]
    mask = ts.between(window_start, window_end) | nts.between(window_start, window_end)
    return df.loc[mask].drop(columns=["_next_ts"])


# ---------------------------------------------------------------------------
# Trace extraction
# ---------------------------------------------------------------------------


def extract_traces(df: pd.DataFrame) -> list[list[dict[str, str]]]:
    """
    Convert a sublog DataFrame into a list of traces suitable for compute_er().

    Each trace is a list of {"concept:name": label} dicts with artificial
    ▶ (start) and ■ (end) markers prepended / appended.
    Special symbols in the log (▶, ■) are normalised to "Start" / "End".
    """
    case_col = "case:concept:name"
    act_col = "concept:name"
    ts_col = "time:timestamp"

    sub = df[[case_col, act_col, ts_col]].sort_values([case_col, ts_col]).copy()
    sub[act_col] = sub[act_col].map(_clean).fillna(sub[act_col])

    traces: list[list[dict[str, str]]] = []
    for _, group in sub.groupby(case_col, sort=False):
        acts = group[act_col].tolist()
        trace: list[dict[str, str]] = (
            [{"concept:name": "▶"}] + [{"concept:name": a} for a in acts] + [{"concept:name": "■"}]
        )
        traces.append(trace)
    return traces


# ---------------------------------------------------------------------------
# DFG building — truth / training  (vectorised pandas)
# ---------------------------------------------------------------------------


def build_truth_dfg(df: pd.DataFrame) -> dict[tuple[str, str], int]:
    """
    Build a {(source, target): frequency} DFG from an event log DataFrame.

    Includes artificial ▶ / ■ start / end arcs.  Fully vectorised — no
    per-case Python loops.

    Args:
        df: event log (or sublog) DataFrame with columns
            case:concept:name, concept:name, time:timestamp

    Returns:
        DFG as dict[tuple[str, str], int]
    """
    case_col = "case:concept:name"
    act_col = "concept:name"
    ts_col = "time:timestamp"

    work = df[[case_col, act_col, ts_col]].sort_values([case_col, ts_col]).copy()
    work[act_col] = work[act_col].map(_clean).fillna(work[act_col])

    # --- consecutive-event (DF) arcs ---
    work["_next"] = work.groupby(case_col)[act_col].shift(-1)
    pairs = work.dropna(subset=["_next"])
    counts = pairs.groupby([act_col, "_next"]).size()
    dfg: dict[tuple[str, str], int] = {(src, tgt): int(cnt) for (src, tgt), cnt in counts.items()}

    # --- artificial start arcs: ▶ → first activity per case ---
    for act, cnt in work.groupby(case_col)[act_col].first().value_counts().items():
        key = ("▶", act)
        dfg[key] = dfg.get(key, 0) + int(cnt)

    # --- artificial end arcs: last activity per case → ■ ---
    for act, cnt in work.groupby(case_col)[act_col].last().value_counts().items():
        key = (act, "■")
        dfg[key] = dfg.get(key, 0) + int(cnt)

    return dfg


# ---------------------------------------------------------------------------
# DFG serialisation  (dict → JSON format consumed by AutomatonER)
# ---------------------------------------------------------------------------


def dfg_to_json(dfg: dict[tuple[str, str], int]) -> dict[str, Any]:
    """
    Serialise a {(source, target): freq} DFG to the JSON node/arc format.

    Node IDs: ▶ = 0, ■ = 1, other activities in encounter order.
    Node frequency = sum of incoming arc frequencies (or outgoing for ▶).
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


# ---------------------------------------------------------------------------
# Prediction DFG  (vectorised numpy)
# ---------------------------------------------------------------------------


def _parse_df_relation(col_name: str) -> tuple[str, str]:
    """Parse 'A -> B' column name → (cleaned_source, cleaned_target)."""
    src, tgt = col_name.split("->", 1)
    return _clean(src.strip()), _clean(tgt.strip())


def build_prediction_dfg(
    window_pred: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """
    Build a prediction DFG in JSON format from one window's numpy predictions.

    TSFMs predict DF-relation frequencies but not trace start / end
    probabilities.  Artificial ▶ → activity and activity → ■ arcs (freq = 1)
    are appended for every activity that has non-zero predicted frequency,
    matching the reference implementation.

    Args:
        window_pred:   shape (prediction_length, n_features) or (n_features,)
        feature_names: 'A -> B' column names aligned with the last axis

    Returns:
        DFG in {"nodes": [...], "arcs": [...]} format
    """
    freqs: np.ndarray = window_pred.sum(axis=0) if window_pred.ndim == 2 else window_pred
    freqs = np.clip(np.round(freqs), 0, None).astype(int)

    pairs = [_parse_df_relation(name) for name in feature_names]

    dfg: dict[tuple[str, str], int] = {}
    active: set[str] = set()

    for (src, tgt), freq in zip(pairs, freqs, strict=True):
        if freq > 0:
            dfg[(src, tgt)] = int(freq)
            active.add(src)
            active.add(tgt)

    # Artificial start / end connections
    for act in active:
        dfg[("▶", act)] = 1
        dfg[(act, "■")] = 1

    return dfg_to_json(dfg)
