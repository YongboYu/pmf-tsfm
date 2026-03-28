"""
Unit and integration tests for the Entropic Relevance (ER) evaluation pipeline.

Test organisation
-----------------
DFG building (dfg.py)
  - test_clean_activity                  symbol normalisation
  - test_build_truth_dfg_arcs            correct DF arcs + artificial ▶/■
  - test_dfg_to_json_ids                 ▶=0, ■=1 node IDs
  - test_dfg_to_json_roundtrip           freq totals preserved
  - test_build_prediction_dfg            sum-over-horizon, clip, artificial arcs
  - test_prediction_dfg_zero_freq        zero-frequency features excluded
  - test_extract_traces_markers          ▶ prepended, ■ appended
  - test_extract_traces_symbol_clean     ▶/■ in activities → Start/End
  - test_extract_sublog_vectorised       next_ts pre-computation works

Automaton / ER computation (automaton.py)
  - test_dfg_to_transitions_probs        transition probabilities correct
  - test_compute_er_perfect_fit_single   deterministic single-trace DFG → h₀ = 0
  - test_compute_er_binary_choice        P(A)=P(B)=½ → 1 bit per trace
  - test_compute_er_asymmetric           known closed-form ER value
  - test_compute_er_non_fitting          non-fitting traces increase ER
  - test_compute_er_empty_dfg            graceful nan for empty DFG
  - test_compute_er_empty_traces         graceful nan for empty traces
  - test_compute_er_fitting_ratio        fitting ratio computed correctly

Evaluate-ER pipeline helpers
  - test_get_test_dates                  correct window start/end from parquet
  - test_build_training_dfg_coverage     training cutoff ≈ 80 % of log
  - test_run_er_evaluation_mock          end-to-end pipeline with mocked XES
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pmf_tsfm.er.automaton import BackgroundModel, compute_er, dfg_to_transitions
from pmf_tsfm.er.dfg import (
    _clean,
    build_prediction_dfg,
    build_truth_dfg,
    dfg_to_json,
    extract_sublog,
    extract_traces,
    prepare_log,
)

# ---------------------------------------------------------------------------
# Helpers shared by multiple tests
# ---------------------------------------------------------------------------


def _make_log(records: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Build a tiny event log DataFrame from (case_id, activity, iso_timestamp) tuples."""
    case_ids, acts, ts = zip(*records, strict=False)
    return pd.DataFrame(
        {
            "case:concept:name": list(case_ids),
            "concept:name": list(acts),
            "time:timestamp": pd.to_datetime(list(ts), utc=True),
        }
    )


SIMPLE_LOG = [
    # case c1: A → B → C
    ("c1", "A", "2024-01-01T08:00:00+00:00"),
    ("c1", "B", "2024-01-01T09:00:00+00:00"),
    ("c1", "C", "2024-01-01T10:00:00+00:00"),
    # case c2: A → C (no B)
    ("c2", "A", "2024-01-02T08:00:00+00:00"),
    ("c2", "C", "2024-01-02T09:00:00+00:00"),
]


# ---------------------------------------------------------------------------
# DFG building tests
# ---------------------------------------------------------------------------


class TestCleanActivity:
    def test_start_symbol(self):
        assert _clean("▶") == "Start"

    def test_end_symbol(self):
        assert _clean("■") == "End"

    def test_regular_activity_unchanged(self):
        assert _clean("O_Create Offer") == "O_Create Offer"


class TestBuildTruthDFG:
    def test_df_arcs(self):
        df = _make_log(SIMPLE_LOG)
        dfg = build_truth_dfg(df)
        # c1: A→B, B→C   c2: A→C
        assert dfg[("A", "B")] == 1
        assert dfg[("B", "C")] == 1
        assert dfg[("A", "C")] == 1

    def test_artificial_start_arcs(self):
        df = _make_log(SIMPLE_LOG)
        dfg = build_truth_dfg(df)
        # Both cases start with A
        assert dfg[("▶", "A")] == 2

    def test_artificial_end_arcs(self):
        df = _make_log(SIMPLE_LOG)
        dfg = build_truth_dfg(df)
        # c1 ends C, c2 ends C
        assert dfg[("C", "■")] == 2

    def test_no_spurious_arcs(self):
        df = _make_log(SIMPLE_LOG)
        dfg = build_truth_dfg(df)
        # B is never the last activity
        assert ("B", "■") not in dfg

    def test_symbol_normalisation(self):
        """▶ and ■ appearing as activity labels are cleaned before arc creation."""
        log = _make_log(
            [
                ("c1", "▶", "2024-01-01T08:00:00+00:00"),
                ("c1", "A", "2024-01-01T09:00:00+00:00"),
            ]
        )
        dfg = build_truth_dfg(log)
        # ▶-as-activity is normalised to "Start"; artificial start is ▶
        assert ("Start", "A") in dfg
        assert ("▶", "Start") in dfg  # Start is the first activity of the case


class TestDFGToJSON:
    def test_start_node_id(self):
        dfg = {("▶", "A"): 2, ("A", "■"): 2}
        j = dfg_to_json(dfg)
        start_node = next(n for n in j["nodes"] if n["label"] == "▶")
        assert start_node["id"] == 0

    def test_end_node_id(self):
        dfg = {("▶", "A"): 2, ("A", "■"): 2}
        j = dfg_to_json(dfg)
        end_node = next(n for n in j["nodes"] if n["label"] == "■")
        assert end_node["id"] == 1

    def test_arc_count(self):
        dfg = {("▶", "A"): 2, ("A", "B"): 1, ("B", "■"): 1}
        j = dfg_to_json(dfg)
        assert len(j["arcs"]) == 3

    def test_arc_freqs_preserved(self):
        dfg = {("▶", "A"): 5, ("A", "■"): 5}
        j = dfg_to_json(dfg)
        start_arc = next(
            a
            for a in j["arcs"]
            if a["from"] == 0  # from ▶
        )
        assert start_arc["freq"] == 5

    def test_roundtrip_node_count(self):
        df = _make_log(SIMPLE_LOG)
        dfg = build_truth_dfg(df)
        j = dfg_to_json(dfg)
        # Nodes: ▶, ■, A, B, C  = 5
        assert len(j["nodes"]) == 5


class TestBuildPredictionDFG:
    FEATURES: ClassVar[list[str]] = ["A -> B", "A -> C", "B -> C"]

    def test_sum_over_horizon(self):
        # 3 days: day0=[1,0,0], day1=[1,0,0], day2=[0,0,0] → total A→B=2
        preds = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        j = build_prediction_dfg(preds, self.FEATURES)
        ab_arc = next(
            (
                a
                for a in j["arcs"]
                if self._label(j, a["from"]) == "A" and self._label(j, a["to"]) == "B"
            ),
            None,
        )
        assert ab_arc is not None
        assert ab_arc["freq"] == 2

    def test_clip_negative_to_zero(self):
        preds = np.array([[-5, 2, 0]], dtype=float)
        j = build_prediction_dfg(preds, self.FEATURES)
        # A→B had -5 after sum → 0 → should not appear as an arc
        ab_arc = next(
            (
                a
                for a in j["arcs"]
                if self._label(j, a["from"]) == "A" and self._label(j, a["to"]) == "B"
            ),
            None,
        )
        assert ab_arc is None

    def test_artificial_start_connections(self):
        preds = np.array([[1, 0, 0]], dtype=float)
        j = build_prediction_dfg(preds, self.FEATURES)
        # A is active (source of A→B). ▶→A must exist.
        start_arcs = [a for a in j["arcs"] if self._label(j, a["from"]) == "▶"]
        targets = {self._label(j, a["to"]) for a in start_arcs}
        assert "A" in targets
        assert "B" in targets  # B is a target in A→B

    def test_artificial_end_connections(self):
        preds = np.array([[1, 0, 0]], dtype=float)
        j = build_prediction_dfg(preds, self.FEATURES)
        end_arcs = [a for a in j["arcs"] if self._label(j, a["to"]) == "■"]
        sources = {self._label(j, a["from"]) for a in end_arcs}
        assert "A" in sources and "B" in sources

    def test_1d_input(self):
        """1-D array (already summed) should work."""
        preds = np.array([3.0, 0.0, 1.0])
        j = build_prediction_dfg(preds, self.FEATURES)
        assert len(j["arcs"]) > 0

    def test_symbol_cleaning_in_feature_names(self):
        """'▶ -> X' column name is cleaned to ('Start', 'X')."""
        preds = np.array([[5.0]])
        j = build_prediction_dfg(preds, ["▶ -> O_Create Offer"])
        # 'Start' should appear as an activity node
        node_labels = {n["label"] for n in j["nodes"]}
        assert "Start" in node_labels
        assert "O_Create Offer" in node_labels

    @staticmethod
    def _label(j: dict, node_id: int) -> str:
        return next(n["label"] for n in j["nodes"] if n["id"] == node_id)


class TestExtractTraces:
    def test_artificial_start(self):
        df = _make_log([("c1", "A", "2024-01-01T08:00:00+00:00")])
        traces = extract_traces(df)
        assert traces[0][0] == {"concept:name": "▶"}

    def test_artificial_end(self):
        df = _make_log([("c1", "A", "2024-01-01T08:00:00+00:00")])
        traces = extract_traces(df)
        assert traces[-1][-1] == {"concept:name": "■"}

    def test_trace_count(self):
        df = _make_log(SIMPLE_LOG)
        traces = extract_traces(df)
        assert len(traces) == 2  # c1, c2

    def test_activity_order(self):
        df = _make_log(
            [
                ("c1", "A", "2024-01-01T08:00:00+00:00"),
                ("c1", "B", "2024-01-01T09:00:00+00:00"),
            ]
        )
        traces = extract_traces(df)
        labels = [e["concept:name"] for e in traces[0]]
        assert labels == ["▶", "A", "B", "■"]

    def test_symbol_cleaning(self):
        """▶ / ■ appearing as actual activities in the log are cleaned."""
        df = _make_log(
            [
                ("c1", "▶", "2024-01-01T08:00:00+00:00"),
                ("c1", "A", "2024-01-01T09:00:00+00:00"),
                ("c1", "■", "2024-01-01T10:00:00+00:00"),
            ]
        )
        traces = extract_traces(df)
        inner = [e["concept:name"] for e in traces[0][1:-1]]  # exclude markers
        assert "Start" in inner
        assert "End" in inner
        assert "▶" not in inner


class TestExtractSublog:
    def test_events_in_window_included(self):
        df = _make_log(
            [
                ("c1", "A", "2024-01-05T08:00:00+00:00"),
                ("c1", "B", "2024-01-07T08:00:00+00:00"),
            ]
        )
        prepared = prepare_log(df)
        ws = pd.Timestamp("2024-01-05", tz="UTC")
        we = pd.Timestamp("2024-01-11 23:59:59.999999", tz="UTC")
        sublog = extract_sublog(prepared, ws, we)
        assert len(sublog) == 2

    def test_events_outside_window_excluded(self):
        df = _make_log(
            [
                ("c1", "A", "2024-01-01T08:00:00+00:00"),  # before
                ("c1", "B", "2024-01-20T08:00:00+00:00"),  # after
            ]
        )
        prepared = prepare_log(df)
        ws = pd.Timestamp("2024-01-05", tz="UTC")
        we = pd.Timestamp("2024-01-11 23:59:59.999999", tz="UTC")
        sublog = extract_sublog(prepared, ws, we)
        assert len(sublog) == 0

    def test_boundary_event_with_next_in_window_included(self):
        """An event just before the window is included if its successor is inside."""
        df = _make_log(
            [
                ("c1", "A", "2024-01-04T23:00:00+00:00"),  # outside window but next is inside
                ("c1", "B", "2024-01-05T08:00:00+00:00"),  # inside window
            ]
        )
        prepared = prepare_log(df)
        ws = pd.Timestamp("2024-01-05", tz="UTC")
        we = pd.Timestamp("2024-01-11 23:59:59.999999", tz="UTC")
        sublog = extract_sublog(prepared, ws, we)
        assert len(sublog) == 2  # both included to preserve the DF relation


# ---------------------------------------------------------------------------
# Automaton / ER computation tests
# ---------------------------------------------------------------------------


class TestDFGToTransitions:
    def _make_chain_dfg(self) -> dict:
        """Simple A→B chain with artificial start/end: ▶→A→B→■"""
        dfg = {("▶", "A"): 1, ("A", "B"): 1, ("B", "■"): 1}
        return dfg_to_json(dfg)

    def test_start_prob_always_one(self):
        j = self._make_chain_dfg()
        trans, _sources = dfg_to_transitions(j["nodes"], j["arcs"])
        # Find the start node (label="▶")
        start_id = next(n["id"] for n in j["nodes"] if n["label"] == "▶")
        start_transition = next(v for (f, _), v in trans.items() if f == start_id)
        _, log2_prob = start_transition
        assert log2_prob == pytest.approx(0.0)  # log2(1.0) = 0

    def test_end_prob_always_one(self):
        j = self._make_chain_dfg()
        trans, _ = dfg_to_transitions(j["nodes"], j["arcs"])
        # X → ■ transitions should have log2_prob = 0
        end_id = next(n["id"] for n in j["nodes"] if n["label"] == "■")
        end_transitions = [(t, p) for (_, lbl), (t, p) in trans.items() if t == end_id]
        for _, log2_prob in end_transitions:
            assert log2_prob == pytest.approx(0.0)

    def test_branch_probabilities_sum_to_one(self):
        """A→B (freq=3) and A→C (freq=1): P(B)=0.75, P(C)=0.25."""
        dfg = {("▶", "A"): 2, ("A", "B"): 3, ("A", "C"): 1, ("B", "■"): 3, ("C", "■"): 1}
        j = dfg_to_json(dfg)
        trans, _ = dfg_to_transitions(j["nodes"], j["arcs"])
        a_id = next(n["id"] for n in j["nodes"] if n["label"] == "A")
        # Transitions from A (excluding →■)
        from_a = {
            lbl: math.exp(p * math.log(2))
            for (f, lbl), (_, p) in trans.items()
            if f == a_id and lbl not in ("■",)
        }
        assert sum(from_a.values()) == pytest.approx(1.0, abs=1e-9)

    def test_sources_identified(self):
        j = self._make_chain_dfg()
        _, sources = dfg_to_transitions(j["nodes"], j["arcs"])
        # ▶ is the only source (no incoming arcs)
        start_id = next(n["id"] for n in j["nodes"] if n["label"] == "▶")
        assert sources == [start_id]


class TestComputeER:
    """Tests with known closed-form ER values."""

    @staticmethod
    def _single_trace_log() -> tuple[dict, list]:
        """DFG and log with one deterministic trace: A → B."""
        dfg = {("▶", "A"): 1, ("A", "B"): 1, ("B", "■"): 1}
        j = dfg_to_json(dfg)
        traces = [
            [
                {"concept:name": "▶"},
                {"concept:name": "A"},
                {"concept:name": "B"},
                {"concept:name": "■"},
            ]
        ]
        return j, traces

    def test_perfect_fit_single_trace(self):
        """One trace, fully deterministic DFG → cost_bits = 0, h₀ = 0 → ER = 0."""
        j, traces = self._single_trace_log()
        er, fit, n = compute_er(j, traces)
        assert er == pytest.approx(0.0, abs=1e-9)
        assert fit == pytest.approx(1.0)
        assert n == 1

    def test_binary_choice_er_one_bit(self):
        """
        DFG: A→B (freq=1) and A→C (freq=1), equal probability.
        Two traces: one [A,B] and one [A,C].
        Both fit. Cost per trace = 1 bit (log2(2)).  h₀ = 0 (all fitting).
        Expected ER = 1.0.
        """
        dfg = {("▶", "A"): 2, ("A", "B"): 1, ("A", "C"): 1, ("B", "■"): 1, ("C", "■"): 1}
        j = dfg_to_json(dfg)
        traces = [
            [
                {"concept:name": "▶"},
                {"concept:name": "A"},
                {"concept:name": "B"},
                {"concept:name": "■"},
            ],
            [
                {"concept:name": "▶"},
                {"concept:name": "A"},
                {"concept:name": "C"},
                {"concept:name": "■"},
            ],
        ]
        er, fit, n = compute_er(j, traces)
        assert er == pytest.approx(1.0, abs=1e-9)
        assert fit == pytest.approx(1.0)
        assert n == 2

    def test_asymmetric_choice_known_er(self):
        """
        DFG: A→B (freq=3), A→C (freq=1).  P(B)=0.75, P(C)=0.25.
        Four traces: three [A,B], one [A,C].  All fitting.
        Expected cost_bits = (3 * log2(1/0.75) + 1 * log2(1/0.25)) / 4
                          = (3 * 0.415 + 1 * 2.0) / 4 = 3.245/4 ≈ 0.8113
        h₀ = 0 (all fitting)
        ER ≈ 0.8113
        """
        dfg = {("▶", "A"): 4, ("A", "B"): 3, ("A", "C"): 1, ("B", "■"): 3, ("C", "■"): 1}
        j = dfg_to_json(dfg)
        traces = [
            [
                {"concept:name": "▶"},
                {"concept:name": "A"},
                {"concept:name": "B"},
                {"concept:name": "■"},
            ]
        ] * 3 + [
            [
                {"concept:name": "▶"},
                {"concept:name": "A"},
                {"concept:name": "C"},
                {"concept:name": "■"},
            ]
        ]
        expected = (3 * math.log2(1 / 0.75) + 1 * math.log2(1 / 0.25)) / 4
        er, fit, n = compute_er(j, traces)
        assert er == pytest.approx(expected, abs=1e-9)
        assert fit == pytest.approx(1.0)
        assert n == 4

    def test_non_fitting_traces_increase_er(self):
        """A trace with activity D not in DFG increases ER."""
        j, _ = self._single_trace_log()
        fitting_trace = [
            {"concept:name": "▶"},
            {"concept:name": "A"},
            {"concept:name": "B"},
            {"concept:name": "■"},
        ]
        non_fitting_trace = [{"concept:name": "▶"}, {"concept:name": "D"}, {"concept:name": "■"}]
        traces_all_fitting = [fitting_trace]
        traces_with_nonfit = [fitting_trace, non_fitting_trace]

        er_fit, _, _ = compute_er(j, traces_all_fitting)
        er_nf, _, _ = compute_er(j, traces_with_nonfit)
        assert er_nf > er_fit

    def test_fitting_ratio_all_fit(self):
        j, traces = self._single_trace_log()
        _, fit, _ = compute_er(j, traces)
        assert fit == pytest.approx(1.0)

    def test_fitting_ratio_none_fit(self):
        """All traces are non-fitting."""
        dfg = {("▶", "A"): 1, ("A", "■"): 1}
        j = dfg_to_json(dfg)
        # Only trace B→C, which is not in the DFG
        traces = [[{"concept:name": "▶"}, {"concept:name": "B"}, {"concept:name": "■"}]]
        _, fit, n = compute_er(j, traces)
        assert fit == pytest.approx(0.0)
        assert n == 1

    def test_empty_dfg_returns_nan(self):
        er, _fit, _n = compute_er({"nodes": [], "arcs": []}, [])
        assert math.isnan(er)

    def test_empty_traces_returns_nan(self):
        dfg = {("▶", "A"): 1, ("A", "■"): 1}
        j = dfg_to_json(dfg)
        er, _fit, n = compute_er(j, [])
        assert math.isnan(er)
        assert n == 0

    def test_er_is_nan_safe(self):
        """compute_er must not raise on empty inputs."""
        assert not math.isnan(math.inf)  # sanity
        er, _, _ = compute_er({}, [])
        assert math.isnan(er)


class TestBackgroundModel:
    """Unit tests for BackgroundModel accumulator."""

    def test_single_fitting_trace_zero_entropy(self):
        """One fully deterministic trace (prob=1 at every step) → ER = 0."""
        model = BackgroundModel()
        model.open_trace()
        model.process_event("A", math.log2(1.0))  # log2_prob = 0
        model.process_event("■", math.log2(1.0))
        model.close_trace(2, fitting=True)
        assert model.compute_relevance() == pytest.approx(0.0)

    def test_non_fitting_trace_adds_cost(self):
        """Non-fitting trace uses fallback cost (1 + len) * log2(1 + |labels|)."""
        model = BackgroundModel()
        model.open_trace()
        model.process_event("X", 0.0)
        model.process_event("■", 0.0)
        model.close_trace(2, fitting=False)
        er = model.compute_relevance()
        # labels = {"X", "■"}, expected cost = (1+2)*log2(1+2) = 3*log2(3)
        expected_cost = (1 + 2) * math.log2(1 + 2)
        assert er == pytest.approx(expected_cost, abs=1e-9)


# ---------------------------------------------------------------------------
# Evaluate-ER pipeline helpers
# ---------------------------------------------------------------------------


class TestGetTestDates:
    def test_window_count(self, tmp_path: Path):
        """n_windows = len(idx) - val_end - pred_len + 1."""
        idx = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
        df = pd.DataFrame({"x": range(20)}, index=idx)
        parquet = tmp_path / "ts.parquet"
        df.to_parquet(parquet)

        from pmf_tsfm.er.evaluate_er import _get_test_dates

        starts, ends = _get_test_dates(parquet, val_end=13, prediction_length=7)
        # n_windows = 20 - 13 - 7 + 1 = 1
        assert len(starts) == 1
        assert len(ends) == 1

    def test_start_date_correct(self, tmp_path: Path):
        idx = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
        df = pd.DataFrame({"x": range(20)}, index=idx)
        parquet = tmp_path / "ts.parquet"
        df.to_parquet(parquet)

        from pmf_tsfm.er.evaluate_er import _get_test_dates

        starts, _ends = _get_test_dates(parquet, val_end=10, prediction_length=7)
        assert starts[0] == idx[10]

    def test_end_date_correct(self, tmp_path: Path):
        idx = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
        df = pd.DataFrame({"x": range(20)}, index=idx)
        parquet = tmp_path / "ts.parquet"
        df.to_parquet(parquet)

        from pmf_tsfm.er.evaluate_er import _get_test_dates

        _starts, ends = _get_test_dates(parquet, val_end=10, prediction_length=7)
        # end = idx[10 + 7 - 1] = idx[16]
        assert ends[0] == idx[16]


class TestBuildTrainingDFG:
    def test_training_cutoff_at_80_percent(self):
        """DFG is built from the first 80 % of the log by time."""
        from pmf_tsfm.er.evaluate_er import _training_cutoff

        # 10-day log: cutoff should be at day 8 (80 % of 10)
        t0 = pd.Timestamp("2024-01-01", tz="UTC")
        records = [("c1", "A", (t0 + pd.Timedelta(days=i)).isoformat()) for i in range(10)]
        df = _make_log(records)
        cutoff = _training_cutoff(df)
        assert cutoff == t0 + pd.Timedelta(days=8)

    def test_training_dfg_excludes_late_events(self):
        """Events after the training cutoff are not in the training DFG."""
        from pmf_tsfm.er.evaluate_er import _build_training_dfg

        t0 = pd.Timestamp("2024-01-01", tz="UTC")
        # c1: A→B on day 1-2 (inside training)
        # c2: X→Y on day 9-10 (outside training if 80% cutoff = day 8)
        records = [
            ("c1", "A", (t0 + pd.Timedelta(days=1)).isoformat()),
            ("c1", "B", (t0 + pd.Timedelta(days=2)).isoformat()),
            ("c2", "X", (t0 + pd.Timedelta(days=9)).isoformat()),
            ("c2", "Y", (t0 + pd.Timedelta(days=10)).isoformat()),
        ]
        df = _make_log(records)
        j = _build_training_dfg(df)
        arc_labels = {
            (n_from["label"], n_to["label"])
            for arc in j["arcs"]
            for n_from in j["nodes"]
            if n_from["id"] == arc["from"]
            for n_to in j["nodes"]
            if n_to["id"] == arc["to"]
        }
        assert ("A", "B") in arc_labels
        assert ("X", "Y") not in arc_labels


class TestRunERPipelineMock:
    """
    End-to-end integration test using a small in-memory event log and
    synthetic numpy predictions.  pm4py is bypassed by patching load_event_log.
    """

    def _make_parquet(self, tmp_path: Path, n: int = 20) -> Path:
        idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
        df = pd.DataFrame({"A -> B": range(n)}, index=idx)
        p = tmp_path / "ts.parquet"
        df.to_parquet(p)
        return p

    def _make_xes_df(self) -> pd.DataFrame:
        """Tiny in-memory event log covering 2020-01-14 to 2020-01-21 (test window)."""
        t0 = pd.Timestamp("2020-01-14", tz="UTC")
        records = []
        for day in range(7):
            for case in range(3):
                records.append((f"c{case}", "A", t0 + pd.Timedelta(days=day, hours=8)))
                records.append((f"c{case}", "B", t0 + pd.Timedelta(days=day, hours=9)))
        return pd.DataFrame(
            {
                "case:concept:name": [r[0] for r in records],
                "concept:name": [r[1] for r in records],
                "time:timestamp": [r[2] for r in records],
            }
        )

    def _make_predictions(self, tmp_path: Path, n_windows: int = 7) -> tuple[Path, Path]:
        preds = np.ones((n_windows, 7, 1)) * 3.0  # (windows, horizon, features)
        feature_names = ["A -> B"]

        pred_path = tmp_path / "BPI2017_chronos_2_predictions.npy"
        meta_path = tmp_path / "BPI2017_chronos_2_metadata.json"
        np.save(pred_path, preds)
        meta_path.write_text(json.dumps({"feature_names": feature_names}))
        return pred_path, meta_path

    def test_pipeline_runs_and_produces_valid_er(self, tmp_path: Path):
        """Full pipeline with mocked XES and synthetic predictions."""
        from omegaconf import OmegaConf

        from pmf_tsfm.er.evaluate_er import run_er_evaluation

        parquet = self._make_parquet(tmp_path, n=21)
        xes_df = self._make_xes_df()

        # parquet has 21 rows, val_end=13, pred_len=7 → n_windows = 21-13-7+1 = 2
        n_windows = 21 - 13 - 7 + 1
        out_dir = tmp_path / "outputs" / "zero_shot" / "BPI2017" / "chronos_2"
        out_dir.mkdir(parents=True)
        preds = np.ones((n_windows, 7, 1)) * 3.0
        np.save(out_dir / "BPI2017_chronos_2_predictions.npy", preds)
        (out_dir / "BPI2017_chronos_2_metadata.json").write_text(
            json.dumps({"feature_names": ["A -> B"]})
        )

        cfg = OmegaConf.create(
            {
                "data": {"name": "BPI2017", "path": str(parquet), "val_end": 13},
                "model": {"name": "chronos_2"},
                "task": "zero_shot",
                "prediction_length": 7,
                "train_ratio": 0.8,
                "save": False,
                "output_dir": str(tmp_path / "er_out"),
                "paths": {"output_dir": str(tmp_path / "outputs"), "log_dir": str(tmp_path)},
            }
        )

        with patch("pmf_tsfm.er.evaluate_er.load_event_log", return_value=xes_df):
            result = run_er_evaluation(cfg)

        assert "summary" in result
        assert "windows" in result
        summary = result["summary"]

        # n_windows = 21 - 13 - 7 + 1 = 2
        assert summary["n_windows"] == 2
        assert summary["n_valid_windows"] >= 0
        # ER is valid when the test log covers the window dates
        assert isinstance(summary["pred_er_mean"], float)

    def test_pipeline_saves_json(self, tmp_path: Path):
        """When save=True, a JSON file is written."""
        from omegaconf import OmegaConf

        from pmf_tsfm.er.evaluate_er import run_er_evaluation

        parquet = self._make_parquet(tmp_path, n=21)
        xes_df = self._make_xes_df()

        out_dir = tmp_path / "outputs" / "zero_shot" / "BPI2017" / "chronos_2"
        out_dir.mkdir(parents=True)
        preds = np.ones((7, 7, 1)) * 3.0
        np.save(out_dir / "BPI2017_chronos_2_predictions.npy", preds)
        (out_dir / "BPI2017_chronos_2_metadata.json").write_text(
            json.dumps({"feature_names": ["A -> B"]})
        )

        # parquet has 21 rows, val_end=13, pred_len=7 → n_windows = 21-13-7+1 = 2
        n_windows = 21 - 13 - 7 + 1
        out_dir = tmp_path / "outputs2" / "zero_shot" / "BPI2017" / "chronos_2"
        out_dir.mkdir(parents=True)
        preds = np.ones((n_windows, 7, 1)) * 3.0
        np.save(out_dir / "BPI2017_chronos_2_predictions.npy", preds)
        (out_dir / "BPI2017_chronos_2_metadata.json").write_text(
            json.dumps({"feature_names": ["A -> B"]})
        )

        er_out = tmp_path / "er_out"
        cfg = OmegaConf.create(
            {
                "data": {"name": "BPI2017", "path": str(parquet), "val_end": 13},
                "model": {"name": "chronos_2"},
                "task": "zero_shot",
                "prediction_length": 7,
                "train_ratio": 0.8,
                "save": True,
                "output_dir": str(er_out),
                "paths": {"output_dir": str(tmp_path / "outputs2"), "log_dir": str(tmp_path)},
            }
        )

        with patch("pmf_tsfm.er.evaluate_er.load_event_log", return_value=xes_df):
            run_er_evaluation(cfg)

        saved = list(er_out.glob("*_er.json"))
        assert len(saved) == 1
        data = json.loads(saved[0].read_text())
        assert "summary" in data
        assert "windows" in data
