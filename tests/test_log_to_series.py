"""Tests for the core XES→daily-DF-series converter (issue #132).

``pmf_tsfm.data.log_to_series`` is the in-repo, faithful port of the external
``pmf-benchmark`` preprocessing that produced this repo's ``data/time_series/*.parquet``.
It turns a raw XES log into the daily ``(days × DF-relations)`` frequency frame the TSFM
forecasts, and ``prepare_series_from_log`` packages it (parquet + split indices + staged
log) for ``pmf_tsfm.api``.

This file mirrors the demo's converter tests against the core copy, adds the
``prepare_series_from_log`` packaging contract, a parity cross-check against the committed
``bpi2017`` series, and a **drift guard** keeping the core and demo converters identical
(the demo keeps its own ``pmf_tsfm``-free copy for its lean serve env — two copies are
unavoidable, so we pin them together).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest

from pmf_tsfm.data.log_to_series import (
    event_df_to_series,
    log_to_df_series,
    prepare_series_from_log,
)

CASE, ACT, TS = "case:concept:name", "concept:name", "time:timestamp"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BPI2017_XES = _REPO_ROOT / "data" / "processed_logs" / "bpi2017.xes"
_BPI2017_PARQUET = _REPO_ROOT / "data" / "time_series" / "bpi2017.parquet"


def _event_df(rows):
    """Build a tidy event-log DataFrame from ``(case, activity, "YYYY-MM-DD HH:MM")`` rows."""
    df = pd.DataFrame(rows, columns=[CASE, ACT, TS])
    df[TS] = pd.to_datetime(df[TS], utc=True)
    return df


# A 2-case log: c1 spans 01-01..01-02, c2 is all on 01-03. Hand-traced expectation
# (bucket every DF pair by its SOURCE event's day):
#   01-01: ▶->A, A->B, B->C        01-02: C->■        01-03: ▶->A, A->B, B->■
_FIXTURE = [
    ("c1", "A", "2020-01-01 09:00"),
    ("c1", "B", "2020-01-01 10:00"),
    ("c1", "C", "2020-01-02 09:00"),
    ("c2", "A", "2020-01-03 09:00"),
    ("c2", "B", "2020-01-03 11:00"),
]


@pytest.fixture
def series():
    return event_df_to_series(_event_df(_FIXTURE))


# ---------------------------------------------------------------------------
# Converter semantics (mirrors demo/tests/test_log_to_series.py)
# ---------------------------------------------------------------------------


def test_index_is_contiguous_daily_over_the_full_span(series):
    """The index is one row per calendar day across the whole (untrimmed) log span."""
    assert list(series.index) == list(pd.date_range("2020-01-01", "2020-01-03", freq="D"))
    assert (series.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


def test_index_is_a_named_datetime_index(series):
    """ER reads test-window dates off this index, so it must be a DatetimeIndex named 'date'."""
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.index.name == "date"


def test_columns_are_df_relations_with_artificial_markers(series):
    """Columns are ``"a -> b"`` DF relations, including the artificial ▶ / ■ markers."""
    cols = set(series.columns)
    assert {"▶ -> A", "A -> B", "B -> C", "C -> ■", "B -> ■"} == cols
    assert all(" -> " in c for c in cols)  # the exact separator dfg_to_json splits on


def test_relations_bucket_on_the_source_event_day(series):
    """Each DF pair is counted on the day of its *source* event (pmf-benchmark semantics)."""
    assert series.at[pd.Timestamp("2020-01-01"), "A -> B"] == 1
    assert series.at[pd.Timestamp("2020-01-03"), "A -> B"] == 1
    assert series.at[pd.Timestamp("2020-01-02"), "A -> B"] == 0


def test_cross_day_relation_counts_on_the_source_not_target_day(series):
    """B(01-01)->C(01-02) is a cross-day pair; it counts on 01-01 (source), not 01-02."""
    assert series.at[pd.Timestamp("2020-01-01"), "B -> C"] == 1
    assert series.at[pd.Timestamp("2020-01-02"), "B -> C"] == 0


def test_end_event_relation_counts_on_its_source_day(series):
    """C->■ (c1's close) buckets on C's day (01-02); the path keeps that final day."""
    assert series.at[pd.Timestamp("2020-01-02"), "C -> ■"] == 1


def test_log_to_df_series_reads_xes(tmp_path):
    """The public entry reads an XES file and yields the same frame as the in-memory path."""
    import pm4py

    df = _event_df(_FIXTURE)
    xes = tmp_path / "tiny.xes"
    pm4py.write_xes(df, str(xes), case_id_key=CASE)

    from_file = log_to_df_series(xes)
    in_memory = event_df_to_series(df)
    a = from_file.reindex(sorted(from_file.columns), axis=1).sort_index()
    b = in_memory.reindex(sorted(in_memory.columns), axis=1).sort_index()
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


# ---------------------------------------------------------------------------
# prepare_series_from_log packaging contract
# ---------------------------------------------------------------------------


def test_prepare_series_from_log_writes_datetime_parquet_and_derives_splits(tmp_path):
    """Writes a datetime-indexed parquet, derives 60/20/20 splits, and stages the XES."""
    import pm4py

    # 15 consecutive daily A→B cases → a 15-row daily series.
    rows = []
    for d in range(15):
        day = f"2021-03-{d + 1:02d}"
        rows += [(f"c{d}", "A", f"{day} 09:00"), (f"c{d}", "B", f"{day} 10:00")]
    xes = tmp_path / "mylog.xes"
    pm4py.write_xes(_event_df(rows), str(xes), case_id_key=CASE)

    out_parquet = tmp_path / "out" / "series.parquet"
    log_dir = tmp_path / "logs"
    info = prepare_series_from_log(xes, out_parquet, log_dir=log_dir)

    assert info["name"] == "mylog"
    assert info["path"] == str(out_parquet)
    # 15 rows → floor(15*0.6)=9, floor(15*0.8)=12.
    assert (info["train_end"], info["val_end"]) == (9, 12)

    # Parquet exists with a retained DatetimeIndex.
    df = pd.read_parquet(out_parquet)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 15

    # XES staged for ER at {log_dir}/{name}.xes.
    assert (log_dir / "mylog.xes").exists()


def test_prepare_series_from_log_strips_xes_gz_suffix(tmp_path):
    """``name`` strips ``.xes.gz`` (not just ``.xes``)."""
    import pm4py

    xes = tmp_path / "compressed.xes.gz"
    pm4py.write_xes(_event_df(_FIXTURE), str(xes), case_id_key=CASE)
    info = prepare_series_from_log(xes, tmp_path / "s.parquet")
    assert info["name"] == "compressed"


# ---------------------------------------------------------------------------
# Parity vs the committed bpi2017 series (slow; needs the 135 MB local XES)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not (_BPI2017_XES.exists() and _BPI2017_PARQUET.exists()),
    reason="needs local data/processed_logs/bpi2017.xes (not in CI)",
)
def test_parity_with_committed_bpi2017_series():
    """The converter faithfully reproduces the committed bpi2017 series.

    This is a *faithful port* of the external pmf-benchmark pipeline, not a byte-exact
    re-run of it (the demo docstring is explicit that fidelity is not a correctness gate).
    Two known, benign sources of divergence on the shared dates:

      * Variant filter: upstream filtered harder (21 relations) than our ``0.0001``
        threshold (28) — so the committed columns are a strict *subset* of ours, and a
        few counts shift where a dropped rare variant fed a kept relation.
      * pm4py version drift since the committed parquet was generated externally.

    So we assert the load-bearing properties — same date span, committed feature space
    ⊆ ours, and high numerical agreement on the shared cells — not literal equality.
    """
    produced = log_to_df_series(_BPI2017_XES)
    committed = pd.read_parquet(_BPI2017_PARQUET)

    # Normalise both indices to tz-naive daily timestamps for alignment.
    produced.index = pd.to_datetime(produced.index).tz_localize(None).normalize()
    cidx = pd.to_datetime(committed.index)
    committed.index = (
        cidx.tz_convert("UTC").tz_localize(None) if cidx.tz is not None else cidx.tz_localize(None)
    ).normalize()

    # Same date span, and every relation the committed series tracks is also produced.
    assert committed.index.isin(produced.index).all()
    assert committed.columns.isin(produced.columns).all()

    common_dates = produced.index.intersection(committed.index)
    common_cols = committed.columns
    a = produced.loc[common_dates, common_cols].sort_index(axis=1)
    b = committed.loc[common_dates, common_cols].sort_index(axis=1)

    exact_fraction = (a == b).to_numpy().mean()
    rel_mass_diff = (a - b).abs().to_numpy().sum() / b.to_numpy().sum()
    assert exact_fraction >= 0.95, f"only {exact_fraction:.3f} of cells match exactly"
    assert rel_mass_diff < 0.02, f"relative count divergence {rel_mass_diff:.4f} too high"


# ---------------------------------------------------------------------------
# Drift guard: the core and demo converters must stay byte-identical
# ---------------------------------------------------------------------------


def test_core_and_demo_converters_are_identical():
    """The demo keeps a standalone ``pmf_tsfm``-free copy; pin the two converter bodies.

    If this fails, a change landed in one copy but not the other — sync them.
    """
    demo_dir = str(_REPO_ROOT / "demo")
    if demo_dir not in sys.path:
        sys.path.insert(0, demo_dir)
    import log_to_series as demo

    from pmf_tsfm.data import log_to_series as core

    for fn in ("event_df_to_series", "log_to_df_series"):
        assert inspect.getsource(getattr(core, fn)) == inspect.getsource(getattr(demo, fn)), (
            f"{fn} diverged between core and demo copies"
        )
