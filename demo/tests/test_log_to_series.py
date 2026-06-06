"""Tests for the live-path log→daily-DF-series converter (slice 3b, #115).

``log_to_series`` is the in-repo, faithful port of the external ``pmf-benchmark``
preprocessing (``preprocessing/{event_log_processor,df_generator,time_series_creator}``)
that produced this repo's ``data/time_series/*.parquet``. It turns an uploaded XES
log into the daily ``(days × DF-relations)`` frequency frame the TSFM forecasts.

The port replicates the pipeline that h5↔parquet verification confirmed is the
producer of the repo's series — variant filter, artificial ▶/■ events, consecutive
DF pairs bucketed by the **source-event day** — but **drops the 10% end-trim**: the
live path forecasts the genuine future *from the log end* (ADR-0004), so trimming
the recent end would be wrong. Fidelity is not an accuracy gate here (the upload
path reports drift, not accuracy); a single shared feature space is what matters.
"""

from __future__ import annotations

import pandas as pd
import pytest
from log_to_series import event_df_to_series, log_to_df_series

CASE, ACT, TS = "case:concept:name", "concept:name", "time:timestamp"


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


def test_index_is_contiguous_daily_over_the_full_span(series):
    """The index is one row per calendar day across the whole (untrimmed) log span."""
    assert list(series.index) == list(pd.date_range("2020-01-01", "2020-01-03", freq="D"))
    # contiguous daily — no gaps, no trimmed ends.
    assert (series.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


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
    """C->■ (c1's close) buckets on C's day (01-02); the live path keeps that final day."""
    assert series.at[pd.Timestamp("2020-01-02"), "C -> ■"] == 1


def test_log_to_df_series_reads_xes(tmp_path):
    """The public entry reads an XES file and yields the same frame as the in-memory path."""
    import pm4py

    df = _event_df(_FIXTURE)
    xes = tmp_path / "tiny.xes"
    pm4py.write_xes(df, str(xes), case_id_key=CASE)

    from_file = log_to_df_series(xes)
    in_memory = event_df_to_series(df)
    # Same relations and same per-day counts (column/row order aside).
    a = from_file.reindex(sorted(from_file.columns), axis=1).sort_index()
    b = in_memory.reindex(sorted(in_memory.columns), axis=1).sort_index()
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
