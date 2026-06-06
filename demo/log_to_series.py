"""Convert an uploaded XES log into the daily DF-relation frequency series.

A faithful in-repo port of the external `pmf-benchmark`_ preprocessing
(``preprocessing/{event_log_processor,df_generator,time_series_creator}.py``) that
produced this repo's ``data/time_series/*.parquet``. h5↔parquet verification (#115)
confirmed that pipeline is the producer of the repo's series, so this port
reproduces the same representation the demo's TSFMs were applied to.

The pipeline, per upstream:

1. read the XES log (pm4py),
2. ``filter_variants_by_coverage_percentage`` — drop ultra-rare variants (mild; a
   no-op on normal logs),
3. ``insert_artificial_start_end`` — add the ▶ / ■ events (the source of the
   ``▶ -> X`` / ``X -> ■`` columns),
4. extract every consecutive-event DF pair ``"a -> b"`` recording the **source
   event's** timestamp,
5. lay the counts on a daily ``date_range`` over the log's span, bucketing each pair
   by its **source-event day**.

**Deliberate deviation from upstream:** upstream also trims 10% of the timespan off
*each* end before step 4. This converter **omits the end-trim** — the live path
forecasts the genuine future *from the log end* (ADR-0004), so trimming the recent
end would discard exactly the window the forecast is anchored on. Fidelity to the
training preprocessing is not a correctness gate on the upload path (it reports
**drift**, not accuracy); what matters is that the forecast and last-known windows
share one feature space, which they do — both are columns of this one frame.

Gradio-free and ``pmf_tsfm``-free (only ``pm4py`` / ``pandas`` / ``numpy``), so it
runs in the lean live-path serve env.

.. _pmf-benchmark: https://github.com/YongboYu/pmf-benchmark
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CASE_KEY = "case:concept:name"
ACT_KEY = "concept:name"
TS_KEY = "time:timestamp"

# Match pmf-benchmark's preprocessing_config.yaml (event_log.filter_percentage).
DEFAULT_FILTER_COVERAGE = 0.0001


def log_to_df_series(
    log: str | Path, *, filter_coverage: float = DEFAULT_FILTER_COVERAGE
) -> pd.DataFrame:
    """Read an XES log and return its daily DF-relation frequency series.

    Args:
        log:             Path to the uploaded XES event log.
        filter_coverage: Variant-coverage threshold for the rare-variant filter
                         (upstream default ``0.0001``); pass ``0`` to skip filtering.

    Returns:
        A ``(days × DF-relations)`` frame: a daily ``DatetimeIndex`` over the log's
        full span and one ``"a -> b"`` column per directly-follows relation (with the
        artificial ▶ / ■ markers), each cell the count of that relation whose source
        event fell on that day.
    """
    import pm4py

    df = pm4py.read_xes(str(log))
    return event_df_to_series(df, filter_coverage=filter_coverage)


def event_df_to_series(
    df: pd.DataFrame, *, filter_coverage: float = DEFAULT_FILTER_COVERAGE
) -> pd.DataFrame:
    """Daily DF-relation series from an already-loaded event-log DataFrame.

    Split out from :func:`log_to_df_series` so the assembly is testable without an
    XES round-trip. Expects the standard XES columns (``case:concept:name``,
    ``concept:name``, ``time:timestamp``).
    """
    import pm4py

    df = pm4py.format_dataframe(df, case_id=CASE_KEY, activity_key=ACT_KEY, timestamp_key=TS_KEY)
    if filter_coverage:
        df = pm4py.filter_variants_by_coverage_percentage(df, filter_coverage)
    # Artificial ▶ / ■ events (timestamps ±1ms around each case's first/last event), so
    # the ``▶ -> first`` / ``last -> ■`` relations exist and bucket on the right day.
    df = pm4py.insert_artificial_start_end(df)

    work = df[[CASE_KEY, ACT_KEY, TS_KEY]].copy()
    work[TS_KEY] = pd.to_datetime(work[TS_KEY], utc=True)
    work = work.sort_values([CASE_KEY, TS_KEY])

    # Each event's directly-following successor within its case; the last event per
    # case has none (NaN) and contributes no outgoing relation.
    work["_next"] = work.groupby(CASE_KEY, sort=False)[ACT_KEY].shift(-1)
    pairs = work.dropna(subset=["_next"]).copy()
    pairs["_relation"] = pairs[ACT_KEY].astype(str) + " -> " + pairs["_next"].astype(str)
    # Bucket by the SOURCE event's calendar day (upstream uses ``start_time``).
    pairs["_day"] = pairs[TS_KEY].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()

    counts = pairs.groupby(["_day", "_relation"]).size().unstack(fill_value=0)

    # Lay the counts on a contiguous daily index over the full (untrimmed) log span —
    # the artificial events keep ▶/■ on the adjacent real-event days, so this spans
    # exactly the log's active days.
    span = work[TS_KEY].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    days = pd.date_range(span.min(), span.max(), freq="D")
    series = counts.reindex(days, fill_value=0).astype(int)
    series.index.name = "date"
    series.columns.name = None
    return series
