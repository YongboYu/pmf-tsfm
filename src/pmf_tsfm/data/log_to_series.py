"""Convert an XES event log into the daily DF-relation frequency series.

A faithful in-repo port of the external `pmf-benchmark`_ preprocessing
(``preprocessing/{event_log_processor,df_generator,time_series_creator}.py``) that
produced this repo's ``data/time_series/*.parquet``. h5↔parquet verification (#115)
confirmed that pipeline is the producer of the repo's series, so this port
reproduces the same representation the paper's TSFMs were applied to.

This is the **core** copy (issue #132): the shared seam under ``pmf_tsfm.api`` lifts
the same converter the demo's live path uses (``demo/log_to_series.py``) so users can
bring a raw ``.xes`` log. The demo keeps its own standalone copy — it is deliberately
``pmf_tsfm``-free for its lean serve env, so it cannot import this one; a drift guard
(``tests/test_log_to_series.py``) keeps the two converter bodies byte-identical.

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
*each* end before step 4. This converter **omits the end-trim** — the backtest path
forecasts the genuine future *from the log end* (ADR-0004), so trimming the recent
end would discard exactly the window the forecast is anchored on. Fidelity to the
training preprocessing is not a correctness gate here; what matters is that the
forecast and last-known windows share one feature space, which they do — both are
columns of this one frame.

Heavy deps are lazy: ``pm4py`` is imported inside the functions, so importing this
module (and ``pmf_tsfm.api``) stays cheap.

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


def _log_name(xes_path: str | Path) -> str:
    """Dataset name from an XES path stem, stripping ``.xes`` / ``.xes.gz``."""
    name = Path(xes_path).name
    for suffix in (".xes.gz", ".xes"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(xes_path).stem


def prepare_series_from_log(
    xes_path: str | Path,
    out_parquet: str | Path,
    *,
    split_ratio: tuple[float, float, float] = (0.6, 0.2, 0.2),
    log_dir: str | Path | None = None,
) -> dict:
    """Build the daily DF-relation parquet a backtest run needs from a raw XES log.

    Writes the ``(days × DF-relations)`` frame to ``out_parquet`` **with its
    ``DatetimeIndex`` retained** — the Entropic Relevance evaluator reads the test-window
    dates straight off this index (``er/evaluate_er.py:_get_test_dates``), so the index
    must stay datetime. When ``log_dir`` is given, the source ``.xes`` is copied to
    ``{log_dir}/{name}.xes`` because ER reads the log at ``cfg.paths.log_dir/<name>.xes``.

    Args:
        xes_path:    Path to the raw XES event log.
        out_parquet: Where to write the daily DF-relation parquet.
        split_ratio: ``(train, val, test)`` fractions; absolute split indices are
                     derived from the row count by flooring, matching the committed
                     dataset configs (e.g. bpi2017 319 → 191/255, sepsis 459 → 275/367).
        log_dir:     If given, copy the XES there as ``{name}.xes`` for ER.

    Returns:
        ``{"name", "path", "train_end", "val_end"}`` — the data config the API sets in
        code for ``ZeroShotDataModule.from_config`` (it reads exactly these keys).
    """
    import shutil

    name = _log_name(xes_path)
    series = log_to_df_series(xes_path)

    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    series.to_parquet(out_parquet)  # DatetimeIndex retained

    n = len(series)
    train_end = int(n * split_ratio[0])
    val_end = int(n * (split_ratio[0] + split_ratio[1]))

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(xes_path, log_dir / f"{name}.xes")

    return {
        "name": name,
        "path": str(out_parquet),
        "train_end": train_end,
        "val_end": val_end,
    }
