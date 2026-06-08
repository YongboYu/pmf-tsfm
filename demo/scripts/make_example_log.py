"""One-off generator for the live-tab example log (not shipped to the Space).

Subsets the bundled sepsis log into a small `demo/examples/sepsis_sample.xes` so the
live tab has a one-click "Try an example" upload. The sample is a **dense contiguous
time-window slice** of the log: events are clipped to the densest `WINDOW_DAYS`-day
window so the daily DF-relation series is active right up to the end (a meaningful
last-known window + drift, not a sparse tail) while the file stays well under the
5 MB small-log gate.

Run from the demo/ dir:  uv run --with pm4py python scripts/make_example_log.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pm4py

SRC = Path(__file__).resolve().parents[2] / "data" / "raw_logs" / "sepsis.xes"
OUT = Path(__file__).resolve().parent.parent / "examples" / "sepsis_sample.xes"
WINDOW_DAYS = 42  # a six-week slice: dense enough for a rich forecast, small on disk

CASE_KEY = "case:concept:name"
TS_KEY = "time:timestamp"


def main() -> int:
    df = pm4py.read_xes(str(SRC))
    df[TS_KEY] = pd.to_datetime(df[TS_KEY], utc=True)

    # Find the densest WINDOW_DAYS window (most events) and clip to it, so the series
    # is active throughout instead of trailing off into a long sparse tail.
    day = df[TS_KEY].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    per_day = day.value_counts().sort_index()
    rolled = per_day.rolling(f"{WINDOW_DAYS}D").sum()
    end = rolled.idxmax()
    start = end - pd.Timedelta(days=WINDOW_DAYS - 1)
    window = (day >= start) & (day <= end)
    sub = df[window].copy()
    # Keep only cases that still have a directly-follows pair (>= 2 events) in-window.
    sizes = sub.groupby(CASE_KEY)["concept:name"].transform("size")
    sub = sub[sizes >= 2].copy()
    # Drop the heavy per-event diagnostic attributes the live path never reads — only
    # case id / activity / timestamp matter to log_to_df_series — so the file stays tiny.
    sub = sub[[CASE_KEY, "concept:name", TS_KEY]].copy()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pm4py.write_xes(sub, str(OUT))

    size_mb = OUT.stat().st_size / 1e6
    span_days = (sub[TS_KEY].max() - sub[TS_KEY].min()).days
    print(
        f"wrote {OUT} — {sub[CASE_KEY].nunique()} cases, {len(sub)} events, "
        f"{size_mb:.2f} MB, window {start.date()}..{end.date()} ({span_days} days)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
