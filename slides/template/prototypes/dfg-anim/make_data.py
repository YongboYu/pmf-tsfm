"""PROTOTYPE — wipe me when slide 3 is finalised.

Bake the shared DFG-evolution frame data that all four pipeline prototypes
(SVG / matplotlib gif / Manim mp4 / cytoscape) consume, so every asset shows
*identical* real numbers and only the rendering differs.

Story: bpi2017 "offer" subgraph. Three observed weeks (t1..t3) then a forecast
week (t4). Every edge is stable except the hero edge `Sent -> Cancelled`, whose
weekly frequency collapses 281 -> 262 -> 88 (an edge-specific behavioural drift,
the same O_Sent->O_Cancelled drift used on slides 2 and 4). The forecast frame is
an illustrative naive-persistence forecast (= last observed week); the held-out
ground truth for that week is kept so the asset can ghost it behind the
accent-blue prediction.

Run:  uv run python slides/template/prototypes/dfg-anim/make_data.py
"""

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
PARQUET = HERE.parents[3] / "data" / "time_series" / "bpi2017.parquet"

# parquet column  ->  (edge id, source node, target node)
EDGE_COLS = {
    "▶ -> O_Create Offer": ("start__create", "start", "create"),
    "O_Create Offer -> O_Created": ("create__created", "create", "created"),
    "O_Created -> O_Sent (mail and online)": ("created__sent", "created", "sent"),
    "O_Sent (mail and online) -> O_Cancelled": ("sent__cancelled", "sent", "cancelled"),
    "O_Sent (mail and online) -> O_Returned": ("sent__returned", "sent", "returned"),
    "O_Returned -> O_Accepted": ("returned__accepted", "returned", "accepted"),
    "O_Accepted -> ■": ("accepted__end", "accepted", "end"),
    "O_Cancelled -> ■": ("cancelled__end", "cancelled", "end"),
}
HERO = "sent__cancelled"

# node layout on a 0..100 canvas (y grows downward). A vertical spine
# (start -> create -> created -> sent -> returned -> accepted) with the
# cancellation path branching out to the right.
NODES = [
    {"id": "start", "label": "▶", "kind": "start", "x": 34, "y": 7},
    {"id": "create", "label": "Create Offer", "kind": "activity", "x": 34, "y": 22},
    {"id": "created", "label": "Created", "kind": "activity", "x": 34, "y": 37},
    {"id": "sent", "label": "Sent", "kind": "activity", "x": 34, "y": 53},
    {"id": "returned", "label": "Returned", "kind": "activity", "x": 34, "y": 69},
    {"id": "accepted", "label": "Accepted", "kind": "activity", "x": 34, "y": 84},
    {"id": "cancelled", "label": "Cancelled", "kind": "activity", "x": 80, "y": 66},
    {"id": "end", "label": "■", "kind": "end", "x": 50, "y": 95},
]

# four consecutive weeks (W-SUN labels): three observed + one forecast target.
# A calm mid-series stretch — gentle week-to-week variation (hero edge 346->314->316),
# naive-persistence forecast 316 vs held-out truth 315. Capability framing: the
# forecast lands on the truth without the dramatic drift reserved for slides 4/7b.
WEEKS = ["2016-10-02", "2016-10-09", "2016-10-16", "2016-10-23"]
FRAME_LABELS = ["t₁ · Oct 02", "t₂ · Oct 09", "t₃ · Oct 16", "t₄ · Oct 23 (forecast)"]


def main() -> None:
    df = pd.read_parquet(PARQUET)
    wk = df.resample("W").sum()
    win = wk.loc[WEEKS[0] : WEEKS[-1], list(EDGE_COLS)]

    # per-week edge weights, keyed by edge id
    weekly = []
    for wkdate in WEEKS:
        row = win.loc[f"{wkdate} 00:00:00+00:00"]
        weekly.append({eid: int(row[col]) for col, (eid, *_) in EDGE_COLS.items()})

    obs = weekly[:3]
    truth = weekly[3]  # held-out ground truth for the forecast week
    # illustrative forecaster: naive persistence (last observed week)
    forecast = dict(obs[2])

    max_freq = max(max(f.values()) for f in weekly)

    frames = []
    for i in range(3):
        frames.append(
            {
                "label": FRAME_LABELS[i],
                "kind": "observed",
                "date": WEEKS[i],
                "weights": obs[i],
            }
        )
    frames.append(
        {
            "label": FRAME_LABELS[3],
            "kind": "forecast",
            "date": WEEKS[3],
            "weights": forecast,
            "truth": truth,
        }
    )

    edges = [
        {"id": eid, "source": src, "target": tgt, "hero": eid == HERO}
        for (eid, src, tgt) in EDGE_COLS.values()
    ]

    data = {
        "dataset": "bpi2017",
        "subgraph": "offer",
        "hero": HERO,
        "accent": "#1d4ed8",
        "max_freq": max_freq,
        "nodes": NODES,
        "edges": edges,
        "frames": frames,
    }

    out = HERE / "data.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    # also emit a JS wrapper so the harness loads over file:// without fetch/CORS
    js = "window.DFG_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n"
    (HERE / "data.js").write_text(js)
    print(f"wrote {out}")
    print(f"wrote {HERE / 'data.js'}")
    print(f"  hero {HERO}: " + " -> ".join(str(f["weights"][HERO]) for f in frames[:3]))
    print(f"  forecast {forecast[HERO]} vs truth {truth[HERO]}")
    print(f"  max_freq {max_freq}")


if __name__ == "__main__":
    main()
