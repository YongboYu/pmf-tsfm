#!/usr/bin/env python3
"""Phase-A preview driver — render three candidate forms of the beat-7a MAE main figure
so the user can pick. Throwaway; the chosen one is rewritten cleanly into make_figures.py
during Phase B, and the _preview-*.png files are then deleted.

Sources values directly from paper Table 4 (camera-ready, defensible) for the 3 latest
models × 4 datasets. All three previews use the same data + same style as make_figures.py.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import figure_manifest as M
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

# Match make_figures.py rcParams so previews read like real slide figures.
plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    }
)

# --- paper Table 4 (camera-ready), 3 latest models x 4 datasets ----------------
# Bracketed values from Table 4 are ↓X% vs the * best baseline = Seasonal-Naive in all 4 logs.
DATASETS = ["BPI2017", "BPI2019_1", "Sepsis", "Hospital_Billing"]
DS_LABEL = {
    "BPI2017": "BPI2017",
    "BPI2019_1": "BPI2019-1",
    "Sepsis": "Sepsis",
    "Hospital_Billing": "Hospital Billing",
}
MODELS = ["Chronos-2", "MOIRAI-2.0", "TimesFM-2.5"]
ABS_MAE = {
    "Chronos-2": {"BPI2017": 7.25, "BPI2019_1": 11.39, "Sepsis": 0.090, "Hospital_Billing": 1.39},
    "MOIRAI-2.0": {"BPI2017": 6.87, "BPI2019_1": 10.99, "Sepsis": 0.084, "Hospital_Billing": 1.39},
    "TimesFM-2.5": {"BPI2017": 6.87, "BPI2019_1": 10.75, "Sepsis": 0.096, "Hospital_Billing": 1.42},
}
PCT_CUT = {  # ↓% vs Seasonal-Naive (paper Table 4 brackets)
    "Chronos-2": {"BPI2017": 13, "BPI2019_1": 21, "Sepsis": 23, "Hospital_Billing": 21},
    "MOIRAI-2.0": {"BPI2017": 17, "BPI2019_1": 24, "Sepsis": 28, "Hospital_Billing": 21},
    "TimesFM-2.5": {"BPI2017": 17, "BPI2019_1": 26, "Sepsis": 18, "Hospital_Billing": 20},
}
all_pcts = [PCT_CUT[m][d] for m in MODELS for d in DATASETS]
AVG_PCT = sum(all_pcts) / len(all_pcts)  # 20.75 -> "~21%"
HERO = f"Zero-shot TSFMs cut MAE ≈{AVG_PCT:.0f}% vs seasonal-naive"
SUB = "Latest model per family · paper Table 4 · no event logs in pretraining"

OUT_DIR = M.FIG_OUT
COLORS = {m: M.FAMILY_COLORS[M.family_of(m)] for m in MODELS}


# ============================================================================
# Preview 1 — grouped %-reduction bars
# ============================================================================
def preview_bars():
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(DATASETS))
    w = 0.26
    for i, m in enumerate(MODELS):
        ys = [PCT_CUT[m][d] for d in DATASETS]
        bars = ax.bar(x + (i - 1) * w, ys, w, color=COLORS[m], label=m)
        for b, v in zip(bars, ys):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.7,
                f"↓{v}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.axhline(AVG_PCT, color="#444", lw=1.1, ls="--", zorder=1)
    ax.text(
        len(DATASETS) - 0.5,
        AVG_PCT + 0.4,
        f"avg ↓{AVG_PCT:.0f}%",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444",
        style="italic",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS], fontsize=10)
    ax.set_ylabel("MAE reduction vs seasonal-naive (%)")
    ax.set_ylim(0, max(all_pcts) * 1.18)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3, fontsize=9.5, frameon=False, bbox_to_anchor=(0.0, -0.13))
    fig.suptitle(HERO, fontsize=14, fontweight="bold", y=1.00)
    ax.set_title(SUB, fontsize=9.5, color="#555", pad=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = OUT_DIR / "_preview-mae-bars.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")


# ============================================================================
# Preview 2 — compact 3×4 heatmap (number = abs MAE, color = % cut)
# ============================================================================
def preview_heatmap():
    rows, cols = MODELS, DATASETS
    grid_pct = np.array([[PCT_CUT[m][d] for d in cols] for m in rows])
    grid_abs = [[ABS_MAE[m][d] for d in cols] for m in rows]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    cmap = plt.get_cmap("Greens")
    norm = mcolors.Normalize(vmin=10, vmax=30)
    ax.imshow(grid_pct, cmap=cmap, norm=norm, aspect="auto")

    for i, m in enumerate(rows):
        for j, d in enumerate(cols):
            cell = cmap(norm(grid_pct[i, j]))
            text_col = (
                "white"
                if (cell[0] * 0.299 + cell[1] * 0.587 + cell[2] * 0.114) < 0.55
                else "#0a3b1a"
            )
            ax.text(
                j,
                i - 0.16,
                f"{grid_abs[i][j]:.2f}".rstrip("0").rstrip(".") or f"{grid_abs[i][j]:.2f}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=text_col,
            )
            ax.text(
                j,
                i + 0.22,
                f"↓{grid_pct[i, j]}%",
                ha="center",
                va="center",
                fontsize=10.5,
                color=text_col,
            )

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([DS_LABEL[d] for d in cols], fontsize=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=10.5)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="major", visible=False)

    fig.suptitle(HERO, fontsize=14, fontweight="bold", y=1.02)
    ax.set_title(
        "cell number = absolute MAE   ·   color = % cut vs seasonal-naive   ·   greener = bigger cut",
        fontsize=9.5,
        color="#555",
        pad=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / "_preview-mae-heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")


# ============================================================================
# Preview 3 — hero number + minimal per-dataset support strip
# ============================================================================
def preview_hero():
    fig = plt.figure(figsize=(10, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.35)

    # top: huge hero number
    ax0 = fig.add_subplot(gs[0])
    ax0.axis("off")
    ax0.text(
        0.5,
        0.62,
        f"↓{AVG_PCT:.0f}%",
        ha="center",
        va="center",
        fontsize=110,
        fontweight="bold",
        color=M.FAMILY_COLORS["chronos"],
    )
    ax0.text(
        0.5,
        0.12,
        "average MAE reduction · 3 latest TSFMs · zero-shot · vs seasonal-naive",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#333",
    )

    # bottom: per-dataset support strip — for each dataset, show the 3 model %s as small dots
    ax1 = fig.add_subplot(gs[1])
    x = np.arange(len(DATASETS))
    for i, m in enumerate(MODELS):
        ys = [PCT_CUT[m][d] for d in DATASETS]
        ax1.scatter(x + (i - 1) * 0.12, ys, s=58, color=COLORS[m], label=m, zorder=3)
    ax1.axhline(AVG_PCT, color="#444", lw=1.0, ls="--", zorder=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([DS_LABEL[d] for d in DATASETS], fontsize=9)
    ax1.set_ylim(0, max(all_pcts) * 1.15)
    ax1.set_ylabel("% cut", fontsize=9)
    ax1.grid(axis="x", visible=False)
    ax1.legend(loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.55))
    ax1.set_title(
        "every model, every log: below the dashed avg line is improvement",
        fontsize=9,
        color="#555",
        pad=6,
    )

    fig.suptitle(HERO, fontsize=13, fontweight="bold", y=0.995)
    out = OUT_DIR / "_preview-mae-hero.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")


# ============================================================================
# Preview 1b — bars + the 2 baselines (Naive at 0, XGBoost below it)
# ============================================================================
def preview_bars_v2():
    # paper Table 4 baselines (vs Seasonal-Naive, the * best). XGBoost is worse on all 4.
    NAIVE = {"BPI2017": 8.30, "BPI2019_1": 14.47, "Sepsis": 0.117, "Hospital_Billing": 1.77}
    XGB = {"BPI2017": 8.50, "BPI2019_1": 14.70, "Sepsis": 0.169, "Hospital_Billing": 2.67}
    pct_xgb = {d: round(100 * (NAIVE[d] - XGB[d]) / NAIVE[d], 1) for d in DATASETS}

    series = [
        ("Seasonal-Naive", "#cfd6dc", dict.fromkeys(DATASETS, 0.0)),
        ("XGBoost", "#8b95a1", pct_xgb),
    ] + [(m, COLORS[m], PCT_CUT[m]) for m in MODELS]

    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    x = np.arange(len(DATASETS))
    n = len(series)
    w = 0.84 / n
    for i, (name, col, vals) in enumerate(series):
        ys = [vals[d] for d in DATASETS]
        bars = ax.bar(
            x + (i - (n - 1) / 2) * w,
            ys,
            w,
            color=col,
            label=name,
            edgecolor="#5f6770" if name == "Seasonal-Naive" else "none",
            linewidth=1.0,
        )
        # Naive's "bar" has height 0 -> draw a small tick on the zero line so it's visible
        if name == "Seasonal-Naive":
            for b in bars:
                ax.add_patch(
                    plt.Rectangle(
                        (b.get_x(), -0.35),
                        b.get_width(),
                        0.7,
                        facecolor=col,
                        edgecolor="#5f6770",
                        lw=0.8,
                        zorder=2,
                    )
                )
        # value labels
        for b, v in zip(bars, ys):
            cx = b.get_x() + b.get_width() / 2
            if name == "Seasonal-Naive":
                ax.text(cx, 1.6, "0", ha="center", va="bottom", fontsize=8, color="#5f6770")
            elif v >= 0:
                ax.text(cx, v + 0.8, f"↓{v:.0f}%", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(
                    cx,
                    v - 1.2,
                    f"↑{abs(v):.0f}%",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#5a2a2a",
                )

    ax.axhline(0, color="#444", lw=1.0, zorder=1)
    ax.axhline(AVG_PCT, color=M.FAMILY_COLORS["chronos"], lw=1.0, ls="--", zorder=1)
    ax.text(
        len(DATASETS) - 0.55,
        AVG_PCT + 0.7,
        f"TSFM avg ↓{AVG_PCT:.0f}%",
        ha="right",
        va="bottom",
        fontsize=9,
        color=M.FAMILY_COLORS["chronos"],
        style="italic",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS], fontsize=10)
    ax.set_ylabel("MAE reduction (%)")
    ymin = min(min(s[2].values()) for s in series)
    ymax = max(max(s[2].values()) for s in series)
    ax.set_ylim(ymin - 8, ymax + 8)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.22))
    fig.suptitle(
        "Zero-shot TSFMs cut MAE ≈21% across all four logs", fontsize=14, fontweight="bold", y=1.00
    )
    ax.set_title(
        "positive = better than Seasonal-Naive · paper Table 4 · no event logs in pretraining",
        fontsize=9.5,
        color="#555",
        pad=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = OUT_DIR / "_preview-mae-bars-v2.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preview_bars()
    preview_heatmap()
    preview_hero()
    preview_bars_v2()
    print("Done — candidates in", OUT_DIR)
