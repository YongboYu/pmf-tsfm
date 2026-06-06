#!/usr/bin/env python3
"""Regenerate the data-driven figures for the CAiSE 2026 deck (Step 1 of the visual pass).

Reads from the authoritative sources pinned in figure_manifest.py and writes PNGs into
slides/template/public/figures/. Idempotent: re-running overwrites in place.

Sourcing (see manifest header for provenance):
  * MAE bars, FT slope, RMSE table  -> results/ comprehensive_evaluation CSVs (machine precise)
  * Drift pair                      -> results/ .npy arrays (only source of per-window series)
  * ER bars, DF-complexity radar    -> transcribed paper Tables 7 / 3 (no clean machine source)

Each scalar chart cross-checks TSFM values against the paper tables and prints any delta.

    python slides/scripts/make_figures.py [--only mae_bars,drift,...]
"""

from __future__ import annotations

import argparse
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import figure_manifest as M
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- deck font: match the Slidev deck (Inter) when installed, else fall back --
from matplotlib import font_manager as _fm


def _deck_font() -> str:
    """Return Inter if matplotlib can find it, else a clean sans fallback.

    The deck is set in Inter; figures should echo it. If Inter is not installed
    on this machine we degrade gracefully (no crash) and warn once.
    """
    available = {f.name for f in _fm.fontManager.ttflist}
    for name in ("Inter", "Inter Variable"):
        if name in available:
            return name
    # Fall back to DejaVu Sans, NOT a system sans like Helvetica Neue: DejaVu has
    # full glyph coverage (incl. the U+2192 arrow used in DF edge names), whereas
    # Helvetica Neue drops it and renders missing-glyph boxes. To get true deck
    # matching everywhere, install Inter (or bundle its TTFs and register them here).
    print(
        "[make_figures] Inter not found in matplotlib's font cache; using DejaVu "
        "Sans. Install/bundle Inter for figures that match the deck font exactly.",
        file=sys.stderr,
    )
    return "DejaVu Sans"


_DECK_FONT = _deck_font()
_INK = M.PALETTE["ink"]
_NEUTRAL = M.PALETTE["neutral"]

# --- global style: clean, Inter + accent, no chartjunk -----------------------
plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": [_DECK_FONT, "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "text.color": _INK,
        "axes.labelcolor": _INK,
        "axes.edgecolor": _NEUTRAL,
        "xtick.color": _NEUTRAL,
        "ytick.color": _NEUTRAL,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    }
)

DELTAS: list[str] = []  # results-vs-paper discrepancies, surfaced at the end


def _read_mae(model_root, subdir, dataset) -> float:
    csv = M.summary_csv(model_root, subdir, dataset)
    return float(pd.read_csv(csv)["mae_mean"].iloc[0])


def _read_rmse(model_root, subdir, dataset) -> float:
    csv = M.summary_csv(model_root, subdir, dataset)
    return float(pd.read_csv(csv)["rmse_mean"].iloc[0])


def _root_for(kind: str):
    return M.baseline_root() if kind == "baseline" else (M.RESULTS / "zero_shot")


def _panel_grid():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    return fig, axes.ravel()


# ---------------------------------------------------------------------------
# 7a — headline zero-shot MAE bars (4 panels, 5 bars each)
# ---------------------------------------------------------------------------
def fig_mae_bars():
    fig, axes = _panel_grid()
    labels = [h[0] for h in M.HEADLINE]
    for ax, ds in zip(axes, M.DATASETS):
        vals = []
        for label, kind, subdir, _key in M.HEADLINE:
            v = _read_mae(_root_for(kind), subdir, ds)
            vals.append(v)
            # sanity-gate TSFMs against paper Table 4 (wide tol: catch wiring errors, not rounding)
            chk = M.TABLE4_MAE_CHECK.get(label, {}).get(ds)
            if chk is not None and abs(v - chk) > max(0.08 * chk, 0.05):
                DELTAS.append(f"MAE {label}/{ds}: results {v:.3f} vs paper~{chk} (check wiring)")
        colors = [
            M.BASELINE_GRAY if lab in M.BASELINE_LABELS else M.FAMILY_COLORS[M.family_of(lab)]
            for lab in labels
        ]
        y = np.arange(len(labels))[::-1]  # first label on top
        ax.barh(y, vals, color=colors, height=0.62)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f"  {v:.2f}", va="center", ha="left", fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(M.DATASET_LABELS[ds], fontweight="bold")
        ax.set_xlim(0, max(vals) * 1.18)
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "Zero-shot MAE — lower is better  (gray = baselines, color = TSFMs)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(M.OUT["mae_bars"])
    plt.close(fig)
    print(f"  wrote {M.OUT['mae_bars'].name}")


# ---------------------------------------------------------------------------
# 3 & 7b — drift pair (XGBoost-only, then TSFM revealed), shared axes
# ---------------------------------------------------------------------------
def _drift_series():
    base = M.baseline_root()
    ds = M.DRIFT_DATASET
    i, h = M.DRIFT_EDGE_INDEX, M.DRIFT_HORIZON_INDEX
    targ = np.load(base / "xgboost" / f"{ds}_chronos_targets.npy")[:, h, i]
    xgb = np.load(base / "xgboost" / f"{ds}_xgboost_all_predictions.npy")[:, h, i]
    zs = M.RESULTS / "zero_shot"
    chr = np.load(zs / "chronos_2" / f"{ds}_chronos2_predictions.npy")[:, h, i]
    moi = np.load(zs / "moirai_2" / f"{ds}_moirai2_uni_predictions.npy")[:, h, i]
    return targ, xgb, chr, moi


def fig_drift():
    targ, xgb, chr, moi = _drift_series()
    x = np.arange(len(targ))
    # shared y-limits computed from ALL series so the two plots overlay exactly
    allv = np.concatenate([targ, xgb, chr, moi])
    ylim = (min(0, allv.min()) - 1, allv.max() * 1.08)

    def _base(ax):
        ax.plot(x, targ, color=M.TRUTH_LINE, lw=2.2, label="Ground truth")
        ax.plot(x, xgb, color=M.BASELINE_GRAY, lw=1.6, label="XGBoost")
        ax.set_xlim(0, len(targ) - 1)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Forecast window")
        ax.set_ylabel(f"7-day count · {M.DRIFT_EDGE_NAME}")

    # beat 3: ground truth + XGBoost only
    fig, ax = plt.subplots(figsize=(9, 4))
    _base(ax)
    ax.set_title(f"{M.DRIFT_DATASET}: {M.DRIFT_EDGE_NAME}", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.savefig(M.OUT["drift_xgb"])
    plt.close(fig)
    print(f"  wrote {M.OUT['drift_xgb'].name}")

    # beat 7b: same plot + revealed TSFM lines (paper Fig 1 reveals TWO)
    fig, ax = plt.subplots(figsize=(9, 4))
    _base(ax)
    ax.plot(x, chr, color=M.FAMILY_COLORS["chronos"], lw=1.8, label="Chronos-2")
    ax.plot(x, moi, color=M.FAMILY_COLORS["moirai"], lw=1.8, label="MOIRAI-2.0")
    ax.set_title(f"{M.DRIFT_DATASET}: {M.DRIFT_EDGE_NAME}", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, ncol=2)
    fig.savefig(M.OUT["drift_tsfm"])
    plt.close(fig)
    print(f"  wrote {M.OUT['drift_tsfm'].name}")


# ---------------------------------------------------------------------------
# 8 left — fine-tuning slope (normalized to each model's zero-shot), faceted
# ---------------------------------------------------------------------------
def fig_ft_slope():
    fig, axes = _panel_grid()
    xpos = {"zs": 0, "lora": 1, "full": 2}
    for ax, ds in zip(axes, M.DATASETS):
        for label, stages in M.FT_MODELS.items():
            color = M.FT_MODEL_COLORS[label]
            zs_root, zs_sub = stages["zs"]
            zs_mae = _read_mae(zs_root, zs_sub, ds)
            xs, ys = [], []
            for st in M.FT_STAGES:
                if stages.get(st) is None:
                    continue
                root, sub = stages[st]
                xs.append(xpos[st])
                ys.append(_read_mae(root, sub, ds) / zs_mae)
            dashed = stages.get("lora") is None  # Chronos-2 has no LoRA midpoint
            ax.plot(
                xs,
                ys,
                marker="o",
                ms=4,
                color=color,
                lw=1.6,
                ls="--" if dashed else "-",
                alpha=0.9,
                label=label + (" (no LoRA)" if dashed else ""),
            )
        ax.axhline(1.0, color="#888", lw=1.0, ls=":")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([M.FT_STAGE_LABELS[s] for s in M.FT_STAGES], fontsize=9)
        ax.set_title(M.DATASET_LABELS[ds], fontweight="bold")
        ax.set_ylabel("MAE relative to zero-shot")
        ax.grid(axis="x", visible=False)
    # one shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Fine-tuning vs zero-shot — below 1.0 = improvement, above = overfit damage",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(M.OUT["ft_slope"])
    plt.close(fig)
    print(f"  wrote {M.OUT['ft_slope'].name}")


# ---------------------------------------------------------------------------
# 9 — Entropic Relevance bars (4 panels) from paper Table 7
# ---------------------------------------------------------------------------
def fig_er_bars():
    fig, axes = _panel_grid()
    for ax, ds in zip(axes, M.DATASETS):
        vals = [M.TABLE7_ER[r][ds] for r in M.ER_BAR_ROWS]
        colors = [
            M.BASELINE_GRAY if r in M.BASELINE_LABELS else M.FAMILY_COLORS[M.family_of(r)]
            for r in M.ER_BAR_ROWS
        ]
        xpos = np.arange(len(M.ER_BAR_ROWS))
        ax.bar(xpos, vals, color=colors, width=0.66)
        for xi, v in zip(xpos, vals):
            ax.text(xi, v, f"{v:.2f}", va="bottom", ha="center", fontsize=8)
        # reference lines
        ax.axhline(M.TABLE7_ER["Truth"][ds], color=M.TRUTH_LINE, lw=1.4, ls="-")
        ax.axhline(M.TABLE7_ER["Training"][ds], color="#888", lw=1.2, ls="--")
        ax.set_xticks(xpos)
        ax.set_xticklabels(
            [r.replace("Seasonal-Naive", "S-Naive") for r in M.ER_BAR_ROWS],
            fontsize=8,
            rotation=20,
            ha="right",
        )
        ax.set_title(M.DATASET_LABELS[ds], fontweight="bold")
        ax.set_ylabel("Entropic Relevance")
        ax.set_ylim(0, max(vals + [M.TABLE7_ER["Training"][ds]]) * 1.18)
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "Entropic Relevance — lower is better  (solid = Truth, dashed = Training reference)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(M.OUT["er_bars"])
    plt.close(fig)
    print(f"  wrote {M.OUT['er_bars'].name}")


# ---------------------------------------------------------------------------
# 9 (main) — single-dataset ER: Truth + Training + 5 forecasters as bars,
# fitting % annotated. Hospital Billing: reusing the historical model is worst.
# ---------------------------------------------------------------------------
def fig_er_single():
    ds = M.ER_SINGLE_DATASET
    rows = M.ER_SINGLE_ROWS
    er = [M.TABLE7_ER[r][ds] for r in rows]
    fit = [M.TABLE7_FIT[r][ds] for r in rows]

    def _color(r):
        if r == "Truth":
            return "#cfd6dc"  # ideal floor — pale, outlined
        if r == "Training":
            return "#c0392b"  # reuse-the-past loser — crimson, made to pop
        if r in M.BASELINE_LABELS:
            return M.BASELINE_GRAY
        return M.FAMILY_COLORS[M.family_of(r)]

    colors = [_color(r) for r in rows]
    edges = ["#7a8893" if r == "Truth" else "none" for r in rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(rows))
    bars = ax.bar(x, er, color=colors, edgecolor=edges, linewidth=1.3, width=0.66)
    top = max(er)
    for xi, (e, f) in enumerate(zip(er, fit)):
        ax.text(
            xi,
            e + top * 0.015,
            f"{e:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        # fitting % inside the bar near the base (white on dark bars, dark on the pale Truth bar)
        inside = "#222" if rows[xi] == "Truth" else "white"
        ax.text(xi, top * 0.04, f"{f:.0f}%", ha="center", va="bottom", fontsize=9, color=inside)
    ax.set_xticks(x)
    labels = [
        r.replace("Seasonal-Naive", "S-Naive").replace("Training", "Training\n(reuse past)")
        for r in rows
    ]
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Entropic Relevance")
    ax.set_ylim(0, top * 1.16)
    ax.grid(axis="x", visible=False)
    ax.set_title(
        f"Entropic Relevance — {M.DATASET_LABELS[ds]}   "
        "(lower = better · in-bar % = traces that fit)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(M.OUT["er_single"])
    plt.close(fig)
    print(f"  wrote {M.OUT['er_single'].name}")


# ---------------------------------------------------------------------------
# backup — full RMSE table (all variants × 4 datasets)
# ---------------------------------------------------------------------------
def fig_rmse_full():
    rows, cell, best = [], [], dict.fromkeys(M.DATASETS, (None, 1000000000.0))
    for label, kind, subdir in M.ALL_VARIANTS:
        vals = {}
        for ds in M.DATASETS:
            try:
                vals[ds] = _read_rmse(_root_for(kind), subdir, ds)
            except FileNotFoundError:
                vals[ds] = None
                DELTAS.append(f"RMSE table: missing {label}/{ds}")
            if vals[ds] is not None and vals[ds] < best[ds][1]:
                best[ds] = (label, vals[ds])
        rows.append((label, kind, vals))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    col_labels = ["Model"] + [M.DATASET_LABELS[d] for d in M.DATASETS]
    text, colors = [], []
    for label, kind, vals in rows:
        line = [label]
        crow = ["#eef0f3" if kind == "baseline" else "white"]
        for ds in M.DATASETS:
            v = vals[ds]
            line.append("—" if v is None else f"{v:.2f}")
            crow.append(
                "#d7e3ff"
                if best[ds][0] == label
                else ("#eef0f3" if kind == "baseline" else "white")
            )
        text.append(line)
        colors.append(crow)
    tbl = ax.table(
        cellText=text, colLabels=col_labels, cellColours=colors, cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for c in range(len(col_labels)):
        tbl[(0, c)].set_text_props(fontweight="bold")
    ax.set_title(
        "Zero-shot RMSE — all variants  (blue = best per log, gray = baselines)",
        fontweight="bold",
        pad=12,
    )
    fig.savefig(M.OUT["rmse_full"])
    plt.close(fig)
    print(f"  wrote {M.OUT['rmse_full'].name}")


# ---------------------------------------------------------------------------
# backup — DF-complexity radar (paper Table 3), 4 datasets on 7 axes
# ---------------------------------------------------------------------------
def fig_df_complexity():
    metrics = M.COMPLEXITY_METRICS
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    for ds, vals in M.TABLE3_COMPLEXITY.items():
        col = M.DATASET_COLORS[ds]
        v = vals + vals[:1]
        ax.plot(angles, v, color=col, lw=1.8, label=M.DATASET_LABELS[ds])
        ax.fill(angles, v, color=col, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 0.8)
    ax.set_title("DF time-series complexity (paper Table 3)", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.10), fontsize=9, frameon=False)
    fig.savefig(M.OUT["df_complexity"])
    plt.close(fig)
    print(f"  wrote {M.OUT['df_complexity'].name}")


FIGURES = {
    "mae_bars": fig_mae_bars,
    "drift": fig_drift,
    "ft_slope": fig_ft_slope,
    "er_bars": fig_er_bars,
    "er_single": fig_er_single,
    "rmse_full": fig_rmse_full,
    "df_complexity": fig_df_complexity,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated subset of: " + ", ".join(FIGURES))
    args = ap.parse_args()
    M.FIG_OUT.mkdir(parents=True, exist_ok=True)
    names = args.only.split(",") if args.only else list(FIGURES)

    failed = []
    for name in names:
        fn = FIGURES.get(name.strip())
        if fn is None:
            print(f"  ! unknown figure '{name}' (have: {', '.join(FIGURES)})")
            continue
        try:
            fn()
        except Exception:
            failed.append(name)
            print(f"  ✗ {name} FAILED:\n{traceback.format_exc()}")

    if DELTAS:
        print("\nResults-vs-paper deltas (expected for baselines/ER; flag if a TSFM appears):")
        for d in DELTAS:
            print(f"  • {d}")
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        sys.exit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
