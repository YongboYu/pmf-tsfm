#!/usr/bin/env python3
"""Regenerate the data-driven figures for the CAiSE 2026 deck (Step 1 of the visual pass).

Reads from the authoritative sources pinned in figure_manifest.py and writes PNGs into
slides/template/public/figures/. Idempotent: re-running overwrites in place.

Sourcing (see manifest header for provenance):
  * MAE bars                        -> paper Table 4 (TABLE4_MAE), cross-checked vs results/ CSVs.
        Baselines on results/ are a re-run that differs from camera-ready (esp. XGBoost Sepsis);
        the slide must be paper-faithful (SLIDES.md), so we plot the paper and print result deltas.
  * FT slope, RMSE table            -> results/ comprehensive_evaluation CSVs (machine precise).
        RMSE is results-only — the paper reports MAE (Table 4); RMSE is a backup/verbal claim, so
        its baselines carry the same re-run caveat (flagged in the slide caption).
  * Drift pair                      -> results/ .npy arrays (only source of per-window series;
        XGBoost line is the re-run — matches paper Fig 1 qualitatively, no paper machine data).
  * ER bars, DF-complexity radar    -> transcribed paper Tables 7 / 3 (no clean machine source)

MAE/FT/RMSE cross-check TSFM values against the paper tables and print any delta.

    python slides/scripts/make_figures.py [--only mae_bars,drift,...]
"""

from __future__ import annotations

import argparse
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import figure_manifest as M
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
            # PLOT the paper Table 4 value (paper-faithful per SLIDES.md), then cross-check the
            # results CSV and print any delta (baseline deltas are expected — see manifest note).
            v = M.TABLE4_MAE[label][ds]
            vals.append(v)
            res = _read_mae(_root_for(kind), subdir, ds)
            tol = 0.08 * v if label not in M.BASELINE_LABELS else 0.0  # tight for TSFMs
            if abs(res - v) > max(tol, 0.05):
                DELTAS.append(f"MAE {label}/{ds}: plotted paper {v:.3f} vs results {res:.3f}")
        colors = [
            M.BASELINE_GRAY if lab in M.BASELINE_LABELS else M.FAMILY_COLORS[M.family_of(lab)]
            for lab in labels
        ]
        y = np.arange(len(labels))[::-1]  # first label on top
        ax.barh(y, vals, color=colors, height=0.62)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f"  {v:.2f}", va="center", ha="left", fontsize=11)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_title(M.DATASET_LABELS[ds], fontweight="bold", fontsize=14)
        ax.set_xlim(0, max(vals) * 1.18)
        ax.grid(axis="y", visible=False)
    # suptitle dropped — the slide's kicker + the two callouts carry the framing (S13 co-draft).
    fig.tight_layout()
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
# S5 / S6 / S14 — DF-series exhibits (drift + intermittent), half-width LINE panels.
# One exhibit shown across three slides: S5 truth-only, S6 +XGBoost, S14 +TSFM.
# x = "Day" (stride = 1 ⇒ consecutive windows are consecutive days); y = "Value"
# (the 7th forecast-day count per window, NOT a 7-day sum). Bigger fonts for the projector.
# ---------------------------------------------------------------------------
def _panel_axes(ax):
    ax.set_xlabel("Day", fontsize=15)
    ax.set_ylabel("Value", fontsize=15)
    ax.tick_params(labelsize=13)
    for lbl in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        lbl.set_color(_NEUTRAL)


def _intermittent_series():
    """truth / XGBoost / best-TSFM last-day series for the chosen intermittent edge
    (BPI2019-1 Cancel Invoice Receipt -> Record Invoice Receipt, feat 59)."""
    base = M.baseline_root()
    ds = M.INTERMITTENT_DATASET
    i, h = M.INTERMITTENT_EDGE_INDEX, M.INTERMITTENT_HORIZON_INDEX
    sub, key = M.INTERMITTENT_TSFM
    targ = np.load(base / "xgboost" / f"{ds}_chronos_targets.npy")[:, h, i]
    xgb = np.load(base / "xgboost" / f"{ds}_xgboost_all_predictions.npy")[:, h, i]
    tsfm = np.load(M.RESULTS / "zero_shot" / sub / f"{ds}_{key}_predictions.npy")[:, h, i]
    return targ, xgb, tsfm


def _drift_ylim():
    """Shared y-limits across the drift slides (S5 truth, S6 +XGBoost, S14 +TSFM)
    so the three views of the Sent→Cancelled exhibit overlay exactly for the callback."""
    targ, xgb, chr, moi = _drift_series()
    allv = np.concatenate([targ, xgb, chr, moi])
    return (min(0, allv.min()) - 1, allv.max() * 1.08)


def _line_panel(out_key, lines, ylim, n, annotate=None):
    """Render one half-width line panel. lines = [(array, color, lw, label)];
    label=None ⇒ no legend entry. Shared style across S5/S6/S14 for the callback.
    annotate(ax) ⇒ optional hook to draw amber attention marks in data coords (S14)."""
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labelled = False
    for arr, color, lw, label in lines:
        ax.plot(x, arr, color=color, lw=lw, label=label)
        labelled = labelled or (label is not None)
    ax.set_xlim(0, n - 1)
    ax.set_ylim(*ylim)
    _panel_axes(ax)
    if labelled:
        ax.legend(loc="upper right", framealpha=0.9, fontsize=12)
    if annotate is not None:
        annotate(ax)
    fig.tight_layout()
    fig.savefig(M.OUT[out_key])
    plt.close(fig)
    print(f"  wrote {M.OUT[out_key].name}")


def fig_s5_drift_truth():
    targ, *_ = _drift_series()
    _line_panel("s5_drift_truth", [(targ, M.TRUTH_LINE, 2.2, None)], _drift_ylim(), len(targ))


def fig_s5_intermittent_truth():
    targ, _xgb, _tsfm = _intermittent_series()
    ylim = (0, targ.max() * 1.10)  # scaled to truth so the sparse spikes read clearly
    _line_panel("s5_intermittent_truth", [(targ, M.TRUTH_LINE, 2.0, None)], ylim, len(targ))


def fig_s6_drift_xgb():
    targ, xgb, *_ = _drift_series()
    _line_panel(
        "s6_drift_xgb",
        [(targ, M.TRUTH_LINE, 2.2, "Ground truth"), (xgb, M.BASELINE_GRAY, 1.8, "XGBoost")],
        _drift_ylim(),
        len(targ),
    )


def fig_s6_intermittent_xgb():
    targ, xgb, _tsfm = _intermittent_series()
    # expand y to include XGBoost's overshoot — the axis growing far past the truth range
    # IS the overfitting visual (XGBoost hallucinates ~40–71 where the truth is mostly 0).
    ylim = (0, max(float(targ.max()), float(xgb.max())) * 1.08)
    _line_panel(
        "s6_intermittent_xgb",
        [(targ, M.TRUTH_LINE, 2.2, "Ground truth"), (xgb, M.BASELINE_GRAY, 1.8, "XGBoost")],
        ylim,
        len(targ),
    )


def fig_s14_drift_tsfm():
    """S14 drift callback: S6 view + MOIRAI-2.0 revealed, amber box over the back-half
    region where it dips toward the truth while XGBoost stays stuck high."""
    targ, xgb, _chr, moi = _drift_series()
    n = len(targ)

    def _mark(ax):
        x0, x1, y0, y1 = 36, n - 1, 0, 68
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, fill=False,
            edgecolor=M.ACCENT, lw=2.5, zorder=5))

    _line_panel(
        "s14_drift_tsfm",
        [(targ, M.TRUTH_LINE, 2.2, "Ground truth"),
         (xgb, M.BASELINE_GRAY, 1.8, "XGBoost"),
         (moi, M.FAMILY_COLORS["moirai"], 2.0, "MOIRAI-2.0")],
        _drift_ylim(),            # SAME ylim as S5/S6 drift → overlays the swap exactly
        n, annotate=_mark,
    )


def fig_s14_intermittent_tsfm():
    """S14 sparsity callback: S6 view + MOIRAI-2.0 revealed (holds near zero), amber arrow
    pointing down to the near-zero line in a clean gap between XGBoost's false bursts."""
    targ, xgb, tsfm = _intermittent_series()   # tsfm = MOIRAI-2.0 per INTERMITTENT_TSFM
    ylim = (0, max(float(targ.max()), float(xgb.max())) * 1.08)  # SAME as fig_s6_intermittent_xgb

    def _mark(ax):
        ax.annotate("", xy=(30, 2.0), xytext=(30, 24),
                    arrowprops=dict(facecolor=M.ACCENT, edgecolor=M.ACCENT,
                                    width=5, headwidth=14, headlength=12), zorder=5)

    _line_panel(
        "s14_intermittent_tsfm",
        [(targ, M.TRUTH_LINE, 2.2, "Ground truth"),
         (xgb, M.BASELINE_GRAY, 1.8, "XGBoost"),
         (tsfm, M.FAMILY_COLORS["moirai"], 2.0, "MOIRAI-2.0")],
        ylim,
        len(targ), annotate=_mark,
    )


# ---------------------------------------------------------------------------
# 8 left — fine-tuning slope (normalized to each model's zero-shot), faceted
# ---------------------------------------------------------------------------
def fig_ft_slope():
    # Colour by OUTCOME, not model: a muted "barely moves" bundle hugging the 1.0 zero-shot
    # baseline + amber overfit lines (full-FT). Chronos-2 is a dashed zs→full segment (no LoRA).
    P = M.PALETTE
    MUTED, AMBER, BAND = P["neutral_soft"], P["accent"], P["surface_alt"]
    FS_TITLE, FS_LABEL, FS_TICK, FS_ANNO, FS_LEG = 12, 11.5, 11.5, 11.5, 11.5
    OVERFIT_THRESH = 1.05
    xpos = {"zs": 0, "lora": 1, "full": 2}
    xlab = [M.FT_STAGE_LABELS[s] for s in M.FT_STAGES]  # Zero-shot, LoRA, Full-FT

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.2))
    axes = axes.ravel()
    for i, (ax, ds) in enumerate(zip(axes, M.DATASETS)):
        bottom_row, left_col = i >= 2, i % 2 == 0
        ax.axhspan(0.95, 1.05, color=BAND, zorder=0)
        ax.axhline(1.0, color=P["neutral"], lw=1.0, ls=":", zorder=1)

        amber = []
        for label, stages in M.FT_MODELS.items():
            zs_root, zs_sub = stages["zs"]
            zs = _read_mae(zs_root, zs_sub, ds)
            xs, ys = [], []
            for st in M.FT_STAGES:
                if stages.get(st) is None:
                    continue  # Chronos-2 has no LoRA -> 2-point dashed segment
                root, sub = stages[st]
                xs.append(xpos[st])
                ys.append(_read_mae(root, sub, ds) / zs)
            ls = (0, (5, 2)) if stages.get("lora") is None else "-"
            if ys[-1] > OVERFIT_THRESH:
                amber.append((label, xs, ys, ls))
            else:
                ax.plot(xs, ys, marker="o", ms=4, lw=1.7, color=MUTED,
                        alpha=0.85, ls=ls, zorder=2)
        for label, xs, ys, ls in amber:
            ax.plot(xs, ys, marker="o", ms=5.5, lw=3.0, color=AMBER, ls=ls, zorder=4)
            if ls != "-":  # dashed Chronos-2 -> label mid-line (avoids endpoint clash)
                ax.annotate(label, xy=(1, (ys[0] + ys[-1]) / 2), xytext=(0, 6),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=FS_ANNO, fontweight="bold", color=AMBER, zorder=5)
            else:
                ax.annotate(label, xy=(xs[-1], ys[-1]), xytext=(-4, 6),
                            textcoords="offset points", ha="right", va="bottom",
                            fontsize=FS_ANNO, fontweight="bold", color=AMBER, zorder=5)

        ax.set_title(M.DATASET_LABELS[ds], fontweight="normal",
                     fontsize=FS_TITLE, color=P["neutral"], pad=4)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(xlab if bottom_row else [], fontsize=FS_TICK)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.set_xlim(-0.18, 2.18)
        if left_col:
            ax.set_ylabel("MAE ÷ zero-shot", fontsize=FS_LABEL)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.20)

    legend = [
        Line2D([0], [0], color=MUTED, lw=2.4, marker="o", ms=5, label="flat / minor change"),
        Line2D([0], [0], color=AMBER, lw=3.0, marker="o", ms=5, label="overfit at full fine-tuning"),
        Line2D([0], [0], color=P["neutral"], lw=1.0, ls=":", label="zero-shot baseline = 1.0"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               fontsize=FS_LEG, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1.0), w_pad=3.0, h_pad=2.5)
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
    for xi, e in enumerate(er):
        ax.text(
            xi,
            e + top * 0.015,
            f"{e:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_xticks(x)
    labels = [
        r.replace("Seasonal-Naive", "S-Naive").replace("Training", "Training\n(reuse past)")
        for r in rows
    ]
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Entropic Relevance")
    ax.set_ylim(0, top * 1.16)
    ax.grid(axis="x", visible=False)
    # internal title dropped — the slide supplies the ER full-form line + caption (S16 co-draft).
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

    # best-per-log highlight: a light tint of the deck accent (amber), per the locked
    # palette ("amber = our emphasis"); baselines a faint neutral gray. Tinting toward
    # white keeps dark cell text legible while still reading unmistakably amber.
    def _tint(hex_color, frac):  # frac = fraction toward white
        r, g, b = mcolors.to_rgb(hex_color)
        return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)

    best_amber = _tint(M.ACCENT, 0.62)
    baseline_gray = "#eef0f3"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    col_labels = ["Model"] + [M.DATASET_LABELS[d] for d in M.DATASETS]
    text, colors = [], []
    for label, kind, vals in rows:
        line = [label]
        crow = [baseline_gray if kind == "baseline" else "white"]
        for ds in M.DATASETS:
            v = vals[ds]
            line.append("—" if v is None else f"{v:.2f}")
            crow.append(
                best_amber
                if best[ds][0] == label
                else (baseline_gray if kind == "baseline" else "white")
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
        "Zero-shot RMSE — all variants  (amber = best per log, gray = baselines)",
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


# ---------------------------------------------------------------------------
# S7 — highlighted complexity radar (paper Table 3). Same 7 axes as the B4 backup,
# but the three metrics the paper flags as higher-than-benchmark (Transition /
# Shifting / Non-Gaussianity, main.tex line 211) are emphasised in amber, fonts
# sized for half-width projection. Single navy min–max envelope across the 4 logs
# (+ median line) — calm and on-message; the band spans the value ranges shown in
# the slide's right-hand strip. Per-log detail stays in the B4 backup radar.
# ---------------------------------------------------------------------------
def _radar_axes(metrics):
    """Shared polar scaffold: angles, amber-highlighted spokes/labels for the 3
    harder-than-benchmark axes, [0,1] radius. Returns (fig, ax, angles)."""
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    closed = angles + angles[:1]
    fig, ax = plt.subplots(figsize=(6.4, 6.0), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([])  # custom labels below so we can recolour the 3 axes
    ax.set_ylim(0, 0.8)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=12, color=_NEUTRAL)
    # place the radial tick labels along the spoke between Trend and Stationarity
    # (the open right-hand sector), clear of the polygons on the left.
    ax.set_rlabel_position(77)
    ax.tick_params(pad=2)
    ax.grid(color="#c9d3de", lw=0.7, alpha=0.9)
    ax.spines["polar"].set_color("#c9d3de")
    # custom axis labels: amber+bold for the highlighted metrics, neutral otherwise.
    for ang, m in zip(angles, metrics):
        hot = m in M.COMPLEXITY_HIGHLIGHT
        ax.plot(
            [ang, ang], [0, 0.8],
            color=M.ACCENT if hot else "#c9d3de",
            lw=2.4 if hot else 0.7,
            alpha=0.9 if hot else 0.6,
            zorder=1,
        )
        ha = "center"
        if 0.05 < ang < np.pi - 0.05:
            ha = "left"
        elif np.pi + 0.05 < ang < 2 * np.pi - 0.05:
            ha = "right"
        ax.text(
            ang, 0.95, m,
            color=M.ACCENT if hot else _INK,
            fontsize=20 if hot else 17,
            fontweight="bold" if hot else "normal",
            ha=ha, va="center",
        )
    return fig, ax, closed


def fig_s7_complexity():
    metrics = M.COMPLEXITY_METRICS
    fig, ax, closed = _radar_axes(metrics)
    rows = np.array(list(M.TABLE3_COMPLEXITY.values()))  # (4 logs, 7 metrics)
    lo = rows.min(axis=0).tolist()
    hi = rows.max(axis=0).tolist()
    med = np.median(rows, axis=0).tolist()
    lo_c, hi_c, med_c = lo + lo[:1], hi + hi[:1], med + med[:1]
    navy = M.PALETTE["brand"]
    ax.fill_between(closed, lo_c, hi_c, color=navy, alpha=0.14, zorder=2)
    ax.plot(closed, hi_c, color=navy, lw=1.2, alpha=0.55, zorder=3)
    ax.plot(closed, lo_c, color=navy, lw=1.2, alpha=0.55, zorder=3)
    ax.plot(closed, med_c, color=navy, lw=2.6, zorder=4)
    # no in-figure legend/caption (a bottom legend collides with the Shifting axis label);
    # the band/median is explained in the slide's HTML caption beneath the figure.
    fig.tight_layout()
    fig.savefig(M.OUT["s7_complexity"])
    plt.close(fig)
    print(f"  wrote {M.OUT['s7_complexity'].name}")


FIGURES = {
    "mae_bars": fig_mae_bars,
    "drift": fig_drift,
    "s5_drift_truth": fig_s5_drift_truth,
    "s5_intermittent_truth": fig_s5_intermittent_truth,
    "s6_drift_xgb": fig_s6_drift_xgb,
    "s6_intermittent_xgb": fig_s6_intermittent_xgb,
    "s14_drift_tsfm": fig_s14_drift_tsfm,
    "s14_intermittent_tsfm": fig_s14_intermittent_tsfm,
    "ft_slope": fig_ft_slope,
    "er_bars": fig_er_bars,
    "er_single": fig_er_single,
    "rmse_full": fig_rmse_full,
    "df_complexity": fig_df_complexity,
    "s7_complexity": fig_s7_complexity,
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
