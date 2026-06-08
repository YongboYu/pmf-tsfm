"""Figure-data manifest for the CAiSE 2026 deck (Step 0 of the visual pass).

Single source of truth pinning every data-driven figure to its authoritative
source file(s) across the FRAGMENTED results tree, plus the model-key -> display
label map. The results tree has duplicate/renamed copies (chronos_2 vs chronos2
vs chronos_2_0, moirai_1_1_small vs ..._small_uni vs ..._mul); resolving paths
here -- once -- keeps make_figures.py from silently reading a stale copy.

Provenance notes (verified 2026-05-25):
  * TSFM zero-shot MAE in results/ matches paper Table 4 EXACTLY
    (chronos_2 BPI2017 7.249 == 7.25, moirai_2 6.871 == 6.87, timesfm_2_5 6.872 == 6.87).
  * Baselines and ER in results/ DIFFER slightly from the camera-ready paper
    (XGBoost BPI2017 8.36 vs Table 4 8.50; Chronos-2 ER 1.117 vs Table 7 1.09) --
    same models, re-run. We build from results/ and flag the delta. See plan.
  * 'multivariate' (_mul) variants are deliberately excluded (univariate per Yu 2025).
"""

import json
from pathlib import Path

# --- palette (single source of truth) ----------------------------------------
# Canonical colour set shared with the deck (slides/template/style.css). Loaded
# from slides/palette.json so figures and slides never drift apart.
PALETTE = json.loads((Path(__file__).resolve().parent.parent / "palette.json").read_text())

# --- roots -------------------------------------------------------------------
# Canonical experiment outputs live OUTSIDE the slides repo, as a sibling of code/.
RESULTS = Path("/Users/yongboyu/Documents/PhD/pmf-tsfm/results")
# ER summaries with a clean per-model structure live in the code-tree outputs/.
ER_ROOT = Path("/Users/yongboyu/Documents/PhD/code/pmf-tsfm/outputs/er/zero_shot")
# Where rendered figures must land (Slidev serves public/ at site root).
FIG_OUT = Path(__file__).resolve().parent.parent / "template" / "public" / "figures"

# --- datasets ----------------------------------------------------------------
DATASETS = ["BPI2017", "BPI2019_1", "Sepsis", "Hospital_Billing"]
DATASET_LABELS = {
    "BPI2017": "BPI2017",
    "BPI2019_1": "BPI2019-1",
    "Sepsis": "Sepsis",
    "Hospital_Billing": "Hospital Billing",
}
# Categorical colours per event log (backup DF-complexity radar) — from palette.json.
DATASET_COLORS = {k: v for k, v in PALETTE["dataset"].items() if not k.startswith("_")}

# --- drift edge (beats 3 / 7b) ----------------------------------------------
# Confirmed: feature index 3 == "O_Sent (mail and online) -> O_Cancelled" across
# baselines AND TSFMs (all arrays (58, 7, 21)). Paper Fig 1 plots the last-day
# (horizon step 7 -> index -1) prediction per window against the target.
DRIFT_DATASET = "BPI2017"
DRIFT_EDGE_INDEX = 3
DRIFT_EDGE_NAME = "O_Sent → O_Cancelled"
DRIFT_HORIZON_INDEX = -1  # last day

# --- intermittent edge (S5 right panel / S6 / S14 reveal) -------------------
# BPI2019-1 "Cancel Invoice Receipt -> Record Invoice Receipt" (feature index 59):
# the chosen intermittent exemplar — ~80% zeros, spikes to 29. XGBoost overfits and
# hallucinates 40–71 in the long zero stretches (MAE 18.4); MOIRAI-2.0 stays controlled
# (MAE 2.0, −89%). This is the "ML/DL overfit" exhibit (S6) and the TSFM-better reveal (S14).
# Targets come from baseline/xgboost/<DS>_chronos_targets.npy (same windows as XGBoost).
INTERMITTENT_DATASET = "BPI2019_1"
INTERMITTENT_EDGE_INDEX = 59
INTERMITTENT_EDGE_NAME = "Cancel Invoice Receipt → Record Invoice Receipt"
INTERMITTENT_HORIZON_INDEX = -1  # last (7th) forecast day
# best TSFM on this series (revealed at S14): (subdir, file-key) under RESULTS/zero_shot
INTERMITTENT_TSFM = ("moirai_2", "moirai2_uni")  # MOIRAI-2.0

# --- style (KU Leuven house-style; CVD-verified family trio; from palette.json) -
ACCENT = PALETTE["accent"]  # attention-only highlight (KU Leuven orange)
BASELINE_GRAY = PALETTE["baseline"]
TRUTH_LINE = PALETTE["truth"]
TARGET_GRAY = "#444444"

# Per-family colours for family-comparison slides (timeline, drift reveal, FT slope).
# Headline 2-group bars use BASELINE_GRAY vs PALETTE["tsfm"] instead.
FAMILY_COLORS = PALETTE["family"]

# --- headline models: zero-shot MAE bars (7a), ER bars (9), drift (7b) -------
# display label -> (kind, results-subdir, file-key)
#   kind 'baseline' -> RESULTS/baseline/<dir>/...   kind 'tsfm' -> RESULTS/zero_shot/<dir>/...
HEADLINE = [
    ("Seasonal-Naive", "baseline", "naive_season", "naive_seasonal"),
    ("XGBoost", "baseline", "xgboost", "xgboost"),
    ("Chronos-2", "tsfm", "chronos_2", "chronos2"),
    ("MOIRAI-2.0", "tsfm", "moirai_2", "moirai2_uni"),
    ("TimesFM-2.5", "tsfm", "timesfm_2_5", "timesfm2_5"),
]
BASELINE_LABELS = {"Seasonal-Naive", "XGBoost"}

# --- fine-tuning slope (8 left): ZS -> LoRA -> Full-FT -----------------------
# display label -> {stage: (root, subdir)}  (None = stage not run, e.g. Chronos-2 LoRA)
FT_MODELS = {
    "Chronos-Bolt-small": {
        "zs": (RESULTS / "zero_shot", "chronos_bolt_small"),
        "lora": (RESULTS / "lora_tune", "chronos_bolt_small"),
        "full": (RESULTS / "full_tune", "chronos_bolt_small"),
    },
    "Chronos-Bolt-base": {
        "zs": (RESULTS / "zero_shot", "chronos_bolt_base"),
        "lora": (RESULTS / "lora_tune", "chronos_bolt_base"),
        "full": (RESULTS / "full_tune", "chronos_bolt_base"),
    },
    "MOIRAI-1.1-R-small": {
        "zs": (RESULTS / "zero_shot", "moirai_1_1_small"),
        "lora": (RESULTS / "lora_tune", "moirai_1_1_small_uni"),
        "full": (RESULTS / "full_tune", "moirai_1_1_small_uni"),
    },
    "MOIRAI-1.1-R-large": {
        "zs": (RESULTS / "zero_shot", "moirai_1_1_large"),
        "lora": (RESULTS / "lora_tune", "moirai_1_1_large_uni"),
        "full": (RESULTS / "full_tune", "moirai_1_1_large_uni"),
    },
    "Chronos-2": {
        "zs": (RESULTS / "zero_shot", "chronos_2"),
        "lora": None,  # no LoRA run for Chronos-2 (paper Table 6) -> 2-point dashed segment
        "full": (RESULTS / "full_tune", "chronos_2"),
    },
}
FT_STAGES = ["zs", "lora", "full"]
FT_STAGE_LABELS = {"zs": "Zero-shot", "lora": "LoRA", "full": "Full-FT"}
# Distinct shade per model, grouped by family hue (azure = Chronos, teal = MOIRAI),
# so the 5 lines stay tellable apart (family color alone collapses 3 Chronos into one).
FT_MODEL_COLORS = {k: v for k, v in PALETTE["ft_shades"].items() if not k.startswith("_")}


def family_of(label: str) -> str:
    l = label.lower()
    if "chronos" in l:
        return "chronos"
    if "moirai" in l:
        return "moirai"
    return "timesfm"


# --- summary CSV access ------------------------------------------------------
# Every model dir (baseline/zero_shot/lora_tune/full_tune) carries a
# comprehensive_evaluation/ with one sub-dir per dataset holding a *_summary.csv
# (columns: dataset,n_sequences,n_series,horizon,mae_mean,mae_std,rmse_mean,...).
# Sub-dir naming is INCONSISTENT across the tree: usually '<DATASET>/<DATASET>_summary.csv'
# but sometimes '<DATASET>_1.1_small_uni/<DATASET>_1.1_small_uni_summary.csv'. Resolve by glob.
def summary_csv(model_root: Path, subdir: str, dataset: str) -> Path:
    ce = model_root / subdir / "comprehensive_evaluation"
    ev_dirs = sorted(d for d in ce.glob(f"{dataset}*") if d.is_dir())
    if not ev_dirs:
        raise FileNotFoundError(f"no eval dir for {dataset} under {ce}")
    csvs = sorted(ev_dirs[0].glob("*summary*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no *summary.csv in {ev_dirs[0]}")
    return csvs[0]


def baseline_root() -> Path:
    return RESULTS / "baseline"


# --- full RMSE table (backup) : all variants ---------------------------------
# display label -> (kind, subdir). kind 'baseline' -> RESULTS/baseline, else RESULTS/zero_shot.
ALL_VARIANTS = [
    ("Seasonal-Naive", "baseline", "naive_season"),
    ("XGBoost", "baseline", "xgboost"),
    ("Chronos-Bolt-tiny", "tsfm", "chronos_bolt_tiny"),
    ("Chronos-Bolt-mini", "tsfm", "chronos_bolt_mini"),
    ("Chronos-Bolt-small", "tsfm", "chronos_bolt_small"),
    ("Chronos-Bolt-base", "tsfm", "chronos_bolt_base"),
    ("Chronos-2", "tsfm", "chronos_2"),
    ("MOIRAI-1.1-R-small", "tsfm", "moirai_1_1_small"),
    ("MOIRAI-1.1-R-large", "tsfm", "moirai_1_1_large"),
    ("MOIRAI-MoE-base", "tsfm", "moirai_moe"),
    ("MOIRAI-2.0", "tsfm", "moirai_2"),
    ("TimesFM-1.0", "tsfm", "timesfm_1"),
    ("TimesFM-2.0", "tsfm", "timesfm_2"),
    ("TimesFM-2.5", "tsfm", "timesfm_2_5"),
]

# --- DF complexity radar (backup) -- transcribed from paper Table 3 ----------
# (no machine source: these are derived statistics reported only in the paper)
COMPLEXITY_METRICS = [
    "Seasonality",
    "Trend",
    "Stationarity",
    "Transition",
    "Shifting",
    "Correlation",
    "Non-Gaussianity",
]
TABLE3_COMPLEXITY = {
    "BPI2017": [0.734, 0.255, 0.222, 0.059, 0.269, 0.646, 0.334],
    "BPI2019_1": [0.698, 0.154, 0.094, 0.145, 0.362, 0.690, 0.465],
    "Sepsis": [0.653, 0.087, 0.003, 0.225, 0.276, 0.705, 0.585],
    "Hospital_Billing": [0.603, 0.260, 0.137, 0.171, 0.483, 0.620, 0.457],
}

# --- paper Table 4 MAE (camera-ready) -- the PLOT SOURCE for the headline MAE bars (fig_mae_bars,
# which reads only the HEADLINE labels) AND the full MAE backup table (fig_mae_full, all 14 rows).
# Slide numbers must be paper-faithful (SLIDES.md hard rule). The results/ re-run baselines
# differ materially from the camera-ready table (esp. XGBoost Sepsis .169 vs the re-run's ~.094,
# also XGBoost on every log + Naive on Hospital), which would understate the baselines and distort
# the "TSFMs beat both baselines on every log" claim. So we PLOT these paper values and cross-check
# the results CSVs in make_figures (baseline deltas are expected and printed). TSFM rows match the
# results within rounding. Best (lower) baseline is Naive Seasonal on all four logs.
# Means transcribed from manuscript/tables/results_1_mae.tex (tab:results_1_MAE).
TABLE4_MAE = {
    "Seasonal-Naive": {"BPI2017": 8.30, "BPI2019_1": 14.47, "Sepsis": 0.117, "Hospital_Billing": 1.77},
    "XGBoost": {"BPI2017": 8.50, "BPI2019_1": 14.70, "Sepsis": 0.169, "Hospital_Billing": 2.67},
    "Chronos-Bolt-tiny": {"BPI2017": 11.64, "BPI2019_1": 13.02, "Sepsis": 0.101, "Hospital_Billing": 1.39},
    "Chronos-Bolt-mini": {"BPI2017": 9.70, "BPI2019_1": 12.00, "Sepsis": 0.098, "Hospital_Billing": 1.40},
    "Chronos-Bolt-small": {"BPI2017": 7.72, "BPI2019_1": 11.85, "Sepsis": 0.100, "Hospital_Billing": 1.40},
    "Chronos-Bolt-base": {"BPI2017": 7.62, "BPI2019_1": 11.64, "Sepsis": 0.098, "Hospital_Billing": 1.40},
    "Chronos-2": {"BPI2017": 7.25, "BPI2019_1": 11.39, "Sepsis": 0.090, "Hospital_Billing": 1.39},
    "MOIRAI-1.1-R-small": {"BPI2017": 10.24, "BPI2019_1": 12.88, "Sepsis": 0.088, "Hospital_Billing": 1.48},
    "MOIRAI-1.1-R-large": {"BPI2017": 9.29, "BPI2019_1": 12.30, "Sepsis": 0.086, "Hospital_Billing": 1.43},
    "MOIRAI-MoE-base": {"BPI2017": 7.34, "BPI2019_1": 11.37, "Sepsis": 0.085, "Hospital_Billing": 1.40},
    "MOIRAI-2.0": {"BPI2017": 6.87, "BPI2019_1": 10.99, "Sepsis": 0.084, "Hospital_Billing": 1.39},
    "TimesFM-1.0": {"BPI2017": 7.18, "BPI2019_1": 12.85, "Sepsis": 0.098, "Hospital_Billing": 1.42},
    "TimesFM-2.0": {"BPI2017": 7.07, "BPI2019_1": 11.54, "Sepsis": 0.097, "Hospital_Billing": 1.42},
    "TimesFM-2.5": {"BPI2017": 6.87, "BPI2019_1": 10.75, "Sepsis": 0.096, "Hospital_Billing": 1.42},
}

# --- paper Table 5 RMSE (camera-ready) -- the PLOT SOURCE for the full RMSE backup table
# (fig_rmse_full). The paper added a dedicated RMSE table, so the backup is now paper-faithful (no
# longer the results/ re-run + caveat). Means transcribed from manuscript/tables/results_1_rmse.tex
# (tab:results_1_RMSE). Labels match ALL_VARIANTS / TABLE4_MAE exactly.
TABLE5_RMSE = {
    "Seasonal-Naive": {"BPI2017": 12.43, "BPI2019_1": 25.58, "Sepsis": 0.187, "Hospital_Billing": 2.21},
    "XGBoost": {"BPI2017": 11.91, "BPI2019_1": 23.87, "Sepsis": 0.209, "Hospital_Billing": 3.03},
    "Chronos-Bolt-tiny": {"BPI2017": 14.41, "BPI2019_1": 21.08, "Sepsis": 0.134, "Hospital_Billing": 1.70},
    "Chronos-Bolt-mini": {"BPI2017": 12.27, "BPI2019_1": 19.98, "Sepsis": 0.131, "Hospital_Billing": 1.71},
    "Chronos-Bolt-small": {"BPI2017": 10.27, "BPI2019_1": 19.35, "Sepsis": 0.133, "Hospital_Billing": 1.71},
    "Chronos-Bolt-base": {"BPI2017": 10.08, "BPI2019_1": 18.86, "Sepsis": 0.133, "Hospital_Billing": 1.72},
    "Chronos-2": {"BPI2017": 9.85, "BPI2019_1": 19.02, "Sepsis": 0.128, "Hospital_Billing": 1.70},
    "MOIRAI-1.1-R-small": {"BPI2017": 13.13, "BPI2019_1": 21.48, "Sepsis": 0.131, "Hospital_Billing": 1.81},
    "MOIRAI-1.1-R-large": {"BPI2017": 11.91, "BPI2019_1": 20.79, "Sepsis": 0.129, "Hospital_Billing": 1.75},
    "MOIRAI-MoE-base": {"BPI2017": 9.94, "BPI2019_1": 20.02, "Sepsis": 0.128, "Hospital_Billing": 1.72},
    "MOIRAI-2.0": {"BPI2017": 9.39, "BPI2019_1": 18.58, "Sepsis": 0.125, "Hospital_Billing": 1.71},
    "TimesFM-1.0": {"BPI2017": 9.66, "BPI2019_1": 20.44, "Sepsis": 0.134, "Hospital_Billing": 1.73},
    "TimesFM-2.0": {"BPI2017": 9.50, "BPI2019_1": 18.94, "Sepsis": 0.133, "Hospital_Billing": 1.72},
    "TimesFM-2.5": {"BPI2017": 9.32, "BPI2019_1": 18.12, "Sepsis": 0.132, "Hospital_Billing": 1.73},
}

# --- paper Table 6 (camera-ready) -- fine-tuning results: MAE & RMSE for zero-shot / LoRA / full
# fine-tuning on the five fine-tunable TSFMs (Chronos-2 has no LoRA -> 2 rows). Source for the
# fine-tuning backup table (fig_ft_table). Means transcribed from manuscript/tables/results_2.tex
# (tab:results_2); each cell is (MAE, RMSE) per log. The story matches S15: gains are marginal and
# inconsistent — sometimes worse (extreme: MOIRAI-1.1-R-large full-tune BPI2019_1 23.06 vs 12.30 ZS).
TABLE6_FT = [
    ("Chronos-Bolt-small", "zero-shot", {"BPI2017": (7.72, 10.27), "BPI2019_1": (11.85, 19.35), "Sepsis": (0.100, 0.133), "Hospital_Billing": (1.40, 1.71)}),
    ("Chronos-Bolt-small", "LoRA", {"BPI2017": (7.74, 10.32), "BPI2019_1": (12.10, 19.83), "Sepsis": (0.105, 0.138), "Hospital_Billing": (1.38, 1.70)}),
    ("Chronos-Bolt-small", "full tune", {"BPI2017": (8.58, 11.13), "BPI2019_1": (11.60, 19.65), "Sepsis": (0.087, 0.127), "Hospital_Billing": (1.38, 1.70)}),
    ("Chronos-Bolt-base", "zero-shot", {"BPI2017": (7.62, 10.08), "BPI2019_1": (11.64, 18.86), "Sepsis": (0.098, 0.133), "Hospital_Billing": (1.40, 1.72)}),
    ("Chronos-Bolt-base", "LoRA", {"BPI2017": (7.39, 9.88), "BPI2019_1": (11.55, 19.36), "Sepsis": (0.104, 0.138), "Hospital_Billing": (1.40, 1.72)}),
    ("Chronos-Bolt-base", "full tune", {"BPI2017": (7.52, 10.08), "BPI2019_1": (11.42, 19.44), "Sepsis": (0.088, 0.127), "Hospital_Billing": (1.37, 1.69)}),
    ("Chronos-2", "zero-shot", {"BPI2017": (7.25, 9.85), "BPI2019_1": (11.39, 19.02), "Sepsis": (0.090, 0.128), "Hospital_Billing": (1.39, 1.70)}),
    ("Chronos-2", "full tune", {"BPI2017": (7.89, 10.49), "BPI2019_1": (11.40, 19.09), "Sepsis": (0.104, 0.140), "Hospital_Billing": (1.37, 1.68)}),
    ("MOIRAI-1.1-R-small", "zero-shot", {"BPI2017": (10.24, 13.13), "BPI2019_1": (12.88, 21.48), "Sepsis": (0.088, 0.131), "Hospital_Billing": (1.48, 1.81)}),
    ("MOIRAI-1.1-R-small", "LoRA", {"BPI2017": (9.62, 12.30), "BPI2019_1": (13.04, 21.72), "Sepsis": (0.096, 0.139), "Hospital_Billing": (1.54, 1.86)}),
    ("MOIRAI-1.1-R-small", "full tune", {"BPI2017": (10.21, 13.16), "BPI2019_1": (12.89, 21.50), "Sepsis": (0.088, 0.131), "Hospital_Billing": (1.48, 1.81)}),
    ("MOIRAI-1.1-R-large", "zero-shot", {"BPI2017": (9.29, 11.91), "BPI2019_1": (12.30, 20.79), "Sepsis": (0.086, 0.129), "Hospital_Billing": (1.43, 1.75)}),
    ("MOIRAI-1.1-R-large", "LoRA", {"BPI2017": (8.27, 10.84), "BPI2019_1": (12.27, 19.95), "Sepsis": (0.090, 0.132), "Hospital_Billing": (1.47, 1.79)}),
    ("MOIRAI-1.1-R-large", "full tune", {"BPI2017": (9.25, 11.90), "BPI2019_1": (23.06, 32.29), "Sepsis": (0.114, 0.159), "Hospital_Billing": (1.43, 1.75)}),
]

# --- Entropic Relevance, beat 9 -- transcribed from paper Table 7 ------------
# Sourced fully from the paper (not results/) because: (a) beat 9 is the load-bearing
# ER slide and must match the paper for Q&A defense; (b) results/ ER differs from the
# camera-ready (chronos_2 BPI2017 1.117 vs 1.09) and carries no baseline ER. Lower = better.
# Per dataset: ER mean values. Truth + Training are drawn as reference lines.
ER_REF_ROWS = ["Truth", "Training"]  # reference lines (not bars)
ER_BAR_ROWS = ["Seasonal-Naive", "XGBoost", "Chronos-2", "MOIRAI-2.0", "TimesFM-2.5"]
TABLE7_ER = {
    # row -> {dataset: er_mean}
    "Truth": {"BPI2017": 1.00, "BPI2019_1": 2.00, "Sepsis": 6.27, "Hospital_Billing": 1.86},
    "Training": {"BPI2017": 1.15, "BPI2019_1": 3.89, "Sepsis": 15.75, "Hospital_Billing": 5.83},
    "Seasonal-Naive": {
        "BPI2017": 1.06,
        "BPI2019_1": 2.40,
        "Sepsis": 21.07,
        "Hospital_Billing": 2.17,
    },
    "XGBoost": {"BPI2017": 1.01, "BPI2019_1": 2.39, "Sepsis": 15.48, "Hospital_Billing": 2.12},
    "Chronos-2": {"BPI2017": 1.09, "BPI2019_1": 2.57, "Sepsis": 30.50, "Hospital_Billing": 2.43},
    "MOIRAI-2.0": {"BPI2017": 1.09, "BPI2019_1": 2.57, "Sepsis": 34.21, "Hospital_Billing": 2.52},
    "TimesFM-2.5": {"BPI2017": 1.10, "BPI2019_1": 2.54, "Sepsis": 27.99, "Hospital_Billing": 2.39},
}
# Fitting ratio % (bracketed values in Table 7) — same keys as TABLE7_ER.
TABLE7_FIT = {
    "Truth": {"BPI2017": 100.0, "BPI2019_1": 100.0, "Sepsis": 100.0, "Hospital_Billing": 100.0},
    "Training": {"BPI2017": 99.4, "BPI2019_1": 95.9, "Sepsis": 77.9, "Hospital_Billing": 83.0},
    "Seasonal-Naive": {
        "BPI2017": 99.9,
        "BPI2019_1": 99.3,
        "Sepsis": 39.4,
        "Hospital_Billing": 99.4,
    },
    "XGBoost": {"BPI2017": 100.0, "BPI2019_1": 100.0, "Sepsis": 74.6, "Hospital_Billing": 99.6},
    "Chronos-2": {"BPI2017": 99.7, "BPI2019_1": 98.7, "Sepsis": 13.2, "Hospital_Billing": 98.3},
    "MOIRAI-2.0": {"BPI2017": 99.7, "BPI2019_1": 98.8, "Sepsis": 4.1, "Hospital_Billing": 98.0},
    "TimesFM-2.5": {"BPI2017": 99.7, "BPI2019_1": 98.9, "Sepsis": 17.5, "Hospital_Billing": 98.5},
}
# Single-dataset ER bar chart (beat 9 main): Truth + Training as bars alongside the 5 forecasters.
# Hospital Billing chosen — Training (reuse-the-past) is dramatically worst there, the paper's own
# example (§4.3): every forecast beats freezing the historical process model.
ER_SINGLE_DATASET = "Hospital_Billing"
ER_SINGLE_ROWS = ER_REF_ROWS + ER_BAR_ROWS  # Truth, Training, Seasonal-Naive, XGBoost, 3 TSFMs

# --- output filenames --------------------------------------------------------
OUT = {
    "mae_bars": FIG_OUT / "results-mae-bars.png",
    "drift_xgb": FIG_OUT / "bpi2017-drift-xgb-only.png",
    "drift_tsfm": FIG_OUT / "bpi2017-drift-with-tsfm.png",
    "s5_drift_truth": FIG_OUT / "s5-drift-truth.png",
    "s5_intermittent_truth": FIG_OUT / "s5-intermittent-truth.png",
    "s6_drift_xgb": FIG_OUT / "s6-drift-xgb.png",
    "s6_intermittent_xgb": FIG_OUT / "s6-intermittent-xgb.png",
    # S6 — same two panels with a baked-in amber attention mark (static, no click): box over the
    # drift back-half (XGBoost stuck high), arrow at an XGBoost overshoot peak. Separate files so
    # the plain s6-*.png stay clean for S14's pre-click "before" state.
    "s6_drift_xgb_box": FIG_OUT / "s6-drift-xgb-box.png",
    "s6_intermittent_xgb_arrow": FIG_OUT / "s6-intermittent-xgb-arrow.png",
    # S14 — drift+sparsity callback with MOIRAI-2.0 revealed + baked-in amber attention marks
    # (same axes/ylim as the S6 panels so the click-swap overlays exactly).
    "s14_drift_tsfm": FIG_OUT / "s14-drift-tsfm.png",
    "s14_intermittent_tsfm": FIG_OUT / "s14-intermittent-tsfm.png",
    "ft_slope": FIG_OUT / "ft-slope.png",
    "er_bars": FIG_OUT / "er-bars.png",
    "er_single": FIG_OUT / "er-hospital-billing.png",
    "mae_full": FIG_OUT / "mae-full.png",
    "rmse_full": FIG_OUT / "rmse-full.png",
    "ft_table": FIG_OUT / "ft-table.png",
    "df_complexity": FIG_OUT / "df-complexity-radar.png",
    # S7 — highlighted complexity radar: navy min–max band across the 4 logs with the
    # 3 harder-than-benchmark axes (Transition/Shifting/Non-Gaussianity) emphasised in
    # amber. The plain full per-log radar above stays the B4 backup.
    "s7_complexity": FIG_OUT / "s7-complexity-radar.png",
}

# --- S7 highlighted complexity radar -----------------------------------------
# The three metrics the paper reports as HIGHER than the 21 public benchmarks
# (Li et al. 2025): emphasise these axes in amber. main.tex line 211.
COMPLEXITY_HIGHLIGHT = ["Transition", "Shifting", "Non-Gaussianity"]
