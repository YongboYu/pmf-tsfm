"""Normalize the nested ``outputs/`` result tree into one long-format table.

The experiment results live as nested per-(task, dataset, model) JSON:

    outputs/{task}/{DATASET}/{model}/{DATASET}_{model}_metrics.json    # summary + per_feature
    outputs/{task}/{DATASET}/{model}/{DATASET}_{model}_metadata.json   # run context
    outputs/er/{task}/{DATASET}/{model}/{DATASET}_{model}_er.json      # ER summary + per-window
    outputs/er/{task}/{DATASET}/{DATASET}_er_all_summary.json          # ER baselines + elapsed_s

This module folds them into one long table with a ``level`` discriminator so the
benchmark is queryable and renders in the HF Data Studio viewer:

    level=summary      one row per (dataset, model, task, metric); ``value`` is the
                       across-window mean, ``std`` its std where the source has one.
    level=per_feature  MAE / RMSE per directly-follows relation (``feature``).
    level=per_window   ER metrics per rolling ER window (``window``, ``n_traces``).

Baselines that are identical across models within a (dataset, task) — the ground-truth
DFG's ER (``truth_er``) and the training-window ER (``training_er``) — are emitted once
as reference rows with ``model="reference"``.

**Provenance:** point ``--outputs-dir`` at the canonical HPC (VSC) tree, which the repo
syncs to ``data/hpc_sync/outputs/``. The top-level local ``outputs/`` is an independent
run whose fine-tuned numbers differ; it is *not* the publishable source.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_PROJECT = "pmf-tsfm"
DEFAULT_PAPER = "arXiv:2512.07624"
DEFAULT_OUTPUTS_DIR = "data/hpc_sync/outputs"
TASKS: tuple[str, ...] = ("zero_shot", "lora_tune", "full_tune")

# Fixed long-format schema. Every emitted row carries all of these keys.
COLUMNS: list[str] = [
    "project",
    "paper",
    "dataset",
    "model",
    "family",
    "task",
    "level",  # summary | per_feature | per_window
    "metric",  # MAE | RMSE | pred_er | pred_fitting_ratio | truth_er | training_er
    "value",
    "std",
    "feature",  # DF relation "A -> B" for per_feature rows
    "window",  # ER window label for per_window rows
    "n_traces",  # trace count for that ER window
    "horizon",
    "n_windows",
    "n_features",
    "runtime_s",  # ER elapsed_s (dataset-level); inference runtime is not on disk
    "device",
    "precision",
    "context_length",
    "base_model_id",
    "seed",
    "commit",  # repo commit the export was generated from
]

# ER metrics: metric name -> (mean key, std key or None) in the ER "summary" block.
_ER_MODEL_METRICS = {
    "pred_er": ("pred_er_mean", "pred_er_std"),
    "pred_fitting_ratio": ("pred_fitting_ratio_mean", None),
}
_ER_REFERENCE_METRICS = {
    "truth_er": ("truth_er_mean", "truth_er_std"),
    "training_er": ("training_er_mean", "training_er_std"),
}


def normalize_dataset(name: str) -> str:
    """Match the repo's config convention: dataset IDs are lowercase (``BPI2017`` -> ``bpi2017``)."""
    return name.lower()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    """A real, usable number. Invalid ER windows carry ``pred_er: NaN`` in the source JSON."""
    if value is None:
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _std_or_none(value: Any) -> Any:
    """Keep a std only if it is a finite number; otherwise drop it to null."""
    return value if _finite(value) else None


def _task_name(metadata: dict[str, Any], cfg: dict[str, Any]) -> str | None:
    """Resolve the task to a string.

    The top-level ``task`` is always the string id (``"lora_tune"``); ``config.task`` is
    the task *config group* — a string in zero-shot but a nested dict in LoRA/full-tune.
    """
    top = metadata.get("task")
    if isinstance(top, str):
        return top
    inner = cfg.get("task")
    if isinstance(inner, dict):
        return inner.get("name")
    return inner if isinstance(inner, str) else None


def _row(ctx: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Build a full row: every schema column present, ``None`` unless supplied."""
    row = dict.fromkeys(COLUMNS)
    for key in ("project", "paper", "dataset", "model", "family", "task", "commit"):
        row[key] = ctx.get(key)
    for key in (
        "horizon",
        "n_windows",
        "n_features",
        "device",
        "precision",
        "context_length",
        "base_model_id",
        "seed",
    ):
        if key in ctx:
            row[key] = ctx[key]
    row.update(overrides)
    return row


def run_context(metadata: dict[str, Any], *, dataset_fallback: str | None = None) -> dict[str, Any]:
    """Extract the per-run context shared by every row of one run, from its metadata JSON."""
    cfg = metadata.get("config", {})
    model = cfg.get("model", {})
    data = cfg.get("data", {})
    dm = metadata.get("data_metadata", {})

    dataset_name = data.get("name") or dm.get("dataset_name") or dataset_fallback or ""
    return {
        "dataset": normalize_dataset(dataset_name),
        "model": model.get("name"),
        "family": model.get("family"),
        "task": _task_name(metadata, cfg),
        "device": cfg.get("device"),
        "precision": model.get("torch_dtype"),
        "seed": cfg.get("seed"),
        "context_length": cfg.get("context_length"),
        "base_model_id": model.get("id"),
        "horizon": cfg.get("prediction_length") or dm.get("prediction_length"),
        "n_features": dm.get("num_features"),
        "n_windows": dm.get("test_num_sequences"),
    }


def rows_from_metrics(metrics: dict[str, Any], ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Summary (mean±std) and per-feature (per DF relation) MAE/RMSE rows."""
    rows: list[dict[str, Any]] = []
    summary = metrics.get("summary", {})
    for metric in ("MAE", "RMSE"):
        mean = summary.get(f"{metric}_mean")
        if not _finite(mean):
            continue
        rows.append(
            _row(
                ctx,
                level="summary",
                metric=metric,
                value=mean,
                std=_std_or_none(summary.get(f"{metric}_std")),
            )
        )
    for feature, values in metrics.get("per_feature", {}).items():
        for metric in ("MAE", "RMSE"):
            value = values.get(metric)
            if not _finite(value):
                continue
            rows.append(_row(ctx, level="per_feature", metric=metric, value=value, feature=feature))
    return rows


def rows_from_er(er: dict[str, Any], ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-model ER summary rows and per-window ER rows (pred_er, pred_fitting_ratio)."""
    rows: list[dict[str, Any]] = []
    summary = er.get("summary", {})
    for metric, (mean_key, std_key) in _ER_MODEL_METRICS.items():
        value = summary.get(mean_key)
        if not _finite(value):
            continue
        std = _std_or_none(summary.get(std_key)) if std_key else None
        rows.append(_row(ctx, level="summary", metric=metric, value=value, std=std))
    for window in er.get("windows", []):
        label = window.get("window")
        n_traces = window.get("n_traces")
        for metric in ("pred_er", "pred_fitting_ratio"):
            value = window.get(metric)
            if not _finite(value):
                continue
            rows.append(
                _row(
                    ctx,
                    level="per_window",
                    metric=metric,
                    value=value,
                    window=label,
                    n_traces=n_traces,
                )
            )
    return rows


def reference_rows(er_all: dict[str, Any], ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """``truth_er`` / ``training_er`` baselines (identical across models) — emitted once."""
    ref_ctx = {**ctx, "model": "reference", "family": "reference", "base_model_id": None}
    runtime_s = er_all.get("elapsed_s")
    n_windows = er_all.get("n_windows")
    rows: list[dict[str, Any]] = []
    for metric, (mean_key, std_key) in _ER_REFERENCE_METRICS.items():
        value = er_all.get(mean_key)
        if not _finite(value):
            continue
        rows.append(
            _row(
                ref_ctx,
                level="summary",
                metric=metric,
                value=value,
                std=_std_or_none(er_all.get(std_key)),
                runtime_s=runtime_s,
                n_windows=n_windows,
            )
        )
    return rows


def collect_rows(
    outputs_dir: str | Path,
    *,
    tasks: tuple[str, ...] = TASKS,
    project: str = DEFAULT_PROJECT,
    paper: str = DEFAULT_PAPER,
    commit: str | None = None,
    exclude_models: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Walk ``outputs_dir`` and flatten every run into long-format rows.

    ``exclude_models`` skips model directories by name (e.g. ``moirai_1_1_base``, which
    the paper's 12 zero-shot variants omit). Dataset-level ER reference rows are kept
    regardless of the model filter.
    """
    outputs_dir = Path(outputs_dir)
    excluded = set(exclude_models)
    stamp = {"project": project, "paper": paper, "commit": commit}
    rows: list[dict[str, Any]] = []

    for task in tasks:
        task_dir = outputs_dir / task
        if not task_dir.is_dir():
            continue
        for ds_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            for model_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
                if model_dir.name in excluded:
                    continue
                metrics_path = next(model_dir.glob("*_metrics.json"), None)
                metadata_path = next(model_dir.glob("*_metadata.json"), None)
                if metrics_path is None or metadata_path is None:
                    continue
                metrics = _load_json(metrics_path)
                metadata = _load_json(metadata_path)

                ctx = {**run_context(metadata, dataset_fallback=ds_dir.name), **stamp}
                # metrics summary is authoritative for the shape counts
                summary = metrics.get("summary", {})
                if "num_sequences" in summary:
                    ctx["n_windows"] = summary["num_sequences"]
                if "num_features" in summary:
                    ctx["n_features"] = summary["num_features"]
                if "prediction_length" in summary:
                    ctx["horizon"] = summary["prediction_length"]

                rows.extend(rows_from_metrics(metrics, ctx))

                er_path = (
                    outputs_dir
                    / "er"
                    / task
                    / ds_dir.name
                    / model_dir.name
                    / f"{ds_dir.name}_{model_dir.name}_er.json"
                )
                if er_path.exists():
                    rows.extend(rows_from_er(_load_json(er_path), ctx))

        # dataset-level ER baselines (once per dataset that has an ER aggregate)
        er_task_dir = outputs_dir / "er" / task
        if er_task_dir.is_dir():
            for ds_dir in sorted(p for p in er_task_dir.iterdir() if p.is_dir()):
                summary_path = ds_dir / f"{ds_dir.name}_er_all_summary.json"
                if not summary_path.exists():
                    continue
                ref_ctx = {**stamp, "dataset": normalize_dataset(ds_dir.name), "task": task}
                rows.extend(reference_rows(_load_json(summary_path), ref_ctx))

    return rows


def build_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble rows into a DataFrame with columns in the fixed schema order."""
    return pd.DataFrame(rows, columns=COLUMNS)


def _git_commit(repo_dir: Path) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 — fixed argv, no shell, resolved git path
            [git, "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _summarize(df: pd.DataFrame) -> str:
    by_level = df["level"].value_counts().to_dict()
    n_runs = df[df["model"] != "reference"].groupby(["task", "dataset", "model"]).ngroups
    return (
        f"{len(df):,} rows from {n_runs} runs · "
        f"levels: {by_level} · "
        f"datasets: {sorted(df['dataset'].unique())} · "
        f"tasks: {sorted(df['task'].dropna().unique())}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize the outputs/ result tree into one long-format Parquet."
    )
    parser.add_argument(
        "--outputs-dir",
        default=DEFAULT_OUTPUTS_DIR,
        help=f"Root of the result tree (default: {DEFAULT_OUTPUTS_DIR}, the canonical HPC sync).",
    )
    parser.add_argument(
        "--out",
        default="data/exports/pmf_tsfm_results.parquet",
        help="Output Parquet path.",
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), help="Tasks to include.")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--paper", default=DEFAULT_PAPER)
    parser.add_argument(
        "--commit",
        default=None,
        help="Provenance commit to stamp; defaults to the current repo's short HEAD.",
    )
    parser.add_argument("--csv", action="store_true", help="Also write a sibling .csv.")
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=[],
        metavar="MODEL",
        help="Model dir names to skip (e.g. moirai_1_1_base to match the paper's 12 variants).",
    )
    args = parser.parse_args(argv)

    commit = args.commit or _git_commit(Path(__file__).resolve().parent)
    rows = collect_rows(
        args.outputs_dir,
        tasks=tuple(args.tasks),
        project=args.project,
        paper=args.paper,
        commit=commit,
        exclude_models=tuple(args.exclude_models),
    )
    df = build_dataframe(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out}")
    if args.csv:
        csv_path = out.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")
    print(_summarize(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
