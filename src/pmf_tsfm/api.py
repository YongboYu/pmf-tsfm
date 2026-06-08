"""Agent-clean, Gradio-free entry points over the core forecasting pipeline (issue #132).

This is the **shared seam** the dev-ready MCP (#133) and Docker (#134) artifacts build on:
plain Python functions that let a user run the *real* ``src/pmf_tsfm`` pipeline on their
**own** process log — a raw ``.xes`` (auto-converted to the daily DF-relation series) or a
prepared DF-relation ``.parquet``.

Scope (locked, ADR-0004): **zero-shot holdout backtest** only — no training. The natural
60/20/20 split holds out the tail of the series; the existing expanding-window zero-shot
pipeline forecasts the held-out region from prior history and scores it (genuine accuracy:
MAE/RMSE, plus Entropic Relevance when an XES log is available).

These functions **compose the existing Hydra configs** and **call the existing cores**
(``inference.run_inference``, ``evaluate.evaluate_single``, ``er.evaluate_er.run_er_evaluation``)
rather than reimplementing anything — so the numbers match the paper/CLI.

Import-light by design: heavy deps (torch via the core modules, the model libraries) are
imported **inside** the functions, so ``import pmf_tsfm.api`` pulls no model library, no
Gradio, and triggers no CUDA init or weight download.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

# Repo layout: this file is src/pmf_tsfm/api.py → parents[2] is the repo root.
_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"

_XES_SUFFIXES = (".xes.gz", ".xes")


def list_models() -> list[str]:
    """Available model config groups, e.g. ``"chronos/chronos2"``, ``"moirai/2_0_small"``.

    These are exactly the values accepted by the ``model=`` argument of the forecast
    functions (the Hydra config-group path under ``configs/model/``). Pure and cheap —
    no model import.
    """
    model_dir = _CONFIGS_DIR / "model"
    return sorted(str(p.relative_to(model_dir).with_suffix("")) for p in model_dir.rglob("*.yaml"))


def _is_xes(path: Path) -> bool:
    return path.name.lower().endswith(_XES_SUFFIXES)


def _resolve_input(
    input_path: Path,
    workdir: Path,
    log_dir: str,
    train_end: int | None,
    val_end: int | None,
) -> tuple[dict, bool]:
    """Build the in-code data config for the run; returns ``(data_cfg, has_xes)``.

    ``.xes``/``.xes.gz`` → convert to the daily DF-relation parquet (and stage the log
    for ER). ``.parquet`` → use as-is. Caller-supplied ``train_end``/``val_end`` override
    the derived 60/20/20 split.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if _is_xes(input_path):
        from pmf_tsfm.data.log_to_series import prepare_series_from_log

        data = prepare_series_from_log(input_path, workdir / "series.parquet", log_dir=log_dir)
        has_xes = True
    elif input_path.suffix.lower() == ".parquet":
        import pandas as pd

        n = len(pd.read_parquet(input_path, columns=[]))
        data = {
            "name": input_path.stem,
            "path": str(input_path.resolve()),
            "train_end": int(n * 0.6),
            "val_end": int(n * 0.8),
        }
        has_xes = False
    else:
        raise ValueError(
            f"Unsupported input '{input_path.name}'. Expected .xes/.xes.gz or .parquet."
        )

    if train_end is not None:
        data["train_end"] = int(train_end)
    if val_end is not None:
        data["val_end"] = int(val_end)
    return data, has_xes


def _run(
    input_path: str | Path,
    model: str,
    horizon: int,
    device: str,
    train_end: int | None,
    val_end: int | None,
    workdir: str | Path | None,
):
    """Compose the config, resolve the input, and run inference. Returns ``(cfg, result)``."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    input_path = Path(input_path)
    workdir = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="pmf_api_"))
    workdir.mkdir(parents=True, exist_ok=True)

    # Compose the same config tree the CLI uses. Overriding paths.root_dir to the scratch
    # workdir isolates all outputs/log_dir/processed_dir per run AND replaces the
    # ${hydra:runtime.cwd} resolver (which is unavailable under bare compose) with a literal.
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIGS_DIR)):
        cfg = compose(
            config_name="inference",
            overrides=[
                f"model={model}",
                f"device={device}",
                f"prediction_length={horizon}",
                "logger=disabled",
                f"paths.root_dir={workdir}",
            ],
        )

    data, has_xes = _resolve_input(input_path, workdir, cfg.paths.log_dir, train_end, val_end)
    OmegaConf.set_struct(cfg, False)
    cfg.data = OmegaConf.create(data)

    from pmf_tsfm.inference import run_inference

    result = run_inference(cfg)
    return cfg, result, has_xes


def _artifact_paths(cfg, output_path: Path) -> tuple[Path, Path, list[str] | None]:
    """Prediction/quantile npy paths + feature names (from the metadata inference wrote)."""
    prefix = f"{cfg.data.name}_{cfg.model.name}"
    predictions_path = output_path / f"{prefix}_predictions.npy"
    quantiles_path = output_path / f"{prefix}_quantiles.npy"

    feature_names: list[str] | None = None
    meta_path = output_path / f"{prefix}_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            feature_names = json.load(f).get("feature_names")
    return predictions_path, quantiles_path, feature_names


def forecast_only(
    input_path: str | Path,
    model: str = "chronos/chronos2",
    horizon: int = 7,
    device: str = "cpu",
    train_end: int | None = None,
    val_end: int | None = None,
    workdir: str | Path | None = None,
) -> dict:
    """Zero-shot forecast over the held-out tail; return predictions, skip scoring.

    See :func:`forecast_backtest` for the argument and return shape; ``metrics`` is
    ``None`` here.
    """
    cfg, result, _ = _run(input_path, model, horizon, device, train_end, val_end, workdir)
    output_path = Path(result["output_path"])
    predictions_path, quantiles_path, feature_names = _artifact_paths(cfg, output_path)
    return {
        "predictions_path": str(predictions_path),
        "quantiles_path": str(quantiles_path),
        "metrics": None,
        "feature_names": feature_names,
        "n_windows": int(result["predictions"].shape[0]),
        "model": cfg.model.name,
        "horizon": int(cfg.prediction_length),
    }


def forecast_backtest(
    input_path: str | Path,
    model: str = "chronos/chronos2",
    horizon: int = 7,
    device: str = "cpu",
    train_end: int | None = None,
    val_end: int | None = None,
    workdir: str | Path | None = None,
    compute_er: bool = True,
) -> dict:
    """Zero-shot holdout backtest on a user's process log; forecast + score.

    Args:
        input_path: Raw ``.xes``/``.xes.gz`` log (auto-converted) or a prepared
                    DF-relation ``.parquet``.
        model:      Model config group, e.g. ``"chronos/chronos2"`` (see :func:`list_models`).
        horizon:    Forecast/holdout length in days.
        device:     ``"cpu"``, ``"cuda"``, or ``"mps"`` (falls back to CPU if unavailable).
        train_end/val_end: Override the derived 60/20/20 split indices.
        workdir:    Scratch dir for the run (default: a fresh temp dir).
        compute_er: Compute Entropic Relevance (only possible for ``.xes`` input).

    Returns:
        ``{predictions_path, quantiles_path, metrics, feature_names, n_windows, model,
        horizon}`` where ``metrics = {mae, mae_std, rmse, rmse_std, er}``. ``er`` is
        ``None`` for ``.parquet`` input (no log to build truth/training DFGs) or when
        ``compute_er=False``.
    """
    cfg, result, has_xes = _run(input_path, model, horizon, device, train_end, val_end, workdir)
    output_path = Path(result["output_path"])
    predictions_path, quantiles_path, feature_names = _artifact_paths(cfg, output_path)

    from pmf_tsfm.evaluate import evaluate_single

    summary = evaluate_single(output_path, cfg.data.name, cfg.model.name, save=True)["summary"]
    metrics = {
        "mae": summary["MAE_mean"],
        "mae_std": summary["MAE_std"],
        "rmse": summary["RMSE_mean"],
        "rmse_std": summary["RMSE_std"],
        "er": None,
    }

    if compute_er and has_xes:
        from omegaconf import OmegaConf

        from pmf_tsfm.er.evaluate_er import run_er_evaluation

        # run_er_evaluation joins cfg.task into a path expecting a string, but the
        # inference config makes cfg.task a node. Fully resolve, then coerce task to its
        # name and pin a concrete ER output dir so no lazy interpolation remains.
        task_name = cfg.task.name
        er_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        er_cfg.task = task_name
        er_cfg.output_dir = str(
            Path(cfg.paths.output_dir) / "er" / task_name / cfg.data.name / cfg.model.name
        )
        er = run_er_evaluation(er_cfg)
        metrics["er"] = er["summary"]["pred_er_mean"]

    return {
        "predictions_path": str(predictions_path),
        "quantiles_path": str(quantiles_path),
        "metrics": metrics,
        "feature_names": feature_names,
        "n_windows": int(result["predictions"].shape[0]),
        "model": cfg.model.name,
        "horizon": int(cfg.prediction_length),
    }
