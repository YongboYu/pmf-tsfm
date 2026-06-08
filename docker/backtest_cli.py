#!/usr/bin/env python
"""Thin argparse CLI over ``pmf_tsfm.api.forecast_backtest`` for the Docker image (#134).

Docker-only shim — kept out of the core package so ``src/`` stays import-clean. It does
**not** reimplement anything: it calls ``forecast_backtest`` (the same composed Hydra
pipeline the CLI runs), so the printed numbers match the paper/CLI.

    backtest --input /data/processed_logs/sepsis.xes --model chronos/chronos2 --horizon 7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="backtest",
        description="Zero-shot holdout backtest on your own process log (XES or DF-relation parquet).",
    )
    p.add_argument(
        "--input",
        required=True,
        help="Raw .xes/.xes.gz log (auto-converted to the daily DF-relation series) or a prepared .parquet.",
    )
    p.add_argument(
        "--model",
        default="chronos/chronos2",
        help="Model config group, e.g. chronos/chronos2, moirai/2_0_small (run `list-models` to see all).",
    )
    p.add_argument(
        "--horizon", type=int, default=7, help="Forecast / holdout length in days (default: 7)."
    )
    p.add_argument(
        "--device",
        default=os.environ.get("PMF_DEVICE", "cpu"),
        help="cpu | cuda | mps (falls back to cpu if unavailable). Default: $PMF_DEVICE or cpu.",
    )
    p.add_argument(
        "--train-end", type=int, default=None, help="Override the derived 60%% split index."
    )
    p.add_argument(
        "--val-end", type=int, default=None, help="Override the derived 80%% split index."
    )
    p.add_argument(
        "--no-er",
        action="store_true",
        help="Skip Entropic Relevance (ER is only computable for .xes input anyway).",
    )
    p.add_argument(
        "--output",
        default=os.environ.get("PMF_OUTPUT", "/work/outputs"),
        help="Run/artifact dir (mount a volume to keep predictions). Default: $PMF_OUTPUT or /work/outputs.",
    )
    p.add_argument("--json", action="store_true", help="Print the raw result dict as JSON only.")
    return p


def _fmt(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "n/a"
    return f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f}"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        print('       mount your data, e.g. -v "$PWD/data:/data"', file=sys.stderr)
        return 1

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Imported here so --help stays instant and no model library loads on an arg error.
    from pmf_tsfm.api import forecast_backtest

    result = forecast_backtest(
        input_path=input_path,
        model=args.model,
        horizon=args.horizon,
        device=args.device,
        train_end=args.train_end,
        val_end=args.val_end,
        workdir=args.output,
        compute_er=not args.no_er,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    m = result.get("metrics") or {}
    n_feat = len(result["feature_names"]) if result.get("feature_names") else "?"
    er = m.get("er")

    print("── pmf-tsfm backtest ─────────────────────────────────────────")
    print(f"  input        : {input_path}")
    print(f"  model        : {result['model']}")
    print(f"  horizon      : {result['horizon']} day(s)")
    print(f"  windows      : {result['n_windows']}")
    print(f"  DF relations : {n_feat}")
    print("── metrics (zero-shot holdout) ───────────────────────────────")
    print(f"  MAE          : {_fmt(m.get('mae'), m.get('mae_std'))}")
    print(f"  RMSE         : {_fmt(m.get('rmse'), m.get('rmse_std'))}")
    if er is not None:
        print(f"  Entropic Rel.: {er:.4f}")
    else:
        print("  Entropic Rel.: n/a (parquet input or --no-er)")
    print("── artifacts ─────────────────────────────────────────────────")
    print(f"  predictions  : {result['predictions_path']}")
    print(f"  quantiles    : {result['quantiles_path']}")
    print("──────────────────────────────────────────────────────────────")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
