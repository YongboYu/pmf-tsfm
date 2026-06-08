# pmf-tsfm — self-host Docker image

Run the **core** Process Model Forecasting pipeline (zero-shot directly-follows forecasting with
Time Series Foundation Models) on **your own** process logs, with no caps. This is the BYO-data
counterpart to the capped Gradio demo: same pipeline, same numbers as the paper/CLI.

The image bundles `pmf_tsfm.api` (the Gradio-free seam), the Hydra CLIs, and a thin `backtest`
command. Default models are **Chronos** and **Moirai**; **TimesFM** is an opt-in build.

## Build

The build context is the **repo root** and the Dockerfile lives in `docker/`:

```bash
docker build -f docker/Dockerfile -t pmf-tsfm .
```

This produces the default **Chronos + Moirai** image (fast — both are regular dependencies).

> **Why uv, not pip.** The build uses `uv sync --frozen` against `uv.lock`. The
> `[tool.uv] override-dependencies = ["torch>=2.8.0"]` in `pyproject.toml` resolves the
> `uni2ts` ↔ `chronos` numpy/torch conflict (uni2ts pins `torch<2.5`, which is overly
> conservative). That resolution **only holds under uv** — a plain `pip install` silently
> reintroduces a numpy/torch clash. Don't swap in pip.

### TimesFM (opt-in)

TimesFM v2.5 is a large, GitHub-pinned zip (no PyPI release) and needs Python ≥ 3.11. It is **not**
in the default image. To include it:

```bash
docker build -f docker/Dockerfile --build-arg INSTALL_TIMESFM=1 -t pmf-tsfm:timesfm .
```

This adds the `timesfm_v25` extra and unlocks `--model timesfm/2_5_200m` (and the other
`timesfm/*` groups). Expect a longer, more fragile build.

## Quick start — `backtest` (BYO data)

Zero-shot holdout backtest on your own log. A raw `.xes`/`.xes.gz` is auto-converted to the daily
DF-relation series (and used for Entropic Relevance); a prepared DF-relation `.parquet` is used
as-is. The natural 60/20/20 split holds out the tail and the pipeline forecasts + scores it.

```bash
docker run --rm \
  -v "$PWD/data:/data" \
  -v pmf-cache:/cache \
  -v "$PWD/outputs:/work/outputs" \
  pmf-tsfm backtest --input /data/processed_logs/sepsis.xes --model chronos/chronos2
```

Prints a forecast summary plus **MAE / RMSE / Entropic Relevance**, e.g.:

```
── pmf-tsfm backtest ─────────────────────────────────────────
  input        : /data/processed_logs/sepsis.xes
  model        : chronos2
  horizon      : 7 day(s)
  windows      : ...
  DF relations : ...
── metrics (zero-shot holdout) ───────────────────────────────
  MAE          : 0.xxxx ± 0.xxxx
  RMSE         : 0.xxxx ± 0.xxxx
  Entropic Rel.: 0.xxxx
── artifacts ─────────────────────────────────────────────────
  predictions  : /work/outputs/.../<log>_<model>_predictions.npy
  quantiles    : /work/outputs/.../<log>_<model>_quantiles.npy
──────────────────────────────────────────────────────────────
```

`backtest` flags: `--input` (required), `--model` (default `chronos/chronos2`), `--horizon`
(default 7), `--device` (`cpu`/`cuda`/`mps`), `--train-end`/`--val-end` (override the split),
`--no-er` (skip Entropic Relevance), `--output` (artifact dir, default `/work/outputs`),
`--json` (machine-readable output). List the model groups with `docker run --rm pmf-tsfm list-models`.

> **Cold start.** Chronos-2 weights download on first run from `s3://autogluon/chronos-2/`
> (needs network; `boto3` is bundled) — a few minutes. Mount the `pmf-cache` volume (below) so the
> weights persist and later runs skip the download. The image unifies every model family's cache
> under `/cache` (`HF_HOME=/cache/huggingface` for HF-hub models; `XGD_CACHE_HOME=/cache` →
> `/cache/chronos/...` for Chronos-2's S3 weights), so a single `-v pmf-cache:/cache` covers all of
> them. ER rendering uses `graphviz` (installed in the image).

## Hydra CLIs

The full research entrypoints are dispatched by the first argument; remaining args are Hydra
overrides:

```bash
docker run --rm -v "$PWD/data:/data" -v pmf-cache:/cache \
  pmf-tsfm inference data=bpi2017 model=chronos/chronos2 device=cpu
```

Available commands: `inference`, `evaluate`, `evaluate_er`, `preprocess`, `train`, plus
`list-models`, `help`, and `mcp` (see below). Dataset IDs are lowercase (`bpi2017`, `bpi2019_1`,
`sepsis`, `hospital_billing`).

## Volumes

| Mount | Purpose |
| --- | --- |
| `-v "$PWD/data:/data"` | Your input logs (XES / parquet). The repo's bundled logs are **not** baked into the image. |
| `-v pmf-cache:/cache` | Model-weight cache for all families (`HF_HOME=/cache/huggingface` + Chronos-2's `/cache/chronos`). Persists downloads across runs. |
| `-v "$PWD/outputs:/work/outputs"` | Run artifacts (predictions, quantiles, metrics). |
| `-v "$PWD/configs/data:/app/configs/data"` | Optional: drop in a custom `configs/data/<mylog>.yaml` for the Hydra path. |

## CPU vs GPU

The image defaults to **CPU** (`PMF_DEVICE=cpu`). The default image ships the CPU torch build, so
GPU requires either a CUDA-torch variant or building on an `nvidia/cuda` base. Once you have a
GPU-capable image, run with the NVIDIA runtime and select the device:

```bash
docker run --rm --gpus all -v "$PWD/data:/data" -v pmf-cache:/cache \
  pmf-tsfm backtest --input /data/processed_logs/sepsis.xes --model chronos/chronos2 --device cuda
```

## Models

- **Chronos** (default): `chronos/chronos2`, `chronos/bolt_{tiny,mini,small,base}`
- **Moirai** (default): `moirai/{1_1_small,1_1_base,1_1_large,2_0_small,moe_base}`
- **TimesFM** (opt-in, `INSTALL_TIMESFM=1`): `timesfm/{1_0_200m,2_0_500m,2_5_200m}`

## MCP

`docker run pmf-tsfm mcp` launches the FastMCP server — but only once the **MCP track (#133,
`mcp/server.py`)** is merged and the image is rebuilt. Until then the arm is wired and exits with a
clear message; it is not bundled.
