<div align="center">

# PMF-TSFM

**Process Model Forecasting with Time Series Foundation Models**

[![CI](https://img.shields.io/github/actions/workflow/status/YongboYu/pmf-tsfm/ci.yml?branch=main&style=flat-square&logo=github&labelColor=2b3137)](https://github.com/YongboYu/pmf-tsfm/actions/workflows/ci.yml)
[![CodeQL](https://img.shields.io/github/actions/workflow/status/YongboYu/pmf-tsfm/codeql.yml?branch=main&label=CodeQL&style=flat-square&logo=github&labelColor=2b3137)](https://github.com/YongboYu/pmf-tsfm/actions/workflows/codeql.yml)
[![codecov](https://img.shields.io/codecov/c/gh/YongboYu/pmf-tsfm?style=flat-square&logo=codecov&logoColor=white&labelColor=2b3137)](https://codecov.io/gh/YongboYu/pmf-tsfm)
[![Python](https://img.shields.io/badge/python-3.10+-3776ab.svg?style=flat-square&logo=python&logoColor=white&labelColor=2b3137)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg?style=flat-square&labelColor=2b3137)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2512.07624-b31b1b.svg?style=flat-square&labelColor=2b3137)](https://arxiv.org/abs/2512.07624)
[![Dataset](https://img.shields.io/badge/dataset-Zenodo-007afc.svg?style=flat-square&logo=zenodo&logoColor=white&labelColor=2b3137)](https://zenodo.org/records/18327515)

</div>

---

Systematic evaluation of Time Series Foundation Models (TSFMs) for Process Model Forecasting (PMF), predicting how directly-follows (DF) relations in a process evolve over time. The repository benchmarks Chronos, Moirai, and TimesFM across zero-shot, LoRA, and full fine-tuning settings on four real-world event logs, using MAE/RMSE alongside Entropic Relevance as a process-aware conformance metric.

## At a Glance

- Zero-shot coverage: 13 TSFM variants across Chronos, Moirai, and TimesFM.
- Fine-tuning coverage: LoRA for Chronos-Bolt and Moirai-1.1; full fine-tuning for Chronos-Bolt, Chronos-2, and Moirai-1.1.
- Data assets: daily DF-count time series in Parquet and XES logs for Entropic Relevance evaluation.
- Outputs: predictions under `outputs/{task}/{dataset}/{model}/` and checkpoints/adapters under `results/{task}/{dataset}/{model}/`.
- Orchestration: [Hydra](https://hydra.cc/)-driven Python entry points plus local orchestration scripts and [VSC](https://www.vscentrum.be/) HPC helpers.

## Supported Models

| Family | Variants |
|--------|----------|
| **Chronos** | Bolt Tiny, Bolt Mini, Bolt Small, Bolt Base, Chronos-2 |
| **Moirai** | 1.1 Small/Base/Large, 2.0 Small, MoE Base |
| **TimesFM** | 1.0-200M, 2.0-500M, 2.5-200M |

- LoRA experiments in this repo cover `chronos/bolt_small`, `chronos/bolt_base`, `moirai/1_1_small`, and `moirai/1_1_large`.
- Full fine-tuning covers `chronos/bolt_small`, `chronos/bolt_base`, `chronos/chronos2`, `moirai/1_1_small`, and `moirai/1_1_large`.

## Datasets

Four process mining event logs from the BPI Challenge and healthcare domains:

| Dataset | Description | Cases | DFs |
|---------|-------------|------:|----:|
| **BPI2017** | Loan application process | 40,229 | 21 |
| **BPI2019_1** | Purchase order process (3-way match) | 197,521 | 149 |
| **Sepsis** | Sepsis clinical pathway | 999 | 135 |
| **Hospital Billing** | Hospital billing process | 78,828 | 73 |

The experiment data assets are published on [Zenodo](https://zenodo.org/records/18327515). After extraction, the archive is organized as:

```text
data/
├── raw_logs/         # original XES logs from source benchmarks
├── processed_logs/   # processed XES logs used by ER evaluation
├── time_series/      # daily DF-count Parquet files used by inference/training
└── metadata/         # release metadata and preprocessing statistics
```

See [Data Setup](#data-setup) for download commands. If you want the upstream preprocessing workflow and source-log preparation details, see [pmf-benchmark](https://github.com/YongboYu/pmf-benchmark).

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/). The `timesfm_v25` extra requires Python 3.11+ because the pinned TimesFM 2.5 package is only installed on 3.11+.

```bash
# Clone and install
git clone https://github.com/YongboYu/pmf-tsfm.git
cd pmf-tsfm
uv sync

# Optional model extras
uv sync --extra timesfm_v25      # TimesFM 2.5
uv sync --extra timesfm_legacy   # TimesFM 1.0 / 2.0

# Optional dev tools
uv sync --group dev

# Optional: activate the uv-managed environment for plain `python -m ...` usage
source .venv/bin/activate
```

Examples below assume either the `.venv` is activated or commands are prefixed with `uv run`.
See [Tested Environments](#tested-environments) for the macOS (Apple Silicon / MPS), Linux (NVIDIA GPUs / CUDA), and VSC wICE HPC cluster setups used with this repo.

Create the local config files that are meant to stay machine-specific:

```bash
cp .env.example .env
cp configs/local/default.yaml.example configs/local/default.yaml
```

- `.env` is used for environment variables such as `PROJECT_ROOT`, `WANDB_API_KEY`, `CUDA_VISIBLE_DEVICES`, and `TIMESFM_V1_PATH`.
- `configs/local/default.yaml` is optional and useful for Hydra-only overrides such as `device`, `training.num_workers`, or a local [Weights & Biases](https://wandb.ai/site) entity.

## Data Setup

Download from the Zenodo record page: <https://zenodo.org/records/18327515>.

```bash
# With zenodo-get
pip install zenodo-get
zenodo_get 10.5281/zenodo.18327515 -o data/
```

Or download the current archive directly:

```bash
wget -O data/pmf_data_v1.1.zip \
  https://zenodo.org/api/records/18327515/files/pmf_data_v1.1.zip/content
```

Extract the downloaded archive into `data/`:

```bash
unzip -o data/pmf_data_v1.1.zip -d data/
```

After extraction the `data/` directory should contain `raw_logs/`, `processed_logs/`, and `time_series/` as described in [Datasets](#datasets).

For the reproducible paper workflow, preprocess the Parquet time series once so training and inference share the exact same split boundaries:

```bash
python -m pmf_tsfm.data.preprocess --multirun \
  data=bpi2017,bpi2019_1,sepsis,hospital_billing
```

This writes `data/processed/{dataset}/full.parquet`, `train.parquet`, `val.parquet`, `test.parquet`, and `metadata.json`, which are then used by training and inference. See [Common Workflows](#common-workflows) for run examples and [HPC](#hpc-vsc-wice-cluster) for the cluster path.

## Quick Start

```bash
# 1. Zero-shot inference on one model/dataset pair
python -m pmf_tsfm.inference model=chronos/bolt_small data=bpi2017

# 2. Evaluate that output directory
python -m pmf_tsfm.evaluate \
  results_dir=outputs/zero_shot/BPI2017/chronos_bolt_small

# 3. Evaluate Entropic Relevance on the same predictions
python -m pmf_tsfm.er.evaluate_er model=chronos/bolt_small data=bpi2017
```

Add `logger=wandb` or `logger=wandb_offline` to any Hydra command if you want W&B tracking.

## Common Workflows

All experiment entry points are [Hydra](https://hydra.cc/)-based.

### Zero-shot inference

```bash
# Default run (Chronos Bolt Small on BPI2017)
python -m pmf_tsfm.inference

# Single model + dataset
python -m pmf_tsfm.inference model=chronos/bolt_small data=bpi2017

# Sweep multiple combinations
python -m pmf_tsfm.inference --multirun \
  model=chronos/bolt_small,chronos/bolt_base \
  data=bpi2017,bpi2019_1,sepsis,hospital_billing
```

### Fine-tuning

```bash
# LoRA fine-tuning
python -m pmf_tsfm.train \
  task=lora_tune model=chronos/bolt_small data=bpi2017 lora=chronos

# Full fine-tuning
python -m pmf_tsfm.train \
  task=full_tune model=chronos/bolt_small data=bpi2017
```

### Inference with fine-tuned models

```bash
# LoRA-adapted inference
python -m pmf_tsfm.inference model=chronos/bolt_small data=bpi2017 \
  task=lora_tune lora_adapter_path=results/lora_tune/BPI2017/chronos_bolt_small/lora_adapter/best

# Fully fine-tuned inference
python -m pmf_tsfm.inference model=chronos/bolt_small data=bpi2017 \
  task=full_tune checkpoint_path=results/full_tune/BPI2017/chronos_bolt_small/checkpoints/best
```

### Evaluation

```bash
# Evaluate all zero-shot outputs
python -m pmf_tsfm.evaluate

# Evaluate all LoRA or full-tune outputs
python -m pmf_tsfm.evaluate task=lora_tune
python -m pmf_tsfm.evaluate task=full_tune

# Evaluate a specific model/dataset directory
python -m pmf_tsfm.evaluate \
  results_dir=outputs/zero_shot/BPI2017/chronos_bolt_small

# Entropic Relevance on one model/dataset pair
python -m pmf_tsfm.er.evaluate_er model=chronos/bolt_small data=bpi2017
```

### Batch scripts

```bash
# All zero-shot combinations
bash scripts/run_inference_all.sh

# All LoRA train + inference runs
bash scripts/run_lora_all.sh

# All full fine-tune + inference runs
bash scripts/run_full_tune_all.sh

# Batch ER evaluation
bash scripts/run_er_all.sh

# Full 10-stage end-to-end pipeline
bash scripts/run_full_pipeline.sh
```

These are local orchestration scripts: shell helpers for running sequential experiment batches on your workstation or server, without [Slurm](https://slurm.schedmd.com/) job submission. The shell scripts source `scripts/env.sh`, which loads `.env` and activates `.venv` automatically when present.

## Tested Environments

- macOS on Apple Silicon: tested with `device=mps` for local development and lighter runs.
- Linux workstation/server with NVIDIA GPUs: tested with `device=cuda` and the local `scripts/env.sh` helpers.
- [VSC](https://www.vscentrum.be/) wICE cluster: tested with [Slurm](https://slurm.schedmd.com/) submission scripts under `scripts/hpc/` for NVIDIA H100 GPU jobs.

For macOS with MPS, keep `training.num_workers=0`. For Linux systems with NVIDIA GPUs and for the HPC cluster, higher worker counts such as `training.num_workers=4` are the intended path.

## Project Structure

```text
pmf-tsfm/
├── src/pmf_tsfm/       # Python package: model adapters, data modules, evaluation
├── configs/            # Hydra configs for tasks, models, datasets, loggers, paths
├── scripts/            # Local orchestration scripts and HPC helpers
├── tests/              # pytest suite
├── data/               # Zenodo assets plus generated processed splits
├── outputs/            # Saved predictions and evaluation artifacts
├── results/            # Checkpoints and LoRA adapters
├── notebooks/          # Analysis notebooks
├── manuscript/         # Paper assets
└── slides/             # Presentation materials
```

## HPC (VSC wICE cluster)

[Slurm](https://slurm.schedmd.com/) submission scripts for the [VSC](https://www.vscentrum.be/) wICE cluster live under `scripts/hpc/`. Use `scripts/hpc/.env.hpc.example` as the cluster-specific starting point.

W&B logging on HPC depends on `LOGGER`:

- `bash scripts/hpc/submit_pipeline.sh` defaults to `LOGGER=disabled`.
- Direct stage scripts such as `submit_zero_shot.sh`, `submit_lora.sh`, and `submit_full_tune.sh` default to `LOGGER=wandb`.
- Use `LOGGER=wandb_offline` only if you want offline runs, then sync them later with `bash scripts/hpc/sync_wandb_offline.sh`.

```bash
bash scripts/hpc/setup_vsc.sh          # One-time environment setup
bash scripts/hpc/submit_pipeline.sh    # Default: no W&B logging
LOGGER=wandb bash scripts/hpc/submit_pipeline.sh
LOGGER=wandb_offline bash scripts/hpc/submit_pipeline.sh
bash scripts/hpc/submit_zero_shot.sh   # Default: LOGGER=wandb for direct stage runs
bash scripts/hpc/sync_wandb_offline.sh # Only after explicit offline runs
```

## Citation

```bibtex
@article{yu2025time,
  title={Time Series Foundation Models for Process Model Forecasting},
  author={Yu, Yongbo and Peeperkorn, Jari and De Smedt, Johannes and De Weerdt, Jochen},
  journal={arXiv preprint arXiv:2512.07624},
  year={2025}
}
```

## License

This project is licensed under the MIT License.
