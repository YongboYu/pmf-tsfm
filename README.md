# PMF-TSFM: Process Mining Forecasting with Time Series Foundation Models

A unified pipeline for time series forecasting using foundation models, with support for zero-shot, LoRA, and full fine-tuning modes.

## Supported Models

### Chronos (Amazon)
| Model | Size | Config |
|-------|------|--------|
| Chronos Bolt Tiny | 8M | `chronos/bolt_tiny` |
| Chronos Bolt Mini | 21M | `chronos/bolt_mini` |
| Chronos Bolt Small | 48M | `chronos/bolt_small` |
| Chronos Bolt Base | 205M | `chronos/bolt_base` |
| **Chronos 2.0** | - | `chronos/chronos2` |

### Moirai (Salesforce)
| Model | Size | Config |
|-------|------|--------|
| Moirai 1.1 Small | 14M | `moirai/1_1_small` |
| Moirai 1.1 Base | 91M | `moirai/1_1_base` |
| Moirai 1.1 Large | 311M | `moirai/1_1_large` |
| **Moirai 2.0 Small** | - | `moirai/2_0_small` |
| **Moirai MoE Base** | - | `moirai/moe_base` |

## Installation

### Using uv (recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync
```

### Using pip

```bash
pip install -e .
```

## Project Structure

```
pmf-tsfm/
├── inference.py              # Zero-shot inference (predictions only)
├── evaluate.py               # Evaluation (metrics computation)
├── train.py                  # Training (future: LoRA/full fine-tuning)
├── pyproject.toml            # uv/pip dependency management
├── configs/
│   ├── inference.yaml        # Inference config
│   ├── eval.yaml             # Evaluation config
│   ├── paths/default.yaml    # Path settings
│   ├── data/                 # Dataset configs (absolute point splits)
│   │   ├── bpi2017.yaml
│   │   ├── bpi2019_1.yaml
│   │   ├── hospital_billing.yaml
│   │   └── sepsis.yaml
│   └── model/
│       ├── chronos/          # Chronos Bolt + Chronos 2.0
│       └── moirai/           # Moirai 1.1, 2.0, MoE
└── src/pmf_tsfm/
    ├── data/
    │   └── zero_shot.py      # Data module with expanding window
    ├── models/
    │   ├── base.py           # Base adapter class
    │   ├── chronos.py        # Chronos adapter (Bolt + 2.0)
    │   └── moirai.py         # Moirai adapter (1.1, 2.0, MoE)
    └── utils/
        └── metrics.py        # MAE + per-sequence RMSE
```

## Data Splits

Uses **absolute point splits** for clear, reproducible data partitioning:

```yaml
# configs/data/bpi2017.yaml
train_end: 191     # Training: [0, 191)
val_end: 255       # Validation: [191, 255)
                   # Test: [255, end)
```

## Quick Start

### Step 1: Run Inference (Generate Predictions)

```bash
# Default: Chronos Bolt Small on BPI2017
python inference.py

# Specific model and dataset
python inference.py model=chronos/bolt_base data=sepsis

# Chronos 2.0
python inference.py model=chronos/chronos2

# Moirai models
python inference.py model=moirai/1_1_large
python inference.py model=moirai/2_0_small
python inference.py model=moirai/moe_base

# Override settings
python inference.py prediction_length=14 device=cuda
```

### Step 2: Evaluate Predictions

```bash
# Evaluate all results in default directory
python evaluate.py

# Evaluate specific directory
python evaluate.py results_dir=results/zero_shot

# Evaluate specific model-dataset pair
python evaluate.py \
    results_dir=results/zero_shot \
    model_name=chronos_bolt_small \
    dataset_name=BPI2017 \
    all=false
```

### Multi-Run Experiments

```bash
# Run on all 4 datasets
python inference.py --multirun data=bpi2017,bpi2019_1,hospital_billing,sepsis

# Run all Chronos models
python inference.py --multirun \
    model=chronos/bolt_tiny,chronos/bolt_small,chronos/bolt_base,chronos/chronos2

# Full grid: all models × all datasets
python inference.py --multirun \
    model=chronos/bolt_tiny,chronos/bolt_small,chronos/chronos2 \
    data=bpi2017,bpi2019_1,hospital_billing,sepsis

# Then evaluate all
python evaluate.py
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error (overall) |
| **Per-sequence RMSE** | RMSE computed per sequence, reported as mean ± std |

### Future: Entropic Relevance
Planned extension for assessing forecasted process models.

## Output Files

After inference:
```
results/zero_shot/
├── BPI2017_chronos_bolt_small_predictions.npy
├── BPI2017_chronos_bolt_small_targets.npy
├── BPI2017_chronos_bolt_small_quantiles.npy
└── BPI2017_chronos_bolt_small_metadata.json
```

After evaluation:
```
results/zero_shot/
└── BPI2017_chronos_bolt_small_metrics.json
```

## Configuration

### Override Examples

```bash
# Print full config
python inference.py print_config=true

# Custom output directory
python inference.py output_dir=./my_results

# Change device
python inference.py device=cuda

# Change prediction horizon
python inference.py prediction_length=14
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | Model config | `chronos/bolt_small` |
| `data` | Dataset config | `bpi2017` |
| `device` | Device (cpu, cuda, mps) | `mps` |
| `prediction_length` | Forecast horizon | `7` |
| `seed` | Random seed | `42` |
| `output_dir` | Results directory | `results/zero_shot` |

## Extending the Pipeline

### Adding a New Model

1. Create adapter in `src/pmf_tsfm/models/`
2. Register in `src/pmf_tsfm/models/__init__.py`
3. Create config in `configs/model/`

### Adding a New Dataset

1. Create config in `configs/data/`:
```yaml
name: MyDataset
path: ${paths.data_dir}/MyDataset.parquet
train_end: 191
val_end: 255
```

2. Place data file in `data/time_series/`

## Future Work

- [ ] LoRA fine-tuning support
- [ ] Full fine-tuning support
- [ ] Entropic Relevance metric
- [ ] TimesFM model support

## License

MIT License
