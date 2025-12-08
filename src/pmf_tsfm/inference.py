"""
Zero-shot forecasting inference script.

Usage:
    python inference.py                                    # Default: chronos/bolt_small on bpi2017
    python inference.py model=chronos/bolt_small           # Specific model
    python inference.py data=sepsis                        # Specific dataset
    python inference.py model=chronos/bolt_base data=sepsis prediction_length=14

    # Run on multiple models
    python inference.py --multirun model=chronos/bolt_tiny,chronos/bolt_mini,chronos/bolt_small
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def inference(cfg: DictConfig) -> dict:
    import numpy as np
    import torch

    from data import ZeroShotDataModule
    from models import ChronosAdapter, MoiraiAdapter
    from utils.metrics import compute_metrics, print_metrics, save_results

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 70)
    print(f"ZERO-SHOT INFERENCE")
    print(f"  Model: {cfg.model.name}")
    print(f"  Data:  {cfg.data.name}")
    print(f"  Device: {cfg.device}")
    print(f"  Prediction Length: {cfg.prediction_length}")
    print("=" * 70)

    print("\n[1/3] Preparing data...")
    data_module = ZeroShotDataModule(
        dataset_name=cfg.data.name,
        data_path=cfg.data.path,
        prediction_length=cfg.prediction_length,
        train_val_test_ratio=list(cfg.data.split_ratio),
    )
    data_module.setup()
    prepared_data = data_module.prepare_data_for_model(split="test")

    print("\n[2/3] Loading model...")

    if cfg.model.family == "chronos":
        adapter = ChronosAdapter.from_config(
            model_cfg=cfg.model,
            device=cfg.device,
            prediction_length=cfg.prediction_length,
        )
    elif cfg.model.family == "moirai":
        adapter = MoiraiAdapter.from_config(
            model_cfg=cfg.model,
            device=cfg.device,
            prediction_length=cfg.prediction_length,
        )
    else:
        raise ValueError(f"Unknown model family: {cfg.model.family}")

    adapter.load_model()

    print("\n[3/3] Generating predictions...")
    predictions, quantiles = adapter.predict(
        prepared_data=prepared_data,
        prediction_length=cfg.prediction_length,
    )

    targets = prepared_data["targets"]
    metrics = compute_metrics(
        predictions=predictions,
        targets=targets,
        feature_names=prepared_data["feature_names"],
    )

    print_metrics(metrics, f"{cfg.model.name} on {cfg.data.name}")

    metadata = OmegaConf.to_container(cfg, resolve=True)
    save_results(
        predictions=predictions,
        targets=targets,
        metrics=metrics,
        output_dir=cfg.output_dir,
        model_name=cfg.model.name,
        dataset_name=cfg.data.name,
        quantiles=quantiles,
        metadata=metadata,
    )

    return {
        "predictions": predictions,
        "targets": targets,
        "quantiles": quantiles,
        "metrics": metrics,
    }


@hydra.main(version_base="1.3", config_path="../../configs", config_name="inference")
def main(cfg: DictConfig):
    if cfg.get("print_config", False):
        print(OmegaConf.to_yaml(cfg))

    return inference(cfg)


if __name__ == "__main__":
    main()
