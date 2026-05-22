"""Smoke-test XAI handoff export without requiring a trained model."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.xai_handoff.export_handoff import save_handoff_package  # noqa: E402


def main() -> None:
    """Create a clearly marked smoke handoff package."""
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        print(f"Smoke handoff export cannot start: missing dependency ({exc})")
        return

    experiment_name = "smoke_test"
    model_name = "cnn"
    npz_index_path = Path("data/npz/final_npz_index.parquet")
    smoke_root = Path("outputs/xai_handoff")
    smoke_source_dir = smoke_root / "_smoke_sources"
    smoke_source_dir.mkdir(parents=True, exist_ok=True)

    try:
        index_df = pd.read_parquet(npz_index_path)
    except Exception as exc:
        print(f"Smoke handoff export cannot load NPZ index: {exc}")
        return

    sample = index_df[index_df["split"] == "test"].head(64).copy()
    if sample.empty:
        sample = index_df.head(64).copy()
    rng = np.random.default_rng(42)
    true_ratio = sample["change_ratio"].astype(float).to_numpy() if "change_ratio" in sample else rng.uniform(0, 1, len(sample))
    pred_ratio = np.clip(true_ratio + rng.normal(0, 0.08, len(sample)), 0, 1)
    predictions = sample[["patch_id", "pair_id", "district", "split", "change_class"]].copy()
    predictions["pair_type"] = sample["pair_type"] if "pair_type" in sample else "unknown"
    predictions["time_gap_group"] = sample["time_gap_group"] if "time_gap_group" in sample else "unknown"
    predictions["y_true_change_ratio"] = true_ratio
    predictions["y_pred_change_ratio"] = pred_ratio
    predictions["abs_error"] = abs(pred_ratio - true_ratio)
    predictions["squared_error"] = (pred_ratio - true_ratio) ** 2
    predictions_path = smoke_source_dir / "smoke_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    checkpoint_path = smoke_source_dir / "SMOKE_PLACEHOLDER_best.pt"
    checkpoint_path.write_text(
        "SMOKE TEST PLACEHOLDER ONLY. This is not a trained PyTorch checkpoint.\n",
        encoding="utf-8",
    )
    config_path = smoke_source_dir / "smoke_model_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "smoke_test": True,
                "model_name": model_name,
                "experiment_name": experiment_name,
                "output_mode": "regression",
                "image_channels": 13,
                "tabular_dim": 146,
                "image_embedding_dim": 256,
                "tabular_embedding_dim": 128,
                "fusion_dim": 256,
            },
            file,
            indent=2,
        )
        file.write("\n")

    try:
        manifest = save_handoff_package(
            experiment_name=experiment_name,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            model_config_path=config_path,
            predictions_path=predictions_path,
            npz_index_path=npz_index_path,
            feature_columns_path="data/tabular/processed/pair_tabular_feature_columns.json",
            output_root=smoke_root,
            num_samples=32,
            random_seed=42,
        )
    except Exception as exc:
        print(f"Smoke handoff export failed: {exc}")
        return

    print("Smoke XAI handoff export completed.")
    print("  This is not a real trained model package.")
    print(f"  selected sample count: {manifest['num_selected_samples']}")
    print(f"  output folder: {manifest['output_dir']}")
    print("  saved files:")
    for file_name in manifest["saved_files"]:
        print(f"    {file_name}")


if __name__ == "__main__":
    main()
