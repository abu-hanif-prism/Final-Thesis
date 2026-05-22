"""Export a real XAI handoff package for one trained model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.xai_handoff.export_handoff import save_handoff_package  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse handoff export arguments."""
    parser = argparse.ArgumentParser(description="Export XAI handoff package.")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--model_name", required=True, choices=["cnn", "swin", "convnext", "maxvit"])
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--model_config_path", default=None)
    parser.add_argument("--predictions_path", default=None)
    parser.add_argument("--npz_index_path", default="data/npz/final_npz_index.parquet")
    parser.add_argument(
        "--feature_columns_path",
        default="data/tabular/processed/pair_tabular_feature_columns.json",
    )
    parser.add_argument("--output_root", default="outputs/xai_handoff")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Export the real XAI handoff package."""
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else Path("checkpoints") / f"{args.experiment_name}_best.pt"
    model_config_path = (
        Path(args.model_config_path)
        if args.model_config_path
        else Path("checkpoints") / f"{args.experiment_name}_model_config.json"
    )
    predictions_path = Path(args.predictions_path) if args.predictions_path else resolve_predictions_path(args.experiment_name)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Handoff export skipped. Train the model or pass --checkpoint_path.")
        return
    if predictions_path is None or not predictions_path.exists():
        print(f"Predictions not found for experiment: {args.experiment_name}")
        print("Expected predictions.parquet or predictions.csv under outputs/evaluation/{experiment}/test/.")
        return

    try:
        manifest = save_handoff_package(
            experiment_name=args.experiment_name,
            model_name=args.model_name,
            checkpoint_path=checkpoint_path,
            model_config_path=model_config_path,
            predictions_path=predictions_path,
            npz_index_path=args.npz_index_path,
            feature_columns_path=args.feature_columns_path,
            output_root=args.output_root,
            num_samples=args.num_samples,
            random_seed=args.random_seed,
        )
    except Exception as exc:
        print(f"XAI handoff export failed: {exc}")
        return

    print("XAI handoff export completed.")
    print(f"  experiment_name: {args.experiment_name}")
    print(f"  model_name: {args.model_name}")
    print(f"  checkpoint_path: {checkpoint_path}")
    print(f"  prediction_path: {predictions_path}")
    print(f"  selected sample count: {manifest['num_selected_samples']}")
    print(f"  output folder: {manifest['output_dir']}")
    print("  saved files:")
    for file_name in manifest["saved_files"]:
        print(f"    {file_name}")
    print_distributions(Path(manifest["output_dir"]) / "sample_metadata.csv")


def resolve_predictions_path(experiment_name: str) -> Path | None:
    """Resolve default predictions path, preferring parquet."""
    base = Path("outputs/evaluation") / experiment_name / "test"
    parquet_path = base / "predictions.parquet"
    csv_path = base / "predictions.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    return parquet_path


def print_distributions(sample_metadata_path: Path) -> None:
    """Print selected sample distributions."""
    import pandas as pd

    df = pd.read_csv(sample_metadata_path)
    for column, label in [
        ("change_class", "class distribution"),
        ("district", "district count"),
        ("pair_type", "pair_type distribution"),
        ("time_gap_group", "time_gap_group distribution"),
    ]:
        if column not in df.columns:
            continue
        print(f"  {label}:")
        for value, count in df[column].value_counts().items():
            print(f"    {value}: {int(count)}")


if __name__ == "__main__":
    main()
