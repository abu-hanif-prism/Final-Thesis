"""Evaluate a trained baseline Siamese CNN checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.prediction_export import (
    make_evaluation_report_text,
    save_evaluation_report,
    save_group_summaries,
    save_metrics,
    save_predictions,
)
from src.models.siamese_cnn import create_siamese_cnn_model
from src.training.losses import get_loss_function
from src.training.npz_dataset import NPZSiameseDataset
from src.training.train_utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse trained evaluation arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained Siamese CNN.")
    parser.add_argument("--index_path", default="data/npz/final_npz_index.parquet")
    parser.add_argument("--checkpoint_path", default="checkpoints/siamese_cnn_regression_best.pt")
    parser.add_argument("--output_mode", choices=["regression", "classification", "multitask"], default="regression")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--output_dir", default="outputs/evaluation/siamese_cnn")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    return parser.parse_args()


def main() -> None:
    """Load checkpoint, evaluate selected split, and export predictions."""
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Evaluation skipped. Train the model first or pass --checkpoint_path.")
        return

    device = torch.device(args.device) if args.device != "auto" else get_device(prefer_gpu=True)
    target_mode = "both" if args.output_mode == "multitask" else args.output_mode
    try:
        dataset = NPZSiameseDataset(args.index_path, split=args.split, target_mode=target_mode)
    except Exception as exc:
        print(f"Could not load NPZ dataset: {exc}")
        return
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = create_siamese_cnn_model(
        output_mode=args.output_mode,
        base_channels=args.base_channels,
        dropout=args.dropout,
    )
    try:
        load_checkpoint(checkpoint_path, model, map_location=device)
    except Exception as exc:
        print(f"Could not load checkpoint {checkpoint_path}: {exc}")
        return

    loss_fn = get_loss_function(args.output_mode)
    evaluator = ModelEvaluator(model, dataloader, output_mode=args.output_mode, device=device, loss_fn=loss_fn)
    result = evaluator.evaluate()
    prediction_df = result["predictions"]
    group_summaries = evaluator.summarize_by_groups(prediction_df)

    output_dir = Path(args.output_dir)
    save_predictions(prediction_df, output_dir / "predictions")
    save_metrics(result["metrics"], output_dir / "metrics")
    save_group_summaries(group_summaries, output_dir, "group_summary")
    report_text = make_evaluation_report_text(result["metrics"], group_summaries)
    save_evaluation_report(report_text, output_dir / "evaluation_report.txt")

    print("Evaluation completed.")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  device: {device}")
    print(f"  split: {args.split}")
    print(f"  dataset size: {len(dataset)}")
    print(f"  output_mode: {args.output_mode}")
    print(f"  average_loss: {result.get('average_loss')}")
    print("  main metrics:")
    for key, value in result["metrics"].items():
        print(f"    {key}: {value}")
    print(f"  output_dir: {output_dir}")
    print("  first 5 predictions:")
    print(prediction_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
