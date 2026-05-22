"""Smoke evaluation for the baseline Siamese CNN with random weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse smoke evaluation arguments."""
    parser = argparse.ArgumentParser(description="Smoke evaluate Siamese CNN.")
    parser.add_argument("--index_path", default="data/npz/final_npz_index.parquet")
    parser.add_argument("--output_mode", choices=["regression", "classification", "multitask"], default="regression")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--test_limit", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output_dir", default="outputs/evaluation/smoke_cnn")
    return parser.parse_args()


def main() -> None:
    """Run smoke evaluation and export outputs."""
    try:
        import torch
        from torch.utils.data import DataLoader, Subset

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
        from src.training.train_utils import get_device, set_random_seed
    except ImportError as exc:
        print(f"Smoke evaluation cannot start: missing dependency ({exc})")
        return

    args = parse_args()
    set_random_seed(42)
    device = torch.device(args.device) if args.device != "auto" else get_device(prefer_gpu=True)
    target_mode = "both" if args.output_mode == "multitask" else args.output_mode
    try:
        dataset = NPZSiameseDataset(args.index_path, split="test", target_mode=target_mode)
    except Exception as exc:
        print(f"Smoke evaluation cannot load NPZ dataset: {exc}")
        return

    subset = Subset(dataset, range(min(args.test_limit, len(dataset))))
    dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = create_siamese_cnn_model(output_mode=args.output_mode)
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

    print("Smoke evaluation completed.")
    print(f"  device: {device}")
    print(f"  output_mode: {args.output_mode}")
    print(f"  dataset size: {len(subset)}")
    print(f"  average_loss: {result.get('average_loss')}")
    print(f"  metrics: {result['metrics']}")
    print(f"  output_dir: {output_dir}")
    print("  first 5 predictions:")
    print(prediction_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
