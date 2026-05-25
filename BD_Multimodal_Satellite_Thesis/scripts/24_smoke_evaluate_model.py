"""Smoke evaluation for any supported multimodal Siamese model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse smoke evaluation arguments."""
    parser = argparse.ArgumentParser(description="Smoke evaluate any supported model.")
    parser.add_argument("--model_name", default="cnn")
    parser.add_argument("--output_mode", choices=["regression", "classification", "multitask"], default="regression")
    parser.add_argument("--index_path", default="data/npz/final_npz_index.csv")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--test_limit", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output_dir", default="outputs/evaluation/smoke")
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--save_predictions_parquet", action="store_true")
    parser.add_argument("--save_parquet", action="store_true")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """Run smoke evaluation and export predictions/metrics."""
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
        from src.models.model_factory import create_model, normalize_model_name
        from src.models.model_utils import count_parameters
        from src.training.losses import get_loss_function
        from src.training.npz_dataset import NPZSiameseDataset
        from src.training.train_utils import get_device, set_random_seed
    except ImportError as exc:
        print(f"Smoke evaluation cannot start: missing dependency ({exc})")
        return

    args = parse_args()
    set_random_seed(42)
    model_name = normalize_model_name(args.model_name)
    device = torch.device(args.device) if args.device != "auto" else get_device(prefer_gpu=True)
    target_mode = "both" if args.output_mode == "multitask" else args.output_mode

    try:
        dataset = NPZSiameseDataset(args.index_path, split=args.split, target_mode=target_mode)
    except Exception as exc:
        print(f"Smoke evaluation cannot load NPZ dataset: {exc}")
        return

    subset = Subset(dataset, range(min(args.test_limit, len(dataset))))
    dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = create_model(
        model_name=model_name,
        output_mode=args.output_mode,
        **build_model_kwargs(args, model_name),
    )
    loss_fn = get_loss_function(args.output_mode)
    evaluator = ModelEvaluator(
        model,
        dataloader,
        output_mode=args.output_mode,
        device=device,
        loss_fn=loss_fn,
        progress_every=args.progress_every,
    )
    result = evaluator.evaluate()
    predictions = result["predictions"]
    group_summaries = evaluator.summarize_by_groups(predictions)

    output_dir = Path(args.output_dir) / f"{model_name}_{args.output_mode}_{args.split}"
    export_results(
        output_dir,
        predictions,
        result["metrics"],
        group_summaries,
        args.save_predictions_parquet or args.save_parquet,
    )
    params = count_parameters(model)

    print("Smoke evaluation completed.")
    print(f"  model_name: {model_name}")
    print(f"  output_mode: {args.output_mode}")
    print(f"  device: {device}")
    print(f"  split: {args.split}")
    print(f"  dataset size: {len(subset)}")
    print(f"  parameters: total={params['total']}, trainable={params['trainable']}")
    print(f"  average_loss: {result.get('average_loss')}")
    print(f"  metrics: {result['metrics']}")
    print(f"  output_dir: {output_dir}")
    print("  first 5 predictions:")
    print(format_prediction_preview(predictions[:5]))


def build_model_kwargs(args: argparse.Namespace, model_name: str) -> dict[str, object]:
    """Build model-specific kwargs."""
    if model_name == "cnn":
        return {"base_channels": args.base_channels}
    if model_name == "swin":
        return {
            "embed_dim": args.embed_dim,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "patch_size": args.patch_size,
        }
    if model_name == "maxvit":
        return {"window_size": args.window_size}
    return {}


def export_results(output_dir: Path, predictions, metrics, group_summaries, save_parquet: bool) -> None:
    """Save prediction, metric, summary, and report outputs."""
    from src.evaluation.prediction_export import (
        make_evaluation_report_text,
        save_evaluation_report,
        save_group_summaries,
        save_metrics,
        save_predictions,
    )

    save_predictions(predictions, output_dir / "predictions", save_parquet=save_parquet)
    save_metrics(metrics, output_dir / "metrics")
    save_group_summaries(group_summaries, output_dir, "group_summary")
    report_text = make_evaluation_report_text(metrics, group_summaries)
    save_evaluation_report(report_text, output_dir / "evaluation_report.txt")


def format_prediction_preview(rows: list[dict[str, object]]) -> str:
    """Format first prediction rows without pandas."""
    if not rows:
        return "<no predictions>"
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    lines = [" | ".join(fieldnames)]
    for row in rows:
        lines.append(" | ".join(str(row.get(key, "")) for key in fieldnames))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
