"""Unified evaluation script for any supported trained model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse unified evaluation arguments."""
    parser = argparse.ArgumentParser(description="Evaluate any supported trained model.")
    parser.add_argument("--model_name", required=True, choices=["cnn", "swin", "convnext", "maxvit"])
    parser.add_argument("--output_mode", choices=["regression", "classification", "multitask"], default="regression")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--model_config_path", default=None)
    parser.add_argument("--index_path", default="data/npz/final_npz_index.parquet")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--loss", action="store_true")
    parser.add_argument("--save_predictions_parquet", action="store_true")
    parser.add_argument("--image_embedding_dim", type=int, default=256)
    parser.add_argument("--tabular_embedding_dim", type=int, default=128)
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=96)
    parser.add_argument("--swin_depth", type=int, default=2)
    parser.add_argument("--swin_num_heads", type=int, default=4)
    parser.add_argument("--swin_patch_size", type=int, default=4)
    parser.add_argument("--maxvit_window_size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """Load model/checkpoint, evaluate split, and export results."""
    try:
        import torch
        from torch.utils.data import DataLoader

        from src.evaluation.evaluator import ModelEvaluator
        from src.models.model_factory import (
            create_model,
            create_model_from_config,
            load_model_config,
            normalize_model_name,
        )
        from src.models.model_utils import count_parameters
        from src.training.losses import get_loss_function
        from src.training.npz_dataset import NPZSiameseDataset
        from src.training.train_utils import get_device, load_checkpoint
    except ImportError as exc:
        print(f"Evaluation cannot start: missing dependency ({exc})")
        return

    args = parse_args()
    model_name = normalize_model_name(args.model_name)
    experiment_name = resolve_experiment_name(args, model_name)
    checkpoint_path = resolve_checkpoint_path(args, experiment_name)
    model_config_path = resolve_model_config_path(args, experiment_name)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/evaluation") / experiment_name / args.split

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Evaluation skipped. Train the model first or pass --checkpoint_path.")
        return

    device = torch.device(args.device) if args.device != "auto" else get_device(prefer_gpu=True)
    model, output_mode = create_model_for_evaluation(args, model_name, model_config_path)
    target_mode = "both" if output_mode == "multitask" else output_mode

    try:
        dataset = NPZSiameseDataset(args.index_path, split=args.split, target_mode=target_mode)
    except Exception as exc:
        print(f"Evaluation cannot load NPZ dataset: {exc}")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    try:
        load_checkpoint(checkpoint_path, model, map_location=device)
    except Exception as exc:
        print(f"Could not load checkpoint {checkpoint_path}: {exc}")
        return

    loss_fn = get_loss_function(output_mode) if args.loss else None
    evaluator = ModelEvaluator(model, dataloader, output_mode=output_mode, device=device, loss_fn=loss_fn)
    result = evaluator.evaluate()
    prediction_df = result["predictions"]
    group_summaries = evaluator.summarize_by_groups(prediction_df)
    export_results(output_dir, prediction_df, result["metrics"], group_summaries, args.save_predictions_parquet)
    params = count_parameters(model)

    print("Evaluation completed.")
    print(f"  model_name: {model_name}")
    print(f"  output_mode: {output_mode}")
    print(f"  experiment_name: {experiment_name}")
    print(f"  checkpoint_path: {checkpoint_path}")
    print(f"  model_config_path: {model_config_path}")
    print(f"  device: {device}")
    print(f"  evaluated split: {args.split}")
    print(f"  dataset size: {len(dataset)}")
    print(f"  parameters: total={params['total']}, trainable={params['trainable']}")
    print(f"  output_dir: {output_dir}")
    if "average_loss" in result:
        print(f"  average_loss: {result['average_loss']}")
    print("  main metrics:")
    for key, value in result["metrics"].items():
        print(f"    {key}: {value}")
    print("  first 5 predictions:")
    print(prediction_df.head(5).to_string(index=False))


def create_model_for_evaluation(args: argparse.Namespace, model_name: str, config_path: Path):
    """Create model from saved config when available, otherwise CLI args."""
    from src.models.model_factory import create_model, create_model_from_config, load_model_config

    if config_path.exists():
        config = load_model_config(config_path)
        model = create_model_from_config(config)
        return model, config.get("output_mode", args.output_mode)
    model = create_model(
        model_name=model_name,
        output_mode=args.output_mode,
        image_embedding_dim=args.image_embedding_dim,
        tabular_embedding_dim=args.tabular_embedding_dim,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        **build_model_kwargs(args, model_name),
    )
    return model, args.output_mode


def build_model_kwargs(args: argparse.Namespace, model_name: str) -> dict[str, object]:
    """Build model-specific kwargs from CLI args."""
    if model_name == "cnn":
        return {"base_channels": args.base_channels}
    if model_name == "swin":
        return {
            "embed_dim": args.embed_dim,
            "depth": args.swin_depth,
            "num_heads": args.swin_num_heads,
            "patch_size": args.swin_patch_size,
        }
    if model_name == "maxvit":
        return {"window_size": args.maxvit_window_size}
    return {}


def resolve_experiment_name(args: argparse.Namespace, model_name: str) -> str:
    """Resolve experiment name."""
    return args.experiment_name or f"{model_name}_{args.output_mode}"


def resolve_checkpoint_path(args: argparse.Namespace, experiment_name: str) -> Path:
    """Resolve checkpoint path."""
    if args.checkpoint_path:
        return Path(args.checkpoint_path)
    return Path("checkpoints") / f"{experiment_name}_best.pt"


def resolve_model_config_path(args: argparse.Namespace, experiment_name: str) -> Path:
    """Resolve model config path."""
    if args.model_config_path:
        return Path(args.model_config_path)
    return Path("checkpoints") / f"{experiment_name}_model_config.json"


def export_results(output_dir: Path, prediction_df, metrics, group_summaries, save_parquet: bool) -> None:
    """Save prediction, metric, summary, and report outputs."""
    from src.evaluation.prediction_export import (
        make_evaluation_report_text,
        save_evaluation_report,
        save_group_summaries,
        save_metrics,
        save_predictions,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_parquet:
        save_predictions(prediction_df, output_dir / "predictions")
    else:
        prediction_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8")
    save_metrics(metrics, output_dir / "metrics")
    save_group_summaries(group_summaries, output_dir, "group_summary")
    report_text = make_evaluation_report_text(metrics, group_summaries)
    save_evaluation_report(report_text, output_dir / "evaluation_report.txt")


if __name__ == "__main__":
    main()
