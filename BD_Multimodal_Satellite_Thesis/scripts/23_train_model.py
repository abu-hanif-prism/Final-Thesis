"""Unified training script for all supported multimodal Siamese models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.model_factory import create_model, normalize_model_name, save_model_config
from src.models.model_utils import count_parameters
from src.training.dataloaders import create_train_val_test_dataloaders
from src.training.losses import get_loss_function
from src.training.trainer import Trainer
from src.training.train_utils import create_optimizer, create_scheduler, get_device, set_random_seed


def parse_args() -> argparse.Namespace:
    """Parse unified training arguments."""
    parser = argparse.ArgumentParser(description="Train any supported multimodal Siamese model.")
    parser.add_argument("--model_name", required=True, choices=["cnn", "swin", "convnext", "maxvit"])
    parser.add_argument("--output_mode", choices=["regression", "classification", "multitask"], default="regression")
    parser.add_argument("--index_path", default="data/npz/final_npz_index.parquet")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--scheduler_type", choices=["none", "plateau", "cosine"], default="plateau")
    parser.add_argument("--image_embedding_dim", type=int, default=256)
    parser.add_argument("--tabular_embedding_dim", type=int, default=128)
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--embed_dim", type=int, default=96)
    parser.add_argument("--swin_depth", type=int, default=2)
    parser.add_argument("--swin_num_heads", type=int, default=4)
    parser.add_argument("--swin_patch_size", type=int, default=4)
    parser.add_argument("--convnext_tiny", action="store_true")
    parser.add_argument("--maxvit_window_size", type=int, default=8)
    parser.add_argument("--regression_loss_type", choices=["mse", "mae", "huber", "smooth_l1"], default="mse")
    parser.add_argument("--huber_delta", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--regression_weight", type=float, default=1.0)
    parser.add_argument("--classification_weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    """Run unified model training."""
    args = parse_args()
    set_random_seed(args.seed)
    model_name = normalize_model_name(args.model_name)
    experiment_name = args.experiment_name or build_experiment_name(model_name, args.output_mode)
    device = torch.device(args.device) if args.device != "auto" else get_device(prefer_gpu=True)
    target_mode = "both" if args.output_mode == "multitask" else args.output_mode
    mixed_precision = bool(args.mixed_precision and device.type == "cuda")
    if args.mixed_precision and not mixed_precision:
        print("Mixed precision requested but CUDA is unavailable; disabling mixed precision.")

    try:
        train_loader, val_loader, test_loader = create_train_val_test_dataloaders(
            index_path=args.index_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_mode=target_mode,
            pin_memory=(device.type == "cuda"),
        )
    except Exception as exc:
        print(f"Training cannot load NPZ index/dataset: {exc}")
        return
    model_kwargs = build_model_kwargs(args, model_name)
    model = create_model(
        model_name=model_name,
        output_mode=args.output_mode,
        image_embedding_dim=args.image_embedding_dim,
        tabular_embedding_dim=args.tabular_embedding_dim,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        **model_kwargs,
    )
    loss_fn = get_loss_function(args.output_mode, **build_loss_kwargs(args))
    optimizer = create_optimizer(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, scheduler_type=args.scheduler_type, T_max=args.epochs)
    params = count_parameters(model)
    config = build_config(args, model_name, experiment_name, model_kwargs)
    config_path = Path(args.checkpoint_dir) / f"{experiment_name}_model_config.json"
    save_model_config(config, config_path)

    print_training_config(args, model_name, experiment_name, device, train_loader, val_loader, test_loader, params, config_path, mixed_precision)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        output_mode=args.output_mode,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=mixed_precision,
        grad_clip_norm=args.grad_clip_norm,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        experiment_name=experiment_name,
        metric_for_best="val_loss",
    )
    trainer.fit(args.epochs)


def build_model_kwargs(args: argparse.Namespace, model_name: str) -> dict[str, object]:
    """Build model-specific kwargs safely."""
    if model_name == "cnn":
        return {"base_channels": args.base_channels}
    if model_name == "swin":
        return {
            "embed_dim": args.embed_dim,
            "depth": args.swin_depth,
            "num_heads": args.swin_num_heads,
            "patch_size": args.swin_patch_size,
        }
    if model_name == "convnext":
        if args.convnext_tiny:
            return {"dims": [32, 64, 128, 256], "depths": [2, 2, 3, 2]}
        return {}
    if model_name == "maxvit":
        return {"window_size": args.maxvit_window_size}
    return {}


def build_loss_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Build loss-function kwargs."""
    return {
        "loss_type": args.regression_loss_type,
        "regression_loss_type": args.regression_loss_type,
        "huber_delta": args.huber_delta,
        "label_smoothing": args.label_smoothing,
        "regression_weight": args.regression_weight,
        "classification_weight": args.classification_weight,
    }


def build_experiment_name(model_name: str, output_mode: str) -> str:
    """Create default experiment name."""
    return f"{model_name}_{output_mode}"


def build_config(
    args: argparse.Namespace,
    model_name: str,
    experiment_name: str,
    model_kwargs: dict[str, object],
) -> dict[str, object]:
    """Build JSON-serializable training/model config."""
    return {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "output_mode": args.output_mode,
        "index_path": args.index_path,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "mixed_precision": args.mixed_precision,
        "grad_clip_norm": args.grad_clip_norm,
        "seed": args.seed,
        "image_embedding_dim": args.image_embedding_dim,
        "tabular_embedding_dim": args.tabular_embedding_dim,
        "fusion_dim": args.fusion_dim,
        "dropout": args.dropout,
        "model_kwargs": model_kwargs,
        "loss": build_loss_kwargs(args),
    }


def print_training_config(
    args: argparse.Namespace,
    model_name: str,
    experiment_name: str,
    device: torch.device,
    train_loader,
    val_loader,
    test_loader,
    params: dict[str, int],
    config_path: Path,
    mixed_precision: bool,
) -> None:
    """Print full training configuration."""
    print("Unified training configuration:")
    print(f"  model_name: {model_name}")
    print(f"  output_mode: {args.output_mode}")
    print(f"  experiment_name: {experiment_name}")
    print(f"  device: {device}")
    print(f"  train size: {len(train_loader.dataset)}")
    print(f"  val size: {len(val_loader.dataset)}")
    print(f"  test size: {len(test_loader.dataset)}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  epochs: {args.epochs}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  mixed_precision: {mixed_precision}")
    print(f"  parameters: total={params['total']}, trainable={params['trainable']}")
    print(f"  checkpoint latest: {Path(args.checkpoint_dir) / f'{experiment_name}_latest.pt'}")
    print(f"  checkpoint best: {Path(args.checkpoint_dir) / f'{experiment_name}_best.pt'}")
    print(f"  model config: {config_path}")
    print(f"  history CSV: {Path(args.log_dir) / f'{experiment_name}_history.csv'}")


if __name__ == "__main__":
    main()
