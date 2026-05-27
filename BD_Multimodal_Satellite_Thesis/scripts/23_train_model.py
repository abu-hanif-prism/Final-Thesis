"""Unified training script for all supported multimodal Siamese models."""

from __future__ import annotations

import argparse
import csv
import json
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
from src.training.train_utils import create_optimizer, create_scheduler, get_device, load_checkpoint, set_random_seed


def parse_args() -> argparse.Namespace:
    """Parse unified training arguments."""
    parser = argparse.ArgumentParser(
        description="Train any supported multimodal Siamese model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Recommended stable MaxViT command:\n"
            "python -u scripts/23_train_model.py --index_path data/npz/final_npz_index.csv "
            "--model_name maxvit --output_mode regression --batch_size 2 --epochs 30 "
            "--learning_rate 3e-5 --weight_decay 5e-4 --regression_loss_type smooth_l1 "
            "--huber_delta 0.1 --grad_clip_norm 1.0 --early_stopping_patience 5 "
            "--num_workers 0 --device cuda --experiment_name maxvit_regression_stable --restart\n"
            "For MaxViT stability, avoid --mixed_precision unless you have verified it is stable."
        ),
    )
    parser.add_argument("--model_name", required=True, choices=["cnn", "swin", "convnext", "maxvit"])
    parser.add_argument("--output_mode", choices=["regression", "classification", "multitask"], default="regression")
    parser.add_argument("--index_path", default="data/npz/final_npz_index.csv")
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--resume_from_latest", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--early_stopping_metric", default="val_total_loss")
    parser.add_argument("--early_stopping_mode", choices=["min", "max"], default="min")
    return parser.parse_args()


def main() -> None:
    """Run unified model training."""
    args = parse_args()
    if args.resume_from_latest:
        args.resume = True
    set_random_seed(args.seed)
    model_name = normalize_model_name(args.model_name)
    experiment_name = args.experiment_name or build_experiment_name(model_name, args.output_mode)
    latest_checkpoint_path = Path(args.checkpoint_dir) / f"{experiment_name}_latest.pt"
    if args.resume and args.restart:
        print("Use either --resume or --restart, not both.")
        return
    if latest_checkpoint_path.exists() and args.restart:
        print(f"Warning: --restart will overwrite existing checkpoints for {experiment_name}.")
    if latest_checkpoint_path.exists() and not args.resume and not args.restart:
        print(f"Existing checkpoint found: {latest_checkpoint_path}")
        print("Use --resume to continue or --restart to overwrite.")
        return

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
    save_config_safely(config, config_path, restart=args.restart, resume=args.resume)

    start_epoch = 1
    best_metric: float | None = None
    history: list[dict[str, object]] = []
    if args.resume:
        resume_checkpoint_path = resolve_resume_checkpoint(args, experiment_name)
        if not resume_checkpoint_path.exists():
            print(f"Resume checkpoint not found: {resume_checkpoint_path}")
            return
        checkpoint = load_checkpoint(
            resume_checkpoint_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        if checkpoint_epoch >= int(args.epochs):
            print("Checkpoint already reached requested epochs")
            print(f"  checkpoint epoch: {checkpoint_epoch}")
            print(f"  requested total epochs: {args.epochs}")
            return
        start_epoch = checkpoint_epoch + 1
        history = load_resume_history(checkpoint, Path(args.log_dir) / f"{experiment_name}_history.csv")
        best_metric = resolve_best_metric(checkpoint, Path(args.checkpoint_dir) / f"{experiment_name}_best.pt")
        print_resume_summary(
            experiment_name=experiment_name,
            checkpoint_path=resume_checkpoint_path,
            checkpoint=checkpoint,
            requested_epochs=args.epochs,
            next_epoch=start_epoch,
            best_metric=best_metric,
            history=history,
        )

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
        start_epoch=start_epoch,
        best_metric=best_metric,
        history=history,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        early_stopping_mode=args.early_stopping_mode,
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


def resolve_resume_checkpoint(args: argparse.Namespace, experiment_name: str) -> Path:
    """Resolve checkpoint path for resume mode."""
    if args.resume_checkpoint:
        return Path(args.resume_checkpoint)
    return Path(args.checkpoint_dir) / f"{experiment_name}_latest.pt"


def save_config_safely(config: dict[str, object], config_path: Path, restart: bool, resume: bool) -> None:
    """Save config without overwriting a different resume config."""
    if restart or not config_path.exists():
        save_model_config(config, config_path)
        return
    try:
        with config_path.open("r", encoding="utf-8") as file:
            existing_config = json.load(file)
    except Exception as exc:
        print(f"Warning: could not read existing model config {config_path}: {exc}")
        if resume:
            print(f"Keeping existing model config path: {config_path}")
        return
    if existing_config == config:
        print(f"Existing model config is identical: {config_path}")
        return
    if resume:
        print(f"Loaded existing model config path: {config_path}")
        print("Existing model config differs from current CLI config; not overwriting during resume.")
        return
    print(f"Existing model config differs and was not overwritten: {config_path}")


def load_resume_history(checkpoint: dict[str, object], history_path: Path) -> list[dict[str, object]]:
    """Load old history from checkpoint or existing history CSV."""
    checkpoint_history = checkpoint.get("history")
    if isinstance(checkpoint_history, list) and checkpoint_history:
        return [dict(row) for row in checkpoint_history if isinstance(row, dict)]
    if not history_path.exists():
        return []
    with history_path.open("r", newline="", encoding="utf-8") as file:
        return [dict(row) for row in csv.DictReader(file)]


def resolve_best_metric(checkpoint: dict[str, object], best_checkpoint_path: Path) -> float | None:
    """Resolve best metric from checkpoint fields, best checkpoint, or latest metrics."""
    if checkpoint.get("best_metric") is not None:
        return float(checkpoint["best_metric"])
    if best_checkpoint_path.exists():
        try:
            best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
            if best_checkpoint.get("best_metric") is not None:
                return float(best_checkpoint["best_metric"])
            metrics = best_checkpoint.get("metrics", {})
            if isinstance(metrics, dict) and metrics.get("val_loss") is not None:
                return float(metrics["val_loss"])
        except Exception as exc:
            print(f"Warning: could not inspect best checkpoint {best_checkpoint_path}: {exc}")
    metrics = checkpoint.get("metrics", {})
    if isinstance(metrics, dict) and metrics.get("val_loss") is not None:
        return float(metrics["val_loss"])
    return None


def print_resume_summary(
    experiment_name: str,
    checkpoint_path: Path,
    checkpoint: dict[str, object],
    requested_epochs: int,
    next_epoch: int,
    best_metric: float | None,
    history: list[dict[str, object]],
) -> None:
    """Print resume state clearly."""
    print("Resume training configuration:")
    print(f"  experiment_name: {experiment_name}")
    print(f"  checkpoint path loaded: {checkpoint_path}")
    print(f"  checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"  requested total epochs: {requested_epochs}")
    print(f"  next epoch: {next_epoch}")
    print(f"  best metric so far: {best_metric}")
    print(f"  history length: {len(history)}")
    print(f"  optimizer restored: {bool(checkpoint.get('_optimizer_restored'))}")
    print(f"  scheduler restored: {bool(checkpoint.get('_scheduler_restored'))}")


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
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_metric": args.early_stopping_metric,
        "early_stopping_mode": args.early_stopping_mode,
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
    print(f"  grad_clip_norm: {args.grad_clip_norm} ({'disabled' if args.grad_clip_norm <= 0 else 'enabled'})")
    print(
        "  early_stopping: "
        f"patience={args.early_stopping_patience}, "
        f"metric={args.early_stopping_metric}, "
        f"mode={args.early_stopping_mode}"
    )
    print(f"  parameters: total={params['total']}, trainable={params['trainable']}")
    print(f"  checkpoint latest: {Path(args.checkpoint_dir) / f'{experiment_name}_latest.pt'}")
    print(f"  checkpoint best: {Path(args.checkpoint_dir) / f'{experiment_name}_best.pt'}")
    print(f"  model config: {config_path}")
    print(f"  history CSV: {Path(args.log_dir) / f'{experiment_name}_history.csv'}")


if __name__ == "__main__":
    main()
