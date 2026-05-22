"""Train the baseline multimodal Siamese CNN."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.model_utils import count_parameters
from src.models.siamese_cnn import create_siamese_cnn_model
from src.training.dataloaders import create_train_val_test_dataloaders
from src.training.losses import get_loss_function
from src.training.trainer import Trainer
from src.training.train_utils import (
    create_optimizer,
    create_scheduler,
    get_device,
    set_random_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse real training arguments."""
    parser = argparse.ArgumentParser(description="Train baseline multimodal Siamese CNN.")
    parser.add_argument("--index_path", default="data/npz/final_npz_index.parquet")
    parser.add_argument("--output_mode", choices=["regression", "classification", "multitask"], default="regression")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--experiment_name", default="siamese_cnn_regression")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    """Create loaders/model/trainer and run training."""
    args = parse_args()
    set_random_seed(42)
    device = torch.device(args.device) if args.device != "auto" else get_device(prefer_gpu=True)
    target_mode = "both" if args.output_mode == "multitask" else args.output_mode

    train_loader, val_loader, test_loader = create_train_val_test_dataloaders(
        index_path=args.index_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_mode=target_mode,
        pin_memory=(device.type == "cuda"),
    )
    model = create_siamese_cnn_model(
        output_mode=args.output_mode,
        base_channels=args.base_channels,
        dropout=args.dropout,
    )
    loss_fn = get_loss_function(args.output_mode)
    optimizer = create_optimizer(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, scheduler_type="plateau")
    params = count_parameters(model)

    print("Training configuration:")
    print(f"  device: {device}")
    print(f"  output_mode: {args.output_mode}")
    print(f"  train size: {len(train_loader.dataset)}")
    print(f"  val size: {len(val_loader.dataset)}")
    print(f"  test size: {len(test_loader.dataset)}")
    print(f"  model parameters: total={params['total']}, trainable={params['trainable']}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  epochs: {args.epochs}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  checkpoint_dir: {args.checkpoint_dir}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        output_mode=args.output_mode,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=args.mixed_precision,
        grad_clip_norm=args.grad_clip_norm,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        metric_for_best="val_total_loss",
    )
    trainer.fit(args.epochs)


if __name__ == "__main__":
    main()
