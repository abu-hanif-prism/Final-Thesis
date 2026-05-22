"""Small smoke training run for the baseline multimodal Siamese CNN."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse smoke-training arguments."""
    parser = argparse.ArgumentParser(description="Smoke train Siamese CNN on a tiny subset.")
    parser.add_argument("--output_mode", choices=["regression", "classification", "multitask"], default="regression")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_limit", type=int, default=64)
    parser.add_argument("--val_limit", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--index_path", default="data/npz/final_npz_index.parquet")
    return parser.parse_args()


def main() -> None:
    """Run a one-epoch smoke training check."""
    try:
        import torch
        from torch.utils.data import DataLoader, Subset

        from src.models.model_utils import count_parameters
        from src.models.siamese_cnn import create_siamese_cnn_model
        from src.training.losses import get_loss_function
        from src.training.npz_dataset import NPZSiameseDataset
        from src.training.trainer import Trainer
        from src.training.train_utils import create_optimizer, create_scheduler, get_device, set_random_seed
    except ImportError as exc:
        print(f"Smoke training cannot start: missing dependency ({exc})")
        return

    args = parse_args()
    set_random_seed(42)
    target_mode = "both" if args.output_mode == "multitask" else args.output_mode

    try:
        train_dataset = NPZSiameseDataset(args.index_path, split="train", target_mode=target_mode)
        val_dataset = NPZSiameseDataset(args.index_path, split="val", target_mode=target_mode)
    except Exception as exc:
        print(f"Smoke training cannot load NPZ index/dataset: {exc}")
        return

    train_subset = Subset(train_dataset, range(min(args.train_limit, len(train_dataset))))
    val_subset = Subset(val_dataset, range(min(args.val_limit, len(val_dataset))))
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device) if args.device else get_device(prefer_gpu=True)
    model = create_siamese_cnn_model(output_mode=args.output_mode)
    loss_fn = get_loss_function(args.output_mode)
    optimizer = create_optimizer(model, learning_rate=1e-4, weight_decay=1e-4)
    scheduler = create_scheduler(optimizer, scheduler_type="plateau")
    params = count_parameters(model)

    print("Smoke training configuration:")
    print(f"  device: {device}")
    print(f"  output_mode: {args.output_mode}")
    print(f"  train subset: {len(train_subset)}")
    print(f"  val subset: {len(val_subset)}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  epochs: {args.epochs}")
    print(f"  parameters: total={params['total']}, trainable={params['trainable']}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        output_mode=args.output_mode,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=False,
        grad_clip_norm=1.0,
        checkpoint_dir="checkpoints",
        log_dir="logs",
        experiment_name=f"siamese_cnn_smoke_{args.output_mode}",
        metric_for_best="val_total_loss",
    )
    history = trainer.fit(args.epochs)
    print("Smoke training completed.")
    print(f"Latest checkpoint: checkpoints/siamese_cnn_smoke_{args.output_mode}_latest.pt")
    print(f"Best checkpoint: checkpoints/siamese_cnn_smoke_{args.output_mode}_best.pt")
    print(f"History CSV: logs/siamese_cnn_smoke_{args.output_mode}_history.csv")
    if history:
        last = history[-1]
        print(f"Final train_loss: {last.get('train_total_loss'):.6f}")
        print(f"Final val_loss: {last.get('val_total_loss'):.6f}")


if __name__ == "__main__":
    main()
