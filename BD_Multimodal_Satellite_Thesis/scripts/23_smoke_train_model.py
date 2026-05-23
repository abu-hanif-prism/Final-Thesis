"""Smoke training for any supported multimodal Siamese model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse smoke-training arguments."""
    parser = argparse.ArgumentParser(description="Smoke train any supported model.")
    parser.add_argument("--model_name", default="cnn")
    parser.add_argument("--output_mode", choices=["regression", "classification", "multitask"], default="regression")
    parser.add_argument("--index_path", default="data/npz/final_npz_index.parquet")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_limit", type=int, default=64)
    parser.add_argument("--val_limit", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--single_batch_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run with traceback reporting for normal Python exceptions."""
    try:
        run_main()
    except Exception:
        print("Smoke training failed with Python exception:", flush=True)
        traceback.print_exc()
        raise


def run_main() -> None:
    """Run a small smoke training job."""
    args = parse_args()
    debug_print(args.debug, f"1. parsed args: {args}")

    try:
        debug_print(args.debug, "2. importing dataset/training modules")
        import torch
        from torch.utils.data import DataLoader, Subset

        from src.models.model_factory import create_model, normalize_model_name
        from src.models.model_utils import count_parameters
        from src.training.losses import compute_loss, get_loss_function
        from src.training.npz_dataset import NPZSiameseDataset
        from src.training.trainer import Trainer
        from src.training.train_utils import (
            create_optimizer,
            create_scheduler,
            get_device,
            move_batch_to_device,
            set_random_seed,
        )
    except ImportError as exc:
        print(f"Smoke training cannot start: missing dependency ({exc})")
        return

    set_random_seed(42)
    model_name = normalize_model_name(args.model_name)
    target_mode = "both" if args.output_mode == "multitask" else args.output_mode
    experiment_name = args.experiment_name or f"{model_name}_{args.output_mode}_smoke"

    try:
        debug_print(args.debug, "3. creating train dataset")
        train_dataset = NPZSiameseDataset(args.index_path, split="train", target_mode=target_mode)
        debug_print(args.debug, "4. creating val dataset")
        val_dataset = NPZSiameseDataset(args.index_path, split="val", target_mode=target_mode)
    except Exception as exc:
        print(f"Smoke training cannot load NPZ index/dataset: {exc}")
        raise

    debug_print(args.debug, "5. limiting datasets")
    train_subset = limit_dataset(Subset, train_dataset, args.train_limit)
    val_subset = limit_dataset(Subset, val_dataset, args.val_limit)
    debug_print(args.debug, "6. creating dataloaders")
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device) if args.device != "auto" else get_device(prefer_gpu=True)
    mixed_precision = bool(args.mixed_precision and device.type == "cuda")
    if args.mixed_precision and not mixed_precision:
        print("Mixed precision requested but CUDA is unavailable; disabling mixed precision.")

    debug_print(args.debug, "7. creating model")
    model = create_model(
        model_name=model_name,
        output_mode=args.output_mode,
        **build_model_kwargs(args, model_name),
    )
    if args.debug:
        debug_print(args.debug, "8. moving model to device")
        model.to(device)

    debug_print(args.debug, "9. creating loss")
    loss_fn = get_loss_function(args.output_mode)
    debug_print(args.debug, "10. creating optimizer")
    optimizer = create_optimizer(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, scheduler_type="plateau")
    params = count_parameters(model)

    print("Unified smoke training configuration:")
    print(f"  model_name: {model_name}")
    print(f"  output_mode: {args.output_mode}")
    print(f"  device: {device}")
    print(f"  train subset: {len(train_subset)}")
    print(f"  val subset: {len(val_subset)}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  epochs: {args.epochs}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  mixed_precision: {mixed_precision}")
    print(f"  parameters: total={params['total']}, trainable={params['trainable']}")

    if args.debug:
        run_single_batch_debug(
            model=model,
            train_dataset=train_dataset,
            train_loader=train_loader,
            loss_fn=loss_fn,
            output_mode=args.output_mode,
            device=device,
            move_batch_to_device=move_batch_to_device,
            compute_loss=compute_loss,
        )
        if args.single_batch_only:
            print("single_batch_only requested; exiting before Trainer.fit.", flush=True)
            return

    debug_print(args.debug, "11. creating trainer")
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
        grad_clip_norm=1.0,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        experiment_name=experiment_name,
        metric_for_best="val_loss",
    )
    debug_print(args.debug, "12. starting fit")
    history = trainer.fit(args.epochs)
    debug_print(args.debug, "13. finished fit")
    print("Smoke training completed.")
    print(f"Latest checkpoint: {Path(args.checkpoint_dir) / f'{experiment_name}_latest.pt'}")
    print(f"Best checkpoint: {Path(args.checkpoint_dir) / f'{experiment_name}_best.pt'}")
    print(f"History CSV: {Path(args.log_dir) / f'{experiment_name}_history.csv'}")
    if history:
        last = history[-1]
        print(f"Final train_loss: {last.get('train_loss'):.6f}")
        print(f"Final val_loss: {last.get('val_loss'):.6f}")


def debug_print(enabled: bool, message: str) -> None:
    """Print a flushed debug checkpoint when enabled."""
    if enabled:
        print(message, flush=True)


def run_single_batch_debug(
    model,
    train_dataset,
    train_loader,
    loss_fn,
    output_mode: str,
    device,
    move_batch_to_device,
    compute_loss,
) -> None:
    """Run one explicit sample/batch/forward/loss/backward debug pass."""
    import torch

    print("debug: loading one train sample", flush=True)
    sample = train_dataset[0]
    print(
        "debug: sample shapes "
        f"image_t1={tuple(sample['image_t1'].shape)} "
        f"tabular={tuple(sample['tabular'].shape)} "
        f"target={sample['target']}",
        flush=True,
    )

    print("debug: loading one train DataLoader batch", flush=True)
    batch = next(iter(train_loader))
    print(
        "debug: batch shapes "
        f"image_t1={tuple(batch['image_t1'].shape)} "
        f"tabular={tuple(batch['tabular'].shape)} "
        f"target={tuple(batch['target'].shape) if isinstance(batch['target'], torch.Tensor) else 'dict'}",
        flush=True,
    )

    batch = move_batch_to_device(batch, device)
    print("batch moved to device", flush=True)
    model.train()
    model.zero_grad(set_to_none=True)
    outputs = model(batch["image_t1"], batch["image_t2"], batch["tabular"])
    print(f"forward output shape: {format_output_shape(outputs)}", flush=True)
    loss_dict = compute_loss(outputs, batch, output_mode, loss_fn)
    loss = loss_dict["total_loss"]
    print(f"loss: {float(loss.detach().cpu().item()):.6f}", flush=True)
    loss.backward()
    print("single batch backward ok", flush=True)
    model.zero_grad(set_to_none=True)


def format_output_shape(outputs) -> object:
    """Return a readable output shape report."""
    import torch

    if isinstance(outputs, torch.Tensor):
        return tuple(outputs.shape)
    if isinstance(outputs, dict):
        return {
            key: tuple(value.shape)
            for key, value in outputs.items()
            if isinstance(value, torch.Tensor)
        }
    return type(outputs).__name__


def build_model_kwargs(args: argparse.Namespace, model_name: str) -> dict[str, object]:
    """Build model-specific kwargs from CLI args."""
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


def limit_dataset(subset_cls, dataset, limit: int):
    """Return a Subset capped to limit samples."""
    return subset_cls(dataset, range(min(int(limit), len(dataset))))


if __name__ == "__main__":
    main()
