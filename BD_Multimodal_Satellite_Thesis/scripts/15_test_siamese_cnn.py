"""Forward-pass tests for the baseline multimodal Siamese CNN."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_utils import count_parameters, validate_model_forward  # noqa: E402
from src.models.siamese_cnn import create_siamese_cnn_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test baseline multimodal Siamese CNN.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--output_mode",
        choices=["regression", "classification", "multitask"],
        default="regression",
    )
    parser.add_argument("--run_all_modes", action="store_true")
    parser.add_argument("--test_real_batch", action="store_true")
    parser.add_argument("--index_path", default="data/npz/final_npz_index.parquet")
    return parser.parse_args()


def main() -> None:
    """Run fake tensor tests and optional real DataLoader batch test."""
    args = parse_args()
    torch.manual_seed(42)
    modes = ["regression", "classification", "multitask"] if args.run_all_modes else [args.output_mode]
    fake_batch = {
        "image_t1": torch.randn(args.batch_size, 13, 128, 128),
        "image_t2": torch.randn(args.batch_size, 13, 128, 128),
        "tabular": torch.randn(args.batch_size, 146),
    }

    print("Fake batch shapes:")
    print(f"  image_t1: {tuple(fake_batch['image_t1'].shape)}")
    print(f"  image_t2: {tuple(fake_batch['image_t2'].shape)}")
    print(f"  tabular: {tuple(fake_batch['tabular'].shape)}")

    for mode in modes:
        print(f"\nTesting fake batch output_mode={mode}")
        model = create_siamese_cnn_model(output_mode=mode)
        test_model_on_batch(model, fake_batch, mode)

    if args.test_real_batch:
        test_real_batch(args.index_path, args.batch_size, args.output_mode)
    else:
        print("\nReal batch test skipped. Use --test_real_batch to enable it.")


def test_model_on_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    output_mode: str,
) -> None:
    """Run one model forward pass and print output/feature details."""
    params = count_parameters(model)
    print(f"  parameter count: total={params['total']}, trainable={params['trainable']}")

    model.eval()
    with torch.no_grad():
        outputs = model(batch["image_t1"], batch["image_t2"], batch["tabular"])
        features = model(
            batch["image_t1"],
            batch["image_t2"],
            batch["tabular"],
            return_features=True,
        )

    print_output_shapes(outputs, output_mode)
    print(f"  output finite: {all_finite(outputs)}")
    print("  return_features shapes:")
    print(f"    img_emb_t1: {tuple(features['img_emb_t1'].shape)}")
    print(f"    img_emb_t2: {tuple(features['img_emb_t2'].shape)}")
    print(f"    tab_emb: {tuple(features['tab_emb'].shape)}")
    print(f"    fused_emb: {tuple(features['fused_emb'].shape)}")

    validation = validate_model_forward(model, batch, output_mode)
    print(f"  validation is_valid: {validation['is_valid']}")
    print(f"  validation output_shapes: {validation['output_shapes']}")
    if validation["errors"]:
        print(f"  validation errors: {validation['errors']}")


def test_real_batch(index_path: str | Path, batch_size: int, output_mode: str) -> None:
    """Optionally test one batch from the NPZ DataLoader if dependencies are available."""
    print("\nTesting one real NPZ DataLoader batch:")
    index_path = Path(index_path)
    if not index_path.exists():
        print(f"  skipped: index file not found: {index_path}")
        return
    try:
        from src.training.dataloaders import create_train_val_test_dataloaders

        train_loader, _, _ = create_train_val_test_dataloaders(
            index_path=index_path,
            batch_size=batch_size,
            num_workers=0,
            target_mode=output_mode if output_mode != "multitask" else "both",
            pin_memory=False,
        )
        batch = next(iter(train_loader))
    except Exception as exc:
        print(f"  skipped: could not load real batch ({exc})")
        return

    model = create_siamese_cnn_model(output_mode=output_mode)
    model.eval()
    with torch.no_grad():
        outputs = model(batch["image_t1"], batch["image_t2"], batch["tabular"])
    print(f"  real batch image_t1 shape: {tuple(batch['image_t1'].shape)}")
    print_output_shapes(outputs, output_mode, prefix="  real batch")
    print(f"  real batch output finite: {all_finite(outputs)}")


def print_output_shapes(
    outputs: torch.Tensor | dict[str, torch.Tensor],
    output_mode: str,
    prefix: str = "  ",
) -> None:
    """Print output shapes for one output mode."""
    if output_mode == "regression":
        assert isinstance(outputs, torch.Tensor)
        print(f"{prefix}regression output shape: {tuple(outputs.shape)}")
    elif output_mode == "classification":
        assert isinstance(outputs, torch.Tensor)
        print(f"{prefix}classification output shape: {tuple(outputs.shape)}")
    else:
        assert isinstance(outputs, dict)
        print(f"{prefix}multitask output keys: {list(outputs.keys())}")
        print(f"{prefix}change_ratio_pred shape: {tuple(outputs['change_ratio_pred'].shape)}")
        print(f"{prefix}change_class_logits shape: {tuple(outputs['change_class_logits'].shape)}")


def all_finite(value: Any) -> bool:
    """Return True when all output tensors are finite."""
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    return False


if __name__ == "__main__":
    main()
