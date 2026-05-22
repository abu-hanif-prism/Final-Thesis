"""Smoke-test the unified multimodal Siamese model factory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_factory import (  # noqa: E402
    create_model,
    create_model_from_config,
    get_model_default_config,
    list_supported_models,
    load_model_config,
    normalize_model_name,
    save_model_config,
)
from src.models.model_utils import count_parameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test unified model factory.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--skip_heavy", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run model factory tests."""
    args = parse_args()
    torch.manual_seed(42)
    supported = list_supported_models()
    print(f"Supported canonical models: {supported['canonical']}")
    print(f"Supported aliases: {supported['aliases']}")

    fake_batch = {
        "image_t1": torch.randn(args.batch_size, 13, 128, 128),
        "image_t2": torch.randn(args.batch_size, 13, 128, 128),
        "tabular": torch.randn(args.batch_size, 146),
    }
    model_names = ["cnn", "convnext"] if args.skip_heavy else ["cnn", "swin", "convnext", "maxvit"]

    for model_name in model_names:
        print(f"\nTesting canonical model: {model_name}")
        model = create_model(model_name, output_mode="regression")
        test_model_forward(model, fake_batch, "regression")

    print("\nTesting CNN output modes:")
    for output_mode in ["regression", "classification", "multitask"]:
        model = create_model("cnn", output_mode=output_mode)
        test_model_forward(model, fake_batch, output_mode)

    print("\nTesting aliases:")
    for alias in ["multimodal_cnn", "siamese_swin", "multimodal_convnext", "siamese_maxvit"]:
        canonical = normalize_model_name(alias)
        model = create_model(alias, output_mode="regression")
        output = forward_output(model, fake_batch)
        print(f"  {alias} -> {canonical}: output shape={shape_tree(output)}, finite={all_finite(output)}")

    print("\nTesting config save/load:")
    config = get_model_default_config("cnn")
    config["output_mode"] = "regression"
    config_path = Path("outputs/reports/test_model_config.json")
    save_model_config(config, config_path)
    loaded_config = load_model_config(config_path)
    loaded_model = create_model_from_config(loaded_config)
    loaded_output = forward_output(loaded_model, fake_batch)
    print(f"  config path: {config_path}")
    print(f"  loaded model output shape: {shape_tree(loaded_output)}")
    print(f"  loaded model finite: {all_finite(loaded_output)}")
    print("\nModel factory test completed successfully.")


def test_model_forward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    output_mode: str,
) -> None:
    """Run one forward pass and print model diagnostics."""
    params = count_parameters(model)
    model.eval()
    with torch.no_grad():
        outputs = model(batch["image_t1"], batch["image_t2"], batch["tabular"])
        features = model(
            batch["image_t1"],
            batch["image_t2"],
            batch["tabular"],
            return_features=True,
        )
    print(f"  parameter count: total={params['total']}, trainable={params['trainable']}")
    print(f"  output_mode: {output_mode}")
    print(f"  output shape: {shape_tree(outputs)}")
    print(f"  output finite: {all_finite(outputs)}")
    print("  feature shapes:")
    print(f"    img_emb_t1: {tuple(features['img_emb_t1'].shape)}")
    print(f"    img_emb_t2: {tuple(features['img_emb_t2'].shape)}")
    print(f"    tab_emb: {tuple(features['tab_emb'].shape)}")
    print(f"    fused_emb: {tuple(features['fused_emb'].shape)}")


def forward_output(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
    """Run no-grad forward pass."""
    model.eval()
    with torch.no_grad():
        return model(batch["image_t1"], batch["image_t2"], batch["tabular"])


def shape_tree(value: Any) -> Any:
    """Return tensor shape or nested tensor shapes."""
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, dict):
        return {key: shape_tree(item) for key, item in value.items()}
    return type(value).__name__


def all_finite(value: Any) -> bool:
    """Return True when all output tensors are finite."""
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    return False


if __name__ == "__main__":
    main()
