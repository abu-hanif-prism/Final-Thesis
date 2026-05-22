"""Smoke-test shared multimodal Siamese model components on CPU."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_utils import count_parameters, validate_model_forward  # noqa: E402
from src.models.siamese_base import BaseMultimodalSiameseModel  # noqa: E402


class DummySiameseModel(BaseMultimodalSiameseModel):
    """Small test-only image encoder for validating shared Siamese components."""

    def __init__(self, output_mode: str) -> None:
        super().__init__(
            image_channels=13,
            tabular_dim=146,
            image_embedding_dim=256,
            tabular_embedding_dim=128,
            fusion_dim=256,
            output_mode=output_mode,
            num_classes=3,
            dropout=0.2,
        )
        self.image_encoder = nn.Sequential(
            nn.Conv2d(13, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, self.image_embedding_dim),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode one image into [B, image_embedding_dim]."""
        return self.image_encoder(image)


def main() -> None:
    torch.manual_seed(42)
    batch = {
        "image_t1": torch.randn(4, 13, 128, 128),
        "image_t2": torch.randn(4, 13, 128, 128),
        "tabular": torch.randn(4, 146),
    }

    print("Input shapes:")
    print(f"  image_t1: {tuple(batch['image_t1'].shape)}")
    print(f"  image_t2: {tuple(batch['image_t2'].shape)}")
    print(f"  tabular: {tuple(batch['tabular'].shape)}")

    for output_mode in ["regression", "classification", "multitask"]:
        print(f"\nTesting output_mode={output_mode}")
        model = DummySiameseModel(output_mode=output_mode)
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

        print_outputs(output_mode, outputs)
        print(f"  return_features keys: {list(features.keys())}")
        print(f"  img_emb_t1 shape: {tuple(features['img_emb_t1'].shape)}")
        print(f"  img_emb_t2 shape: {tuple(features['img_emb_t2'].shape)}")
        print(f"  tab_emb shape: {tuple(features['tab_emb'].shape)}")
        print(f"  fused_emb shape: {tuple(features['fused_emb'].shape)}")

        validation = validate_model_forward(model, batch, output_mode=output_mode)
        print(f"  validation is_valid: {validation['is_valid']}")
        print(f"  validation output_shapes: {validation['output_shapes']}")
        print(f"  finite check: {validation['finite']}")
        if validation["errors"]:
            print(f"  validation errors: {validation['errors']}")


def print_outputs(output_mode: str, outputs: torch.Tensor | dict[str, torch.Tensor]) -> None:
    """Print output shape details for one mode."""
    if output_mode == "regression":
        print(f"  regression output shape: {tuple(outputs.shape)}")
    elif output_mode == "classification":
        print(f"  classification output shape: {tuple(outputs.shape)}")
    else:
        assert isinstance(outputs, dict)
        print(f"  multitask output keys: {list(outputs.keys())}")
        print(f"  change_ratio_pred shape: {tuple(outputs['change_ratio_pred'].shape)}")
        print(f"  change_class_logits shape: {tuple(outputs['change_class_logits'].shape)}")


if __name__ == "__main__":
    main()
