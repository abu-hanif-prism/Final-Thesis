"""Base interface for multimodal Siamese change-detection models."""

from __future__ import annotations

import torch
from torch import nn

from src.models.common_blocks import (
    ClassificationHead,
    MultiTaskHead,
    RegressionHead,
    SiameseFeatureFusion,
    TabularEncoder,
)


class BaseMultimodalSiameseModel(nn.Module):
    """Shared tabular, fusion, and head logic for future Siamese backbones."""

    SUPPORTED_OUTPUT_MODES = {"regression", "classification", "multitask"}

    def __init__(
        self,
        image_channels: int = 13,
        tabular_dim: int = 146,
        image_embedding_dim: int = 256,
        tabular_embedding_dim: int = 128,
        fusion_dim: int = 256,
        output_mode: str = "regression",
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if output_mode not in self.SUPPORTED_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {sorted(self.SUPPORTED_OUTPUT_MODES)}. "
                f"Got {output_mode!r}."
            )
        self.image_channels = int(image_channels)
        self.tabular_dim = int(tabular_dim)
        self.image_embedding_dim = int(image_embedding_dim)
        self.tabular_embedding_dim = int(tabular_embedding_dim)
        self.fusion_dim = int(fusion_dim)
        self.output_mode = output_mode
        self.num_classes = int(num_classes)

        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_dim,
            output_dim=tabular_embedding_dim,
            dropout=dropout,
        )
        self.fusion = SiameseFeatureFusion(
            image_embedding_dim=image_embedding_dim,
            tabular_embedding_dim=tabular_embedding_dim,
            fusion_dim=fusion_dim,
            fusion_type="concat_diff_absdiff_product",
            dropout=dropout,
        )
        if output_mode == "regression":
            self.head = RegressionHead(fusion_dim, dropout=dropout)
        elif output_mode == "classification":
            self.head = ClassificationHead(fusion_dim, num_classes=num_classes, dropout=dropout)
        else:
            self.head = MultiTaskHead(fusion_dim, num_classes=num_classes, dropout=dropout)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode one image tensor. Future backbones must override this method."""
        raise NotImplementedError("Subclasses must implement encode_image(image).")

    def forward(
        self,
        image_t1: torch.Tensor,
        image_t2: torch.Tensor,
        tabular: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor] | dict[str, object]:
        """Run shared Siamese forward pass."""
        img_emb_t1 = self.encode_image(image_t1)
        img_emb_t2 = self.encode_image(image_t2)
        tab_emb = self.tabular_encoder(tabular)
        fused_emb = self.fusion(img_emb_t1, img_emb_t2, tab_emb)
        outputs = self.head(fused_emb)

        if not return_features:
            return outputs
        return {
            "outputs": outputs,
            "img_emb_t1": img_emb_t1,
            "img_emb_t2": img_emb_t2,
            "tab_emb": tab_emb,
            "fused_emb": fused_emb,
        }
