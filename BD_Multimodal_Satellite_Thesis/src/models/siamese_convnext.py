"""Multimodal Siamese ConvNeXt model."""

from __future__ import annotations

import torch
from torch import nn

from src.models.siamese_base import BaseMultimodalSiameseModel


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for channel-first image tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize [B, C, H, W] over the channel dimension."""
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    """Lightweight ConvNeXt-style residual block."""

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pointwise_mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ConvNeXt residual block output."""
        residual = x
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.pointwise_mlp(x)
        if self.gamma is not None:
            x = x * self.gamma[:, None, None]
        return residual + self.drop_path(x)


class ConvNeXtImageEncoder(nn.Module):
    """Encode one Sentinel patch into a fixed-size image embedding."""

    def __init__(
        self,
        in_channels: int = 13,
        image_embedding_dim: int = 256,
        depths: list[int] | tuple[int, int, int, int] = (2, 2, 3, 2),
        dims: list[int] | tuple[int, int, int, int] = (32, 64, 128, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("depths and dims must each contain four values")
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(*[ConvNeXtBlock(dims[index], dropout=dropout) for _ in range(depths[index])])
                for index in range(4)
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm2d(dims[index]),
                    nn.Conv2d(dims[index], dims[index + 1], kernel_size=2, stride=2),
                )
                for index in range(3)
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], image_embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return image embedding with shape [B, image_embedding_dim]."""
        x = self.stem(image)
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        x = self.pool(x)
        return self.head(x)


class MultimodalSiameseConvNeXt(BaseMultimodalSiameseModel):
    """ConvNeXt image backbone with shared multimodal Siamese heads."""

    def __init__(
        self,
        image_channels: int = 13,
        tabular_dim: int = 146,
        image_embedding_dim: int = 256,
        tabular_embedding_dim: int = 128,
        fusion_dim: int = 256,
        output_mode: str = "regression",
        num_classes: int = 3,
        depths: list[int] | tuple[int, int, int, int] = (2, 2, 3, 2),
        dims: list[int] | tuple[int, int, int, int] = (32, 64, 128, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__(
            image_channels=image_channels,
            tabular_dim=tabular_dim,
            image_embedding_dim=image_embedding_dim,
            tabular_embedding_dim=tabular_embedding_dim,
            fusion_dim=fusion_dim,
            output_mode=output_mode,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.image_encoder = ConvNeXtImageEncoder(
            in_channels=image_channels,
            image_embedding_dim=image_embedding_dim,
            depths=depths,
            dims=dims,
            dropout=dropout,
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode one image patch using the ConvNeXt image encoder."""
        return self.image_encoder(image)


def create_siamese_convnext_model(
    output_mode: str = "regression",
    image_channels: int = 13,
    tabular_dim: int = 146,
    image_embedding_dim: int = 256,
    tabular_embedding_dim: int = 128,
    fusion_dim: int = 256,
    num_classes: int = 3,
    depths: list[int] | tuple[int, int, int, int] = (2, 2, 3, 2),
    dims: list[int] | tuple[int, int, int, int] = (32, 64, 128, 256),
    dropout: float = 0.2,
) -> MultimodalSiameseConvNeXt:
    """Create a multimodal Siamese ConvNeXt model."""
    return MultimodalSiameseConvNeXt(
        image_channels=image_channels,
        tabular_dim=tabular_dim,
        image_embedding_dim=image_embedding_dim,
        tabular_embedding_dim=tabular_embedding_dim,
        fusion_dim=fusion_dim,
        output_mode=output_mode,
        num_classes=num_classes,
        depths=depths,
        dims=dims,
        dropout=dropout,
    )
