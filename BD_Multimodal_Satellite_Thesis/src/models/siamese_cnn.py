"""Baseline multimodal Siamese CNN model."""

from __future__ import annotations

import torch
from torch import nn

from src.models.siamese_base import BaseMultimodalSiameseModel


def _activation_layer(name: str) -> nn.Module:
    """Create an activation layer for convolution blocks."""
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU(inplace=True)
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError("activation must be one of: relu, gelu, silu")


class ConvBlock(nn.Module):
    """Reusable Conv2d block with optional batch norm and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batchnorm: bool = True,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_batchnorm,
            )
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(_activation_layer(activation))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return encoded feature map."""
        return self.block(x)


class CNNImageEncoder(nn.Module):
    """Encode one Sentinel-2 patch into a fixed-size image embedding."""

    def __init__(
        self,
        in_channels: int = 13,
        image_embedding_dim: int = 256,
        base_channels: int = 32,
        dropout: float = 0.2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        conv_dropout = min(float(dropout), 0.1)

        self.features = nn.Sequential(
            ConvBlock(in_channels, c1, activation=activation, dropout=conv_dropout),
            ConvBlock(c1, c1, activation=activation, dropout=conv_dropout),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(c1, c2, activation=activation, dropout=conv_dropout),
            ConvBlock(c2, c2, activation=activation, dropout=conv_dropout),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(c2, c3, activation=activation, dropout=conv_dropout),
            ConvBlock(c3, c3, activation=activation, dropout=conv_dropout),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(c3, c4, activation=activation, dropout=conv_dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(c4, image_embedding_dim),
            nn.LayerNorm(image_embedding_dim),
            _activation_layer(activation),
            nn.Dropout(dropout),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return image embedding with shape [B, image_embedding_dim]."""
        features = self.features(image)
        return self.projection(features)


class MultimodalSiameseCNN(BaseMultimodalSiameseModel):
    """Baseline CNN image backbone with shared multimodal Siamese heads."""

    def __init__(
        self,
        image_channels: int = 13,
        tabular_dim: int = 146,
        image_embedding_dim: int = 256,
        tabular_embedding_dim: int = 128,
        fusion_dim: int = 256,
        output_mode: str = "regression",
        num_classes: int = 3,
        base_channels: int = 32,
        dropout: float = 0.2,
        activation: str = "relu",
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
        self.image_encoder = CNNImageEncoder(
            in_channels=image_channels,
            image_embedding_dim=image_embedding_dim,
            base_channels=base_channels,
            dropout=dropout,
            activation=activation,
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode one image patch using the CNN image encoder."""
        return self.image_encoder(image)


def create_siamese_cnn_model(
    output_mode: str = "regression",
    image_channels: int = 13,
    tabular_dim: int = 146,
    image_embedding_dim: int = 256,
    tabular_embedding_dim: int = 128,
    fusion_dim: int = 256,
    num_classes: int = 3,
    base_channels: int = 32,
    dropout: float = 0.2,
) -> MultimodalSiameseCNN:
    """Create a baseline multimodal Siamese CNN model."""
    return MultimodalSiameseCNN(
        image_channels=image_channels,
        tabular_dim=tabular_dim,
        image_embedding_dim=image_embedding_dim,
        tabular_embedding_dim=tabular_embedding_dim,
        fusion_dim=fusion_dim,
        output_mode=output_mode,
        num_classes=num_classes,
        base_channels=base_channels,
        dropout=dropout,
    )
