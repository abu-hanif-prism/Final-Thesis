"""Multimodal Siamese MaxViT-inspired model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.siamese_base import BaseMultimodalSiameseModel


class ConvStem(nn.Module):
    """Initial convolutional feature extractor for Sentinel patches."""

    def __init__(
        self,
        in_channels: int = 13,
        out_channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.stem = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return stem features with shape [B, out_channels, 64, 64]."""
        return self.stem(image)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel recalibration."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale input channels by learned global context."""
        return x * self.excitation(self.pool(x))


class MBConvBlock(nn.Module):
    """Mobile inverted bottleneck convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        expanded_channels = in_channels * expansion_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.GELU(),
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.GELU(),
            SqueezeExcitation(expanded_channels),
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return MBConv output with optional residual connection."""
        output = self.dropout(self.block(x))
        if self.use_residual:
            return x + output
        return output


class GridAttentionBlock(nn.Module):
    """Simplified global/grid attention over flattened spatial tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply token attention and reshape back to [B, C, H, W]."""
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        norm_tokens = self.norm1(tokens)
        attended, _ = self.attention(norm_tokens, norm_tokens, norm_tokens, need_weights=False)
        tokens = tokens + self.dropout1(attended)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)


class WindowAttentionBlock(nn.Module):
    """Local non-overlapping window attention block."""

    def __init__(
        self,
        dim: int,
        window_size: int = 8,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.window_size = int(window_size)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention inside local windows and preserve [B, C, H, W]."""
        batch_size, channels, height, width = x.shape
        padded, pad_h, pad_w = _pad_to_window_size(x, self.window_size)
        _, _, padded_h, padded_w = padded.shape
        windows = _partition_windows(padded, self.window_size)
        tokens = windows.reshape(-1, channels, self.window_size * self.window_size).transpose(1, 2)
        norm_tokens = self.norm1(tokens)
        attended, _ = self.attention(norm_tokens, norm_tokens, norm_tokens, need_weights=False)
        tokens = tokens + self.dropout1(attended)
        tokens = tokens + self.mlp(self.norm2(tokens))
        windows = tokens.transpose(1, 2).reshape(-1, channels, self.window_size, self.window_size)
        merged = _merge_windows(windows, batch_size, channels, padded_h, padded_w, self.window_size)
        if pad_h or pad_w:
            merged = merged[:, :, :height, :width]
        return merged


class MaxViTBlock(nn.Module):
    """MaxViT-style block combining MBConv, window attention, and grid attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mbconv = MBConvBlock(dim, dim, dropout=dropout)
        self.window_attention = WindowAttentionBlock(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.grid_attention = GridAttentionBlock(dim=dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return same-shape MaxViT-style block output."""
        x = self.mbconv(x)
        x = self.window_attention(x)
        x = self.grid_attention(x)
        return x


class MaxViTImageEncoder(nn.Module):
    """Encode one Sentinel image patch into a fixed-size embedding."""

    def __init__(
        self,
        in_channels: int = 13,
        image_embedding_dim: int = 256,
        dims: list[int] | tuple[int, int, int] = (64, 128, 256),
        depths: list[int] | tuple[int, int, int] = (1, 1, 1),
        num_heads: list[int] | tuple[int, int, int] = (4, 4, 8),
        window_size: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(dims) != 3 or len(depths) != 3 or len(num_heads) != 3:
            raise ValueError("dims, depths, and num_heads must each contain three values")
        self.stem = ConvStem(in_channels=in_channels, out_channels=dims[0], dropout=dropout)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        MaxViTBlock(
                            dim=dims[index],
                            num_heads=num_heads[index],
                            window_size=window_size,
                            dropout=dropout,
                        )
                        for _ in range(depths[index])
                    ]
                )
                for index in range(3)
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dims[index], dims[index + 1], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dims[index + 1]),
                    nn.GELU(),
                )
                for index in range(2)
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


class MultimodalSiameseMaxViT(BaseMultimodalSiameseModel):
    """MaxViT-inspired image backbone with shared multimodal Siamese heads."""

    def __init__(
        self,
        image_channels: int = 13,
        tabular_dim: int = 146,
        image_embedding_dim: int = 256,
        tabular_embedding_dim: int = 128,
        fusion_dim: int = 256,
        output_mode: str = "regression",
        num_classes: int = 3,
        dims: list[int] | tuple[int, int, int] = (64, 128, 256),
        depths: list[int] | tuple[int, int, int] = (1, 1, 1),
        num_heads: list[int] | tuple[int, int, int] = (4, 4, 8),
        window_size: int = 8,
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
        self.image_encoder = MaxViTImageEncoder(
            in_channels=image_channels,
            image_embedding_dim=image_embedding_dim,
            dims=dims,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode one image patch using the MaxViT-inspired image encoder."""
        return self.image_encoder(image)


def create_siamese_maxvit_model(
    output_mode: str = "regression",
    image_channels: int = 13,
    tabular_dim: int = 146,
    image_embedding_dim: int = 256,
    tabular_embedding_dim: int = 128,
    fusion_dim: int = 256,
    num_classes: int = 3,
    dims: list[int] | tuple[int, int, int] = (64, 128, 256),
    depths: list[int] | tuple[int, int, int] = (1, 1, 1),
    num_heads: list[int] | tuple[int, int, int] = (4, 4, 8),
    window_size: int = 8,
    dropout: float = 0.2,
) -> MultimodalSiameseMaxViT:
    """Create a multimodal Siamese MaxViT-inspired model."""
    return MultimodalSiameseMaxViT(
        image_channels=image_channels,
        tabular_dim=tabular_dim,
        image_embedding_dim=image_embedding_dim,
        tabular_embedding_dim=tabular_embedding_dim,
        fusion_dim=fusion_dim,
        output_mode=output_mode,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        dropout=dropout,
    )


def _pad_to_window_size(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, int, int]:
    _, _, height, width = x.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, pad_h, pad_w


def _partition_windows(x: torch.Tensor, window_size: int) -> torch.Tensor:
    batch_size, channels, height, width = x.shape
    x = x.view(batch_size, channels, height // window_size, window_size, width // window_size, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(-1, channels, window_size, window_size)


def _merge_windows(
    windows: torch.Tensor,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    window_size: int,
) -> torch.Tensor:
    x = windows.view(batch_size, height // window_size, width // window_size, channels, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(batch_size, channels, height, width)
