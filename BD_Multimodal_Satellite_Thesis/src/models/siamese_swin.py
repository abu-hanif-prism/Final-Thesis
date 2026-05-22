"""Multimodal Siamese Swin-style Transformer model."""

from __future__ import annotations

import torch
from torch import nn

from src.models.siamese_base import BaseMultimodalSiameseModel


class PatchEmbedding(nn.Module):
    """Convert a channel-first Sentinel image patch into patch tokens."""

    def __init__(
        self,
        in_channels: int = 13,
        embed_dim: int = 96,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return patch tokens with shape [B, num_patches, embed_dim]."""
        tokens = self.projection(image)
        return tokens.flatten(2).transpose(1, 2)


class SimpleSwinBlock(nn.Module):
    """Lightweight Swin-like transformer block with global self-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return updated token sequence with residual connections."""
        norm_tokens = self.norm1(tokens)
        attended, _ = self.attention(norm_tokens, norm_tokens, norm_tokens, need_weights=False)
        tokens = tokens + self.dropout1(attended)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class SwinImageEncoder(nn.Module):
    """Encode one Sentinel image patch into a fixed-size embedding."""

    def __init__(
        self,
        in_channels: int = 13,
        image_embedding_dim: int = 256,
        embed_dim: int = 96,
        patch_size: int = 4,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        image_size: int = 128,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        num_patches_per_side = image_size // patch_size
        num_patches = num_patches_per_side * num_patches_per_side
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.position_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[
                SimpleSwinBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, image_embedding_dim),
            nn.LayerNorm(image_embedding_dim),
            nn.Dropout(dropout),
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return image embedding with shape [B, image_embedding_dim]."""
        tokens = self.patch_embedding(image)
        if tokens.shape[1] != self.position_embedding.shape[1]:
            raise ValueError(
                "Unexpected number of patch tokens: "
                f"got {tokens.shape[1]}, expected {self.position_embedding.shape[1]}"
            )
        tokens = self.position_dropout(tokens + self.position_embedding)
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        return self.projection(pooled)


class MultimodalSiameseSwin(BaseMultimodalSiameseModel):
    """Swin-style image backbone with shared multimodal Siamese heads."""

    def __init__(
        self,
        image_channels: int = 13,
        tabular_dim: int = 146,
        image_embedding_dim: int = 256,
        tabular_embedding_dim: int = 128,
        fusion_dim: int = 256,
        output_mode: str = "regression",
        num_classes: int = 3,
        embed_dim: int = 96,
        patch_size: int = 4,
        depth: int = 2,
        num_heads: int = 4,
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
        self.image_encoder = SwinImageEncoder(
            in_channels=image_channels,
            image_embedding_dim=image_embedding_dim,
            embed_dim=embed_dim,
            patch_size=patch_size,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode one image patch using the Swin-style image encoder."""
        return self.image_encoder(image)


def create_siamese_swin_model(
    output_mode: str = "regression",
    image_channels: int = 13,
    tabular_dim: int = 146,
    image_embedding_dim: int = 256,
    tabular_embedding_dim: int = 128,
    fusion_dim: int = 256,
    num_classes: int = 3,
    embed_dim: int = 96,
    patch_size: int = 4,
    depth: int = 2,
    num_heads: int = 4,
    dropout: float = 0.2,
) -> MultimodalSiameseSwin:
    """Create a multimodal Siamese Swin-style model."""
    return MultimodalSiameseSwin(
        image_channels=image_channels,
        tabular_dim=tabular_dim,
        image_embedding_dim=image_embedding_dim,
        tabular_embedding_dim=tabular_embedding_dim,
        fusion_dim=fusion_dim,
        output_mode=output_mode,
        num_classes=num_classes,
        embed_dim=embed_dim,
        patch_size=patch_size,
        depth=depth,
        num_heads=num_heads,
        dropout=dropout,
    )
