"""Shared neural network blocks for multimodal Siamese models."""

from __future__ import annotations

import torch
from torch import nn


def _activation_layer(name: str) -> nn.Module:
    """Create an activation layer from a small supported set."""
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU(inplace=True)
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError("activation must be one of: relu, gelu, silu")


class MLPBlock(nn.Module):
    """Generic fully connected block with optional normalization and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        output_dim: int,
        dropout: float = 0.2,
        activation: str = "gelu",
        use_batchnorm: bool = True,
        norm_type: str | None = "layernorm",
    ) -> None:
        super().__init__()
        resolved_norm_type = self._resolve_norm_type(use_batchnorm, norm_type)
        dims = [int(input_dim), *[int(dim) for dim in hidden_dims], int(output_dim)]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            in_dim = dims[index]
            out_dim = dims[index + 1]
            is_hidden = index < len(dims) - 2
            layers.append(nn.Linear(in_dim, out_dim))
            if is_hidden:
                norm_layer = self._normalization_layer(out_dim, resolved_norm_type)
                if norm_layer is not None:
                    layers.append(norm_layer)
                layers.append(_activation_layer(activation))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the MLP output tensor."""
        return self.net(x)

    @staticmethod
    def _resolve_norm_type(use_batchnorm: bool, norm_type: str | None) -> str:
        """Resolve legacy use_batchnorm into the new normalization option."""
        if not use_batchnorm:
            return "none"
        if norm_type is None:
            return "layernorm"
        normalized = norm_type.lower()
        if normalized not in {"layernorm", "batchnorm", "none"}:
            raise ValueError("norm_type must be one of: layernorm, batchnorm, none")
        return normalized

    @staticmethod
    def _normalization_layer(dim: int, norm_type: str) -> nn.Module | None:
        """Create a hidden-layer normalization module."""
        if norm_type == "layernorm":
            return nn.LayerNorm(dim)
        if norm_type == "batchnorm":
            return nn.BatchNorm1d(dim)
        return None


class TabularEncoder(nn.Module):
    """Encode a tabular feature vector into a compact embedding."""

    def __init__(
        self,
        input_dim: int = 146,
        hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        output_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = MLPBlock(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation="gelu",
            use_batchnorm=True,
            norm_type="layernorm",
        )

    def forward(self, tabular: torch.Tensor) -> torch.Tensor:
        """Return tabular embedding with shape [B, output_dim]."""
        return self.encoder(tabular)


class SiameseFeatureFusion(nn.Module):
    """Fuse two image embeddings and one tabular embedding."""

    SUPPORTED_FUSION_TYPES = {"concat", "concat_diff", "concat_diff_absdiff_product"}

    def __init__(
        self,
        image_embedding_dim: int,
        tabular_embedding_dim: int,
        fusion_dim: int = 256,
        fusion_type: str = "concat_diff_absdiff_product",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if fusion_type not in self.SUPPORTED_FUSION_TYPES:
            raise ValueError(
                f"fusion_type must be one of {sorted(self.SUPPORTED_FUSION_TYPES)}. "
                f"Got {fusion_type!r}."
            )
        self.fusion_type = fusion_type
        self.image_embedding_dim = int(image_embedding_dim)
        self.tabular_embedding_dim = int(tabular_embedding_dim)
        self.fused_input_dim = self._compute_fused_input_dim()
        self.projection = MLPBlock(
            input_dim=self.fused_input_dim,
            hidden_dims=(fusion_dim,),
            output_dim=fusion_dim,
            dropout=dropout,
            activation="gelu",
            use_batchnorm=True,
            norm_type="layernorm",
        )

    def forward(
        self,
        img_t1: torch.Tensor,
        img_t2: torch.Tensor,
        tabular_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Return fused multimodal embedding."""
        diff = img_t2 - img_t1
        if self.fusion_type == "concat":
            features = [img_t1, img_t2, tabular_emb]
        elif self.fusion_type == "concat_diff":
            features = [img_t1, img_t2, diff, tabular_emb]
        else:
            features = [img_t1, img_t2, diff, diff.abs(), img_t1 * img_t2, tabular_emb]
        fused = torch.cat(features, dim=1)
        return self.projection(fused)

    def _compute_fused_input_dim(self) -> int:
        if self.fusion_type == "concat":
            image_factor = 2
        elif self.fusion_type == "concat_diff":
            image_factor = 3
        else:
            image_factor = 5
        return image_factor * self.image_embedding_dim + self.tabular_embedding_dim


class RegressionHead(nn.Module):
    """Predict scalar change ratio in [0, 1]."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.head = MLPBlock(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=dropout,
            activation="gelu",
            use_batchnorm=True,
            norm_type="layernorm",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return change-ratio prediction with shape [B]."""
        return torch.sigmoid(self.head(x)).squeeze(-1)


class ClassificationHead(nn.Module):
    """Predict low/medium/high change-class logits."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.head = MLPBlock(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_classes,
            dropout=dropout,
            activation="gelu",
            use_batchnorm=True,
            norm_type="layernorm",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits with shape [B, num_classes]."""
        return self.head(x)


class MultiTaskHead(nn.Module):
    """Predict both change ratio and change class."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.regression_head = RegressionHead(input_dim, hidden_dims, dropout)
        self.classification_head = ClassificationHead(input_dim, num_classes, hidden_dims, dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return multitask prediction dictionary."""
        return {
            "change_ratio_pred": self.regression_head(x),
            "change_class_logits": self.classification_head(x),
        }
