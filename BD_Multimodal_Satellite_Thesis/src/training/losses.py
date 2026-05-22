"""Reusable loss functions for regression, classification, and multitask models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class RegressionLoss(nn.Module):
    """Loss for change-ratio regression."""

    def __init__(self, loss_type: str = "mse", huber_delta: float = 0.1) -> None:
        super().__init__()
        if loss_type not in {"mse", "mae", "huber", "smooth_l1"}:
            raise ValueError("loss_type must be one of: mse, mae, huber, smooth_l1")
        self.loss_type = loss_type
        self.huber_delta = float(huber_delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return scalar regression loss."""
        pred = _squeeze_regression_tensor(pred)
        target = _squeeze_regression_tensor(target).to(dtype=pred.dtype, device=pred.device)
        if pred.shape != target.shape:
            raise ValueError(f"Regression pred/target shape mismatch: {pred.shape} vs {target.shape}")
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        if self.loss_type == "mae":
            return F.l1_loss(pred, target)
        if self.loss_type == "huber":
            return F.huber_loss(pred, target, delta=self.huber_delta)
        return F.smooth_l1_loss(pred, target, beta=self.huber_delta)


class ClassificationLoss(nn.Module):
    """Cross-entropy loss for low/medium/high change classification."""

    def __init__(
        self,
        class_weights: torch.Tensor | list[float] | tuple[float, ...] | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return scalar classification loss."""
        if logits.ndim != 2:
            raise ValueError(f"Classification logits must have shape [B, C]; got {tuple(logits.shape)}")
        target = target.to(device=logits.device, dtype=torch.long).view(-1)
        if logits.shape[0] != target.shape[0]:
            raise ValueError(f"Classification batch mismatch: {logits.shape[0]} vs {target.shape[0]}")
        weights = self.class_weights
        if weights is not None:
            weights = weights.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits, target, weight=weights, label_smoothing=self.label_smoothing)


class MultiTaskLoss(nn.Module):
    """Combined regression and classification loss."""

    def __init__(
        self,
        regression_weight: float = 1.0,
        classification_weight: float = 1.0,
        regression_loss_type: str = "mse",
        class_weights: torch.Tensor | list[float] | tuple[float, ...] | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.regression_weight = float(regression_weight)
        self.classification_weight = float(classification_weight)
        self.regression_loss = RegressionLoss(loss_type=regression_loss_type)
        self.classification_loss = ClassificationLoss(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return total, regression, and classification losses."""
        if "change_ratio_pred" not in outputs or "change_class_logits" not in outputs:
            raise KeyError("Multitask outputs must include change_ratio_pred and change_class_logits")
        if "change_ratio" not in batch or "change_class_id" not in batch:
            raise KeyError("Batch must include change_ratio and change_class_id")

        reg_loss = self.regression_loss(outputs["change_ratio_pred"], batch["change_ratio"])
        cls_loss = self.classification_loss(outputs["change_class_logits"], batch["change_class_id"])
        total = self.regression_weight * reg_loss + self.classification_weight * cls_loss
        return {
            "total_loss": total,
            "regression_loss": reg_loss,
            "classification_loss": cls_loss,
        }


def get_loss_function(output_mode: str, **kwargs: Any) -> nn.Module:
    """Create the appropriate loss object for an output mode."""
    if output_mode == "regression":
        return RegressionLoss(
            loss_type=kwargs.get("loss_type", kwargs.get("regression_loss_type", "mse")),
            huber_delta=kwargs.get("huber_delta", 0.1),
        )
    if output_mode == "classification":
        return ClassificationLoss(
            class_weights=kwargs.get("class_weights"),
            label_smoothing=kwargs.get("label_smoothing", 0.0),
        )
    if output_mode == "multitask":
        return MultiTaskLoss(
            regression_weight=kwargs.get("regression_weight", 1.0),
            classification_weight=kwargs.get("classification_weight", 1.0),
            regression_loss_type=kwargs.get("regression_loss_type", "mse"),
            class_weights=kwargs.get("class_weights"),
            label_smoothing=kwargs.get("label_smoothing", 0.0),
        )
    raise ValueError("output_mode must be one of: regression, classification, multitask")


def compute_loss(
    outputs: torch.Tensor | dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    output_mode: str,
    loss_fn: nn.Module,
) -> dict[str, torch.Tensor]:
    """Compute loss dictionary for regression, classification, or multitask output."""
    if output_mode == "regression":
        pred = outputs["change_ratio_pred"] if isinstance(outputs, dict) else outputs
        loss = loss_fn(pred, batch["change_ratio"])
        return {"total_loss": loss, "regression_loss": loss}
    if output_mode == "classification":
        logits = outputs["change_class_logits"] if isinstance(outputs, dict) else outputs
        loss = loss_fn(logits, batch["change_class_id"])
        return {"total_loss": loss, "classification_loss": loss}
    if output_mode == "multitask":
        if not isinstance(outputs, dict):
            raise TypeError("Multitask outputs must be a dictionary")
        return loss_fn(outputs, batch)
    raise ValueError("output_mode must be one of: regression, classification, multitask")


def _squeeze_regression_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Squeeze [B, 1] regression tensors to [B] while preserving scalar batches."""
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        return tensor.squeeze(1)
    return tensor.view(-1) if tensor.ndim == 0 else tensor
