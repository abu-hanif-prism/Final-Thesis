"""Utility helpers for shared model validation and parameter control."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def freeze_module(module: nn.Module) -> None:
    """Disable gradients for all parameters in a module."""
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """Enable gradients for all parameters in a module."""
    for parameter in module.parameters():
        parameter.requires_grad = True


def get_model_output_shapes(model: nn.Module, batch: dict[str, torch.Tensor]) -> Any:
    """Run a no-grad forward pass and return output shapes."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(batch["image_t1"], batch["image_t2"], batch["tabular"])
    if was_training:
        model.train()
    return _shape_tree(outputs)


def validate_model_forward(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    output_mode: str,
) -> dict[str, Any]:
    """Validate forward output shapes and finite values for one batch."""
    if output_mode not in {"regression", "classification", "multitask"}:
        raise ValueError("output_mode must be one of: regression, classification, multitask")

    batch_size = int(batch["image_t1"].shape[0])
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(batch["image_t1"], batch["image_t2"], batch["tabular"])
    if was_training:
        model.train()

    validation: dict[str, Any] = {
        "output_mode": output_mode,
        "batch_size": batch_size,
        "output_shapes": _shape_tree(outputs),
        "is_valid": True,
        "errors": [],
        "finite": _finite_tree(outputs),
    }

    if output_mode == "regression":
        _check_tensor_shape(outputs, (batch_size,), "regression output", validation)
    elif output_mode == "classification":
        _check_tensor_shape(outputs, (batch_size, 3), "classification output", validation)
    else:
        if not isinstance(outputs, dict):
            validation["is_valid"] = False
            validation["errors"].append("multitask output must be a dictionary")
        else:
            _check_tensor_shape(
                outputs.get("change_ratio_pred"),
                (batch_size,),
                "change_ratio_pred",
                validation,
            )
            _check_tensor_shape(
                outputs.get("change_class_logits"),
                (batch_size, 3),
                "change_class_logits",
                validation,
            )

    if not _all_finite(outputs):
        validation["is_valid"] = False
        validation["errors"].append("model output contains NaN or inf")
    return validation


def _check_tensor_shape(
    tensor: Any,
    expected_shape: tuple[int, ...],
    name: str,
    validation: dict[str, Any],
) -> None:
    if not isinstance(tensor, torch.Tensor):
        validation["is_valid"] = False
        validation["errors"].append(f"{name} is not a tensor")
        return
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        validation["is_valid"] = False
        validation["errors"].append(
            f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}"
        )


def _shape_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, dict):
        return {key: _shape_tree(item) for key, item in value.items()}
    return type(value).__name__


def _finite_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return {key: _finite_tree(item) for key, item in value.items()}
    return None


def _all_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    return False
