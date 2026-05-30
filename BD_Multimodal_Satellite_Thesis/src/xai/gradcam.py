"""Grad-CAM utilities for CNN and ConvNeXt Siamese models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F


@dataclass
class GradCAMResult:
    """Grad-CAM result for a Siamese pair."""

    t1_cam: torch.Tensor
    t2_cam: torch.Tensor
    prediction: float
    target_layer_name: str


def is_gradcam_supported(model_name: str) -> bool:
    """Return whether Grad-CAM is supported for this model family."""
    return model_name.lower() in {"cnn", "convnext"}


def find_last_conv2d(model: torch.nn.Module) -> tuple[str, torch.nn.Conv2d]:
    """Find the last Conv2d layer in a model."""
    last_name = ""
    last_layer: torch.nn.Conv2d | None = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_name = name
            last_layer = module
    if last_layer is None:
        raise ValueError("No Conv2d layer found for Grad-CAM.")
    return last_name, last_layer


def compute_siamese_gradcam(
    model: torch.nn.Module,
    image_t1: torch.Tensor,
    image_t2: torch.Tensor,
    tabular: torch.Tensor,
    target_layer: torch.nn.Module | None = None,
    target_layer_name: str | None = None,
) -> GradCAMResult:
    """Compute Grad-CAM maps for image_t1 and image_t2.

    The Siamese image encoder is shared, so the same convolution layer is
    usually called once for t1 and once for t2. Forward-hook records are used
    to keep those two calls separate.
    """
    model.eval()
    if target_layer is None:
        target_layer_name, target_layer = find_last_conv2d(model)
    elif target_layer_name is None:
        target_layer_name = target_layer.__class__.__name__

    records: list[dict[str, torch.Tensor]] = []

    def forward_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        record: dict[str, torch.Tensor] = {"activation": output}

        def gradient_hook(gradient: torch.Tensor) -> None:
            record["gradient"] = gradient

        output.register_hook(gradient_hook)
        records.append(record)

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        output = model(image_t1, image_t2, tabular)
        prediction_tensor = _extract_prediction_tensor(output)
        prediction = float(prediction_tensor.detach().cpu().view(-1)[0].item())
        prediction_tensor.sum().backward()
    finally:
        handle.remove()

    if len(records) < 2:
        raise RuntimeError(
            f"Expected Grad-CAM hook to capture t1 and t2 activations, got {len(records)} records."
        )
    t1_cam = _make_cam(records[0]["activation"], records[0].get("gradient"), image_t1.shape[-2:])
    t2_cam = _make_cam(records[1]["activation"], records[1].get("gradient"), image_t2.shape[-2:])
    return GradCAMResult(
        t1_cam=t1_cam.detach().cpu(),
        t2_cam=t2_cam.detach().cpu(),
        prediction=prediction,
        target_layer_name=str(target_layer_name),
    )


def _make_cam(
    activation: torch.Tensor,
    gradient: torch.Tensor | None,
    output_size: tuple[int, int],
) -> torch.Tensor:
    if gradient is None:
        raise RuntimeError("Gradient was not captured for Grad-CAM.")
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
    cam = cam.squeeze(0).squeeze(0)
    min_value = cam.min()
    max_value = cam.max()
    if float((max_value - min_value).detach().cpu().item()) <= 1e-12:
        return torch.zeros_like(cam)
    return (cam - min_value) / (max_value - min_value)


def _extract_prediction_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output.view(-1)
    if isinstance(output, dict) and "change_ratio_pred" in output:
        return output["change_ratio_pred"].view(-1)
    raise TypeError(f"Unsupported model output type for Grad-CAM: {type(output).__name__}")
