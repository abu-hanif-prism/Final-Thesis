"""Occlusion sensitivity maps for Siamese image inputs."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def compute_occlusion_map(
    model: torch.nn.Module,
    image_t1: torch.Tensor,
    image_t2: torch.Tensor,
    tabular: torch.Tensor,
    target_image: str,
    patch_size: int = 16,
    stride: int = 16,
    mask_value: str = "zero",
) -> tuple[torch.Tensor, float]:
    """Compute a 128x128 occlusion sensitivity heatmap for image_t1 or image_t2.

    Importance is abs(full_prediction - masked_prediction) for each occluded
    spatial region. Overlapping windows are averaged.
    """
    if target_image not in {"t1", "t2"}:
        raise ValueError("target_image must be one of: t1, t2")
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive integers")

    model.eval()
    with torch.no_grad():
        full_prediction = _predict_scalar(model, image_t1, image_t2, tabular)
        _, _, height, width = image_t1.shape
        heatmap = torch.zeros((height, width), dtype=torch.float32, device=image_t1.device)
        counts = torch.zeros((height, width), dtype=torch.float32, device=image_t1.device)

        for y in _window_starts(height, patch_size, stride):
            for x in _window_starts(width, patch_size, stride):
                y_end = min(y + patch_size, height)
                x_end = min(x + patch_size, width)
                masked_t1 = image_t1.clone()
                masked_t2 = image_t2.clone()
                if target_image == "t1":
                    _apply_mask(masked_t1, y, y_end, x, x_end, mask_value)
                else:
                    _apply_mask(masked_t2, y, y_end, x, x_end, mask_value)
                masked_prediction = _predict_scalar(model, masked_t1, masked_t2, tabular)
                importance = abs(full_prediction - masked_prediction)
                heatmap[y:y_end, x:x_end] += float(importance)
                counts[y:y_end, x:x_end] += 1.0

        heatmap = heatmap / counts.clamp_min(1.0)
        if tuple(heatmap.shape) != (128, 128):
            heatmap = F.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=(128, 128),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        return heatmap.detach().cpu(), float(full_prediction)


def _predict_scalar(
    model: torch.nn.Module,
    image_t1: torch.Tensor,
    image_t2: torch.Tensor,
    tabular: torch.Tensor,
) -> float:
    output = model(image_t1, image_t2, tabular)
    if isinstance(output, torch.Tensor):
        return float(output.detach().cpu().view(-1)[0].item())
    if isinstance(output, dict):
        if "change_ratio_pred" in output:
            return float(output["change_ratio_pred"].detach().cpu().view(-1)[0].item())
        if "outputs" in output:
            return _extract_scalar(output["outputs"])
    raise TypeError(f"Unsupported model output type: {type(output).__name__}")


def _extract_scalar(output: Any) -> float:
    if isinstance(output, torch.Tensor):
        return float(output.detach().cpu().view(-1)[0].item())
    if isinstance(output, dict) and "change_ratio_pred" in output:
        return float(output["change_ratio_pred"].detach().cpu().view(-1)[0].item())
    raise TypeError(f"Unsupported nested output type: {type(output).__name__}")


def _apply_mask(
    image: torch.Tensor,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
    mask_value: str,
) -> None:
    if mask_value == "zero":
        image[:, :, y_start:y_end, x_start:x_end] = 0.0
        return
    if mask_value == "channel_mean":
        channel_mean = image.mean(dim=(2, 3), keepdim=True)
        image[:, :, y_start:y_end, x_start:x_end] = channel_mean
        return
    raise ValueError("mask_value must be one of: zero, channel_mean")


def _window_starts(size: int, patch_size: int, stride: int) -> list[int]:
    starts = list(range(0, max(size - patch_size + 1, 1), stride))
    final_start = max(size - patch_size, 0)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts
