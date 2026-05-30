"""Visualization helpers for XAI heatmaps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def normalize_image_for_display(image_tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    """Robustly normalize channel-first image data to [0, 1]."""
    image = _to_numpy_chw(image_tensor).astype(np.float32)
    normalized = np.zeros_like(image, dtype=np.float32)
    for channel in range(image.shape[0]):
        values = image[channel]
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            normalized[channel] = 0.0
        else:
            normalized[channel] = np.clip((values - low) / (high - low), 0.0, 1.0)
    return normalized


def make_rgb_preview(image_tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    """Create RGB preview from first three channels."""
    normalized = normalize_image_for_display(image_tensor)
    if normalized.shape[0] < 3:
        raise ValueError("Need at least three channels for RGB preview")
    return np.moveaxis(normalized[:3], 0, -1)


def save_heatmap_png(heatmap: torch.Tensor | np.ndarray, path: str | Path) -> Path:
    """Save a heatmap PNG."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_np = _normalize_heatmap(heatmap)
    _save_rgb_array(_heatmap_to_rgb(heatmap_np), output_path)
    return output_path


def save_overlay_png(rgb_image: np.ndarray, heatmap: torch.Tensor | np.ndarray, path: str | Path) -> Path:
    """Save RGB image with heatmap overlay."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_np = _normalize_heatmap(heatmap)
    rgb_uint8 = _float_rgb_to_uint8(rgb_image)
    heat_uint8 = _float_rgb_to_uint8(_heatmap_to_rgb(heatmap_np))
    overlay = (0.55 * rgb_uint8.astype(np.float32) + 0.45 * heat_uint8.astype(np.float32)).astype(np.uint8)
    _save_uint8_rgb(overlay, output_path)
    return output_path


def save_side_by_side_xai(
    t1_rgb: np.ndarray,
    t2_rgb: np.ndarray,
    t1_heatmap: torch.Tensor | np.ndarray,
    t2_heatmap: torch.Tensor | np.ndarray,
    path: str | Path,
    title: str = "",
) -> Path:
    """Save side-by-side RGB previews and occlusion overlays."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t1_heat = _normalize_heatmap(t1_heatmap)
    t2_heat = _normalize_heatmap(t2_heatmap)
    t1_rgb_uint8 = _float_rgb_to_uint8(t1_rgb)
    t2_rgb_uint8 = _float_rgb_to_uint8(t2_rgb)
    t1_overlay = (
        0.55 * t1_rgb_uint8.astype(np.float32)
        + 0.45 * _float_rgb_to_uint8(_heatmap_to_rgb(t1_heat)).astype(np.float32)
    ).astype(np.uint8)
    t2_overlay = (
        0.55 * t2_rgb_uint8.astype(np.float32)
        + 0.45 * _float_rgb_to_uint8(_heatmap_to_rgb(t2_heat)).astype(np.float32)
    ).astype(np.uint8)
    _save_grid(
        [
            ("T1 RGB", t1_rgb_uint8),
            ("T2 RGB", t2_rgb_uint8),
            ("T1 occlusion", t1_overlay),
            ("T2 occlusion", t2_overlay),
        ],
        output_path,
        title=title,
    )
    return output_path


def save_rgb_png(rgb_image: np.ndarray, path: str | Path) -> Path:
    """Save RGB preview PNG."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_rgb_array(rgb_image, output_path)
    return output_path


def _to_numpy_chw(image_tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image_tensor, torch.Tensor):
        image = image_tensor.detach().cpu().numpy()
    else:
        image = np.asarray(image_tensor)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected image shape [C,H,W] or [B,C,H,W], got {image.shape}")
    return image


def _normalize_heatmap(heatmap: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(heatmap, torch.Tensor):
        heat = heatmap.detach().cpu().numpy()
    else:
        heat = np.asarray(heatmap)
    heat = heat.astype(np.float32)
    if heat.ndim == 3:
        heat = heat.squeeze()
    min_value = float(np.nanmin(heat))
    max_value = float(np.nanmax(heat))
    if max_value <= min_value:
        return np.zeros_like(heat, dtype=np.float32)
    return (heat - min_value) / (max_value - min_value)


def _heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    """Approximate an inferno-like heatmap without matplotlib."""
    heat = np.clip(heatmap, 0.0, 1.0).astype(np.float32)
    stops = np.array(
        [
            [0.001, 0.000, 0.014],
            [0.230, 0.060, 0.438],
            [0.550, 0.160, 0.506],
            [0.870, 0.320, 0.230],
            [0.988, 0.998, 0.645],
        ],
        dtype=np.float32,
    )
    scaled = heat * (len(stops) - 1)
    low = np.floor(scaled).astype(np.int32)
    high = np.clip(low + 1, 0, len(stops) - 1)
    weight = (scaled - low)[..., None]
    return stops[low] * (1.0 - weight) + stops[high] * weight


def _save_rgb_array(rgb_image: np.ndarray, path: Path) -> None:
    _save_uint8_rgb(_float_rgb_to_uint8(rgb_image), path)


def _float_rgb_to_uint8(rgb_image: np.ndarray) -> np.ndarray:
    return (np.clip(rgb_image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _save_uint8_rgb(rgb_image: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "PNG export requires either matplotlib or Pillow. Install one with "
            "`pip install matplotlib` or `pip install pillow`."
        ) from exc
    Image.fromarray(rgb_image, mode="RGB").save(path)


def _save_grid(panels: list[tuple[str, np.ndarray]], path: Path, title: str = "") -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError(
            "PNG export requires either matplotlib or Pillow. Install one with "
            "`pip install matplotlib` or `pip install pillow`."
        ) from exc

    label_height = 24
    title_height = 30 if title else 0
    height, width, _ = panels[0][1].shape
    canvas = Image.new("RGB", (width * 2, title_height + (height + label_height) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((8, 8), title[:160], fill=(0, 0, 0))
    for index, (label, image_array) in enumerate(panels):
        row = index // 2
        col = index % 2
        x = col * width
        y = title_height + row * (height + label_height)
        draw.text((x + 8, y + 4), label, fill=(0, 0, 0))
        canvas.paste(Image.fromarray(image_array, mode="RGB"), (x, y + label_height))
    canvas.save(path)
