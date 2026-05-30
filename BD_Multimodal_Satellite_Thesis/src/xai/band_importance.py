"""Sentinel band ablation importance for selected XAI samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.xai.handoff_loader import load_npz_sample, prepare_xai_batch
from src.xai.sample_selector import safe_float
from src.xai.xai_config import SENTINEL_BAND_NAMES


BAND_COLUMNS = [
    "model_name",
    "experiment_name",
    "patch_id",
    "pair_id",
    "district",
    "split",
    "change_class",
    "pair_type",
    "time_gap_group",
    "true_change_ratio",
    "full_prediction",
    "band_index",
    "band_name",
    "ablated_prediction",
    "importance",
]

SUMMARY_COLUMNS = [
    "band_index",
    "band_name",
    "sample_count",
    "mean_importance",
    "median_importance",
    "max_importance",
]


def compute_band_importance(
    model: torch.nn.Module,
    selected_rows: list[dict[str, Any]],
    model_name: str,
    experiment_name: str,
    device: torch.device | str,
    num_samples: int | None = None,
    band_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Ablate each image channel in both time steps and record prediction change."""
    rows = selected_rows[: int(num_samples)] if num_samples is not None else selected_rows
    names = _band_names(band_names)
    output_rows: list[dict[str, Any]] = []
    model.eval()

    for index, row in enumerate(rows, start=1):
        print(f"Band importance sample {index}/{len(rows)}: {row.get('patch_id')}", flush=True)
        sample = load_npz_sample(row["npz_path"])
        batch = prepare_xai_batch(sample, device=device)
        full_prediction = predict_scalar(model, batch["image_t1"], batch["image_t2"], batch["tabular"])
        channel_count = int(batch["image_t1"].shape[1])
        true_change_ratio = safe_float(row.get("y_true_change_ratio"), safe_float(sample.get("change_ratio"), None))

        for channel in range(channel_count):
            masked_t1 = batch["image_t1"].clone()
            masked_t2 = batch["image_t2"].clone()
            masked_t1[:, channel, :, :] = 0.0
            masked_t2[:, channel, :, :] = 0.0
            ablated_prediction = predict_scalar(model, masked_t1, masked_t2, batch["tabular"])
            output_rows.append(
                {
                    "model_name": model_name,
                    "experiment_name": experiment_name,
                    "patch_id": row.get("patch_id", sample.get("patch_id", "")),
                    "pair_id": row.get("pair_id", sample.get("pair_id", "")),
                    "district": row.get("district", sample.get("district", "")),
                    "split": row.get("split", sample.get("split", "")),
                    "change_class": row.get("change_class", sample.get("change_class", "")),
                    "pair_type": row.get("pair_type", sample.get("pair_type", "")),
                    "time_gap_group": row.get("time_gap_group", sample.get("time_gap_group", "")),
                    "true_change_ratio": true_change_ratio if true_change_ratio is not None else "",
                    "full_prediction": full_prediction,
                    "band_index": channel + 1,
                    "band_name": names[channel] if channel < len(names) else f"band_{channel + 1:02d}",
                    "ablated_prediction": ablated_prediction,
                    "importance": abs(full_prediction - ablated_prediction),
                }
            )
    return output_rows


def summarize_band_importance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize per-band importance across selected samples."""
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        importance = safe_float(row.get("importance"))
        if importance is None:
            continue
        key = (int(row.get("band_index", 0)), str(row.get("band_name", "")))
        grouped.setdefault(key, []).append(importance)

    summary_rows: list[dict[str, Any]] = []
    for (band_index, band_name), values in sorted(grouped.items()):
        sorted_values = sorted(values)
        summary_rows.append(
            {
                "band_index": band_index,
                "band_name": band_name,
                "sample_count": len(values),
                "mean_importance": sum(values) / len(values),
                "median_importance": _median(sorted_values),
                "max_importance": max(values),
            }
        )
    return sorted(summary_rows, key=lambda row: safe_float(row.get("mean_importance"), 0.0) or 0.0, reverse=True)


def save_band_bar_plot(summary_rows: list[dict[str, Any]], path: str | Path, title: str) -> Path:
    """Save a simple horizontal bar plot with Pillow."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Band importance PNG output requires Pillow.") from exc

    rows = summary_rows[:]
    width = 900
    row_height = 34
    margin_left = 190
    margin_right = 40
    top = 60
    height = top + row_height * max(len(rows), 1) + 30
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 18), title[:120], fill=(0, 0, 0))
    max_value = max([safe_float(row.get("mean_importance"), 0.0) or 0.0 for row in rows] + [1e-12])
    bar_width = width - margin_left - margin_right

    for idx, row in enumerate(rows):
        y = top + idx * row_height
        label = f"{row.get('band_index')}. {row.get('band_name')}"
        value = safe_float(row.get("mean_importance"), 0.0) or 0.0
        length = int((value / max_value) * bar_width)
        draw.text((20, y + 7), label[:26], fill=(0, 0, 0))
        draw.rectangle((margin_left, y + 5, margin_left + length, y + 24), fill=(51, 113, 181))
        draw.text((margin_left + length + 8, y + 7), f"{value:.6f}", fill=(0, 0, 0))

    image.save(output_path)
    return output_path


def build_band_report(
    model_name: str,
    experiment_name: str,
    sample_count: int,
    per_sample_path: Path,
    summary_path: Path,
    plot_path: Path,
    summary_rows: list[dict[str, Any]],
) -> str:
    """Build Markdown report for band importance."""
    top_rows = summary_rows[:5]
    lines = [
        f"# Band Importance Report: {experiment_name}",
        "",
        f"- model_name: {model_name}",
        f"- experiment_name: {experiment_name}",
        f"- samples explained: {sample_count}",
        f"- per-sample CSV: {per_sample_path}",
        f"- summary CSV: {summary_path}",
        f"- bar plot: {plot_path}",
        "",
        "## Method",
        "",
        "Each of the 13 image channels is set to zero in both image_t1 and image_t2. Importance is the absolute change between the full prediction and the ablated prediction.",
        "",
        "## Top Bands",
        "",
    ]
    for row in top_rows:
        lines.append(
            f"- {row.get('band_name')} (band {row.get('band_index')}): "
            f"mean importance {safe_float(row.get('mean_importance'), 0.0):.6f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Higher values mean the model prediction changed more when that band was removed, suggesting stronger reliance on that channel for the selected XAI samples.",
        ]
    )
    return "\n".join(lines) + "\n"


def predict_scalar(
    model: torch.nn.Module,
    image_t1: torch.Tensor,
    image_t2: torch.Tensor,
    tabular: torch.Tensor,
) -> float:
    """Run model and extract a scalar change-ratio prediction."""
    with torch.no_grad():
        output = model(image_t1, image_t2, tabular)
    if isinstance(output, torch.Tensor):
        return float(output.detach().cpu().view(-1)[0].item())
    if isinstance(output, dict) and "change_ratio_pred" in output:
        return float(output["change_ratio_pred"].detach().cpu().view(-1)[0].item())
    raise TypeError(f"Unsupported model output type: {type(output).__name__}")


def _band_names(band_names: list[str] | None) -> list[str]:
    names = band_names or SENTINEL_BAND_NAMES
    if len(names) >= 13:
        return names[:13]
    return [f"band_{index + 1:02d}" for index in range(13)]


def _median(values: list[float]) -> float:
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2.0
