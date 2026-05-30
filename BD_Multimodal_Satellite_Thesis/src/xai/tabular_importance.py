"""Tabular feature ablation importance for selected XAI samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.xai.handoff_loader import load_npz_sample, prepare_xai_batch
from src.xai.sample_selector import safe_float


TABULAR_COLUMNS = [
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
    "feature_index",
    "feature_name",
    "feature_group",
    "ablated_prediction",
    "importance",
]

TABULAR_SUMMARY_COLUMNS = [
    "feature_index",
    "feature_name",
    "feature_group",
    "sample_count",
    "mean_importance",
    "median_importance",
    "max_importance",
]

GROUP_SUMMARY_COLUMNS = [
    "feature_group",
    "feature_count",
    "mean_importance",
    "max_importance",
]


def load_feature_names(
    feature_columns_path: str | Path = "data/tabular/processed/pair_tabular_feature_columns.json",
    feature_count: int = 146,
) -> list[str]:
    """Load tabular feature names or return feature_000 style defaults."""
    path = Path(feature_columns_path)
    if not path.exists():
        return [f"feature_{index:03d}" for index in range(feature_count)]
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    for key in ["tabular_feature_columns", "feature_columns", "processed_feature_columns", "raw_feature_columns"]:
        values = payload.get(key)
        if isinstance(values, list) and values:
            names = [str(value) for value in values]
            if len(names) >= feature_count:
                return names[:feature_count]
            return names + [f"feature_{index:03d}" for index in range(len(names), feature_count)]
    return [f"feature_{index:03d}" for index in range(feature_count)]


def compute_tabular_importance(
    model: torch.nn.Module,
    selected_rows: list[dict[str, Any]],
    model_name: str,
    experiment_name: str,
    device: torch.device | str,
    feature_names: list[str],
    num_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Set each tabular feature to zero and record prediction change."""
    rows = selected_rows[: int(num_samples)] if num_samples is not None else selected_rows
    output_rows: list[dict[str, Any]] = []
    model.eval()

    for index, row in enumerate(rows, start=1):
        print(f"Tabular importance sample {index}/{len(rows)}: {row.get('patch_id')}", flush=True)
        sample = load_npz_sample(row["npz_path"])
        batch = prepare_xai_batch(sample, device=device)
        full_prediction = predict_scalar(model, batch["image_t1"], batch["image_t2"], batch["tabular"])
        feature_count = int(batch["tabular"].shape[1])
        true_change_ratio = safe_float(row.get("y_true_change_ratio"), safe_float(sample.get("change_ratio"), None))

        for feature_index in range(feature_count):
            masked_tabular = batch["tabular"].clone()
            masked_tabular[:, feature_index] = 0.0
            ablated_prediction = predict_scalar(model, batch["image_t1"], batch["image_t2"], masked_tabular)
            feature_name = feature_names[feature_index] if feature_index < len(feature_names) else f"feature_{feature_index:03d}"
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
                    "feature_index": feature_index,
                    "feature_name": feature_name,
                    "feature_group": infer_feature_group(feature_name),
                    "ablated_prediction": ablated_prediction,
                    "importance": abs(full_prediction - ablated_prediction),
                }
            )
    return output_rows


def summarize_tabular_importance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize feature importance across samples."""
    grouped: dict[tuple[int, str, str], list[float]] = {}
    for row in rows:
        importance = safe_float(row.get("importance"))
        if importance is None:
            continue
        key = (
            int(row.get("feature_index", 0)),
            str(row.get("feature_name", "")),
            str(row.get("feature_group", "other")),
        )
        grouped.setdefault(key, []).append(importance)
    summary: list[dict[str, Any]] = []
    for (feature_index, feature_name, feature_group), values in grouped.items():
        values_sorted = sorted(values)
        summary.append(
            {
                "feature_index": feature_index,
                "feature_name": feature_name,
                "feature_group": feature_group,
                "sample_count": len(values),
                "mean_importance": sum(values) / len(values),
                "median_importance": _median(values_sorted),
                "max_importance": max(values),
            }
        )
    return sorted(summary, key=lambda row: safe_float(row.get("mean_importance"), 0.0) or 0.0, reverse=True)


def summarize_tabular_groups(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate top-level feature-group importance."""
    grouped: dict[str, list[float]] = {}
    for row in summary_rows:
        value = safe_float(row.get("mean_importance"))
        if value is None:
            continue
        grouped.setdefault(str(row.get("feature_group", "other")), []).append(value)
    output: list[dict[str, Any]] = []
    for group, values in grouped.items():
        output.append(
            {
                "feature_group": group,
                "feature_count": len(values),
                "mean_importance": sum(values) / len(values),
                "max_importance": max(values),
            }
        )
    return sorted(output, key=lambda row: safe_float(row.get("mean_importance"), 0.0) or 0.0, reverse=True)


def infer_feature_group(feature_name: str) -> str:
    """Infer a broad environmental group from a feature name."""
    name = feature_name.lower()
    groups = [
        ("temperature", ["temp", "dewpoint", "heat"]),
        ("humidity_rainfall", ["humidity", "rain", "precip"]),
        ("water_soil", ["water", "soil", "runoff", "flood"]),
        ("agriculture", ["crop", "rice", "irrigation", "cropping"]),
        ("terrain", ["elevation", "slope"]),
        ("coastal_distance", ["coast", "distance"]),
        ("population_settlement", ["population", "nightlight", "built", "urban"]),
        ("time", ["time_gap", "year", "season"]),
        ("ratio_change", ["_diff", "_ratio"]),
    ]
    for group, keywords in groups:
        if any(keyword in name for keyword in keywords):
            return group
    return "other"


def save_top_feature_plot(summary_rows: list[dict[str, Any]], path: str | Path, title: str, top_k: int = 25) -> Path:
    """Save a top-feature bar plot with Pillow."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Tabular importance PNG output requires Pillow.") from exc

    rows = summary_rows[:top_k]
    width = 1100
    row_height = 30
    margin_left = 360
    margin_right = 45
    top = 60
    height = top + row_height * max(len(rows), 1) + 30
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 18), title[:140], fill=(0, 0, 0))
    max_value = max([safe_float(row.get("mean_importance"), 0.0) or 0.0 for row in rows] + [1e-12])
    bar_width = width - margin_left - margin_right
    for idx, row in enumerate(rows):
        y = top + idx * row_height
        label = str(row.get("feature_name", ""))[:46]
        value = safe_float(row.get("mean_importance"), 0.0) or 0.0
        length = int((value / max_value) * bar_width)
        draw.text((20, y + 6), label, fill=(0, 0, 0))
        draw.rectangle((margin_left, y + 5, margin_left + length, y + 22), fill=(56, 142, 60))
        draw.text((margin_left + length + 8, y + 5), f"{value:.6f}", fill=(0, 0, 0))
    image.save(output_path)
    return output_path


def build_tabular_report(
    model_name: str,
    experiment_name: str,
    sample_count: int,
    per_sample_path: Path,
    summary_path: Path,
    group_path: Path,
    plot_path: Path,
    summary_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
) -> str:
    """Build Markdown report for tabular importance."""
    lines = [
        f"# Tabular Feature Importance Report: {experiment_name}",
        "",
        f"- model_name: {model_name}",
        f"- experiment_name: {experiment_name}",
        f"- samples explained: {sample_count}",
        f"- per-sample CSV: {per_sample_path}",
        f"- feature summary CSV: {summary_path}",
        f"- group summary CSV: {group_path}",
        f"- top feature plot: {plot_path}",
        "",
        "## Method",
        "",
        "Each tabular feature is set to zero while image inputs are unchanged. Importance is the absolute change between the full prediction and the ablated prediction.",
        "",
        "## Top Features",
        "",
    ]
    for row in summary_rows[:10]:
        lines.append(
            f"- {row.get('feature_name')} ({row.get('feature_group')}): "
            f"mean importance {safe_float(row.get('mean_importance'), 0.0):.6f}"
        )
    lines.extend(["", "## Top Feature Groups", ""])
    for row in group_rows[:10]:
        lines.append(
            f"- {row.get('feature_group')}: mean importance "
            f"{safe_float(row.get('mean_importance'), 0.0):.6f}"
        )
    return "\n".join(lines) + "\n"


def predict_scalar(
    model: torch.nn.Module,
    image_t1: torch.Tensor,
    image_t2: torch.Tensor,
    tabular: torch.Tensor,
) -> float:
    """Run model and extract a scalar prediction."""
    with torch.no_grad():
        output = model(image_t1, image_t2, tabular)
    if isinstance(output, torch.Tensor):
        return float(output.detach().cpu().view(-1)[0].item())
    if isinstance(output, dict) and "change_ratio_pred" in output:
        return float(output["change_ratio_pred"].detach().cpu().view(-1)[0].item())
    raise TypeError(f"Unsupported model output type: {type(output).__name__}")


def _median(values: list[float]) -> float:
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2.0
