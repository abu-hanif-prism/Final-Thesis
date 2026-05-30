"""Modality and temporal ablation utilities for selected XAI samples."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

import torch

from src.xai.handoff_loader import load_npz_sample, prepare_xai_batch
from src.xai.sample_selector import safe_float


ABLATION_COLUMNS = [
    "patch_id",
    "pair_id",
    "district",
    "split",
    "change_class",
    "pair_type",
    "time_gap_group",
    "npz_path",
    "target_change_ratio",
    "full_prediction",
    "no_tabular_prediction",
    "no_image_prediction",
    "no_t1_prediction",
    "no_t2_prediction",
    "same_time_prediction",
    "swapped_time_prediction",
    "image_contribution",
    "tabular_contribution",
    "t1_contribution",
    "t2_contribution",
    "temporal_difference_contribution",
    "temporal_order_sensitivity",
    "dominant_modality",
    "dominant_time_step",
]


def run_modality_temporal_ablation(
    model: torch.nn.Module,
    selected_rows: list[dict[str, Any]],
    device: str | torch.device = "cpu",
    num_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Run modality and temporal ablation for selected samples."""
    torch_device = torch.device(device)
    model.eval()
    rows = selected_rows[: int(num_samples)] if num_samples is not None else selected_rows
    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            print(f"Ablating sample {index}/{len(rows)}: {row.get('patch_id')}", flush=True)
            sample = load_npz_sample(row["npz_path"])
            batch = prepare_xai_batch(sample, device=torch_device)
            predictions = compute_ablation_predictions(model, batch)
            result = build_ablation_result(row, sample, predictions)
            results.append(result)
    return results


def compute_ablation_predictions(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute full and ablated predictions for one batched sample."""
    image_t1 = batch["image_t1"]
    image_t2 = batch["image_t2"]
    tabular = batch["tabular"]
    zero_t1 = torch.zeros_like(image_t1)
    zero_t2 = torch.zeros_like(image_t2)
    zero_tabular = torch.zeros_like(tabular)

    variants = {
        "full_prediction": (image_t1, image_t2, tabular),
        "no_tabular_prediction": (image_t1, image_t2, zero_tabular),
        "no_image_prediction": (zero_t1, zero_t2, tabular),
        "no_t1_prediction": (zero_t1, image_t2, tabular),
        "no_t2_prediction": (image_t1, zero_t2, tabular),
        "same_time_prediction": (image_t1, image_t1, tabular),
        "swapped_time_prediction": (image_t2, image_t1, tabular),
    }
    return {
        name: extract_regression_prediction(model(img1, img2, tab))
        for name, (img1, img2, tab) in variants.items()
    }


def extract_regression_prediction(output: torch.Tensor | dict[str, Any]) -> float:
    """Extract scalar regression prediction from model output."""
    if isinstance(output, torch.Tensor):
        return float(output.detach().cpu().view(-1)[0].item())
    if isinstance(output, dict):
        if "change_ratio_pred" in output:
            return float(output["change_ratio_pred"].detach().cpu().view(-1)[0].item())
        if "outputs" in output:
            return extract_regression_prediction(output["outputs"])
    raise TypeError(f"Unsupported model output type: {type(output).__name__}")


def build_ablation_result(
    selected_row: dict[str, Any],
    sample: dict[str, Any],
    predictions: dict[str, float],
) -> dict[str, Any]:
    """Build one output row with contribution scores and labels."""
    full = predictions["full_prediction"]
    image_contribution = abs(full - predictions["no_image_prediction"])
    tabular_contribution = abs(full - predictions["no_tabular_prediction"])
    t1_contribution = abs(full - predictions["no_t1_prediction"])
    t2_contribution = abs(full - predictions["no_t2_prediction"])
    temporal_difference = abs(full - predictions["same_time_prediction"])
    temporal_order = abs(full - predictions["swapped_time_prediction"])

    row = {
        "patch_id": selected_row.get("patch_id") or sample.get("patch_id"),
        "pair_id": selected_row.get("pair_id") or sample.get("pair_id"),
        "district": selected_row.get("district") or sample.get("district"),
        "split": selected_row.get("split") or sample.get("split"),
        "change_class": selected_row.get("change_class") or sample.get("change_class"),
        "pair_type": selected_row.get("pair_type") or sample.get("pair_type"),
        "time_gap_group": selected_row.get("time_gap_group") or sample.get("time_gap_group"),
        "npz_path": selected_row.get("npz_path"),
        "target_change_ratio": selected_row.get("y_true_change_ratio") or sample.get("change_ratio"),
        **predictions,
        "image_contribution": image_contribution,
        "tabular_contribution": tabular_contribution,
        "t1_contribution": t1_contribution,
        "t2_contribution": t2_contribution,
        "temporal_difference_contribution": temporal_difference,
        "temporal_order_sensitivity": temporal_order,
        "dominant_modality": "image" if image_contribution >= tabular_contribution else "tabular",
        "dominant_time_step": "t1" if t1_contribution >= t2_contribution else "t2",
    }
    return row


def summarize_ablation_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize ablation contribution outputs as CSV-friendly rows."""
    summary: list[dict[str, Any]] = [{"metric": "num_samples", "group": "all", "value": len(rows)}]
    for column in [
        "image_contribution",
        "tabular_contribution",
        "t1_contribution",
        "t2_contribution",
        "temporal_difference_contribution",
        "temporal_order_sensitivity",
    ]:
        values = [value for value in (safe_float(row.get(column)) for row in rows) if value is not None]
        summary.append({"metric": f"average_{column}", "group": "all", "value": mean(values) if values else ""})

    for column in ["dominant_modality", "dominant_time_step", "change_class", "pair_type", "time_gap_group"]:
        counts: dict[str, int] = {}
        for row in rows:
            key = str(row.get(column, ""))
            counts[key] = counts.get(key, 0) + 1
        for key, count in sorted(counts.items()):
            summary.append({"metric": f"{column}_count", "group": key, "value": count})
    return summary


def default_selected_samples_path(output_dir: str | Path, experiment_name: str) -> Path:
    """Return default selected samples path for an experiment."""
    return Path(output_dir) / "selected_samples" / f"xai_selected_samples_{experiment_name}.csv"
