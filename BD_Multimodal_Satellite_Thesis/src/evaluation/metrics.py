"""Reusable metrics for regression, classification, and multitask outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


CLASS_NAMES = ("low", "medium", "high")


def regression_metrics(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> dict[str, float]:
    """Compute regression metrics for change-ratio prediction."""
    pred_np = _to_numpy(pred).reshape(-1).astype(float)
    target_np = _to_numpy(target).reshape(-1).astype(float)
    if pred_np.shape != target_np.shape:
        raise ValueError(f"Regression pred/target shape mismatch: {pred_np.shape} vs {target_np.shape}")

    error = pred_np - target_np
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    target_var_sum = float(np.sum((target_np - target_np.mean()) ** 2))
    r2 = float(1.0 - np.sum(error**2) / target_var_sum) if target_var_sum > 0 else float("nan")
    if pred_np.size > 1 and np.std(pred_np) > 0 and np.std(target_np) > 0:
        pearson_corr = float(np.corrcoef(pred_np, target_np)[0, 1])
    else:
        pearson_corr = float("nan")
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "pearson_corr": pearson_corr,
        "mean_pred": float(np.mean(pred_np)),
        "mean_target": float(np.mean(target_np)),
    }


def classification_metrics(
    logits_or_pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics without sklearn."""
    pred_np = _classification_predictions(logits_or_pred)
    target_np = _to_numpy(target).reshape(-1).astype(int)
    if pred_np.shape != target_np.shape:
        raise ValueError(f"Classification pred/target shape mismatch: {pred_np.shape} vs {target_np.shape}")

    cm = confusion_matrix_np(pred_np, target_np, num_classes=3)
    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total) if total else 0.0
    precisions = []
    recalls = []
    f1_scores = []
    supports = []
    class_accuracies = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        tp = float(cm[class_id, class_id])
        fp = float(cm[:, class_id].sum() - tp)
        fn = float(cm[class_id, :].sum() - tp)
        support = float(cm[class_id, :].sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)
        class_accuracies[f"class_accuracy_{class_name}"] = _safe_div(tp, support)

    support_sum = float(sum(supports))
    weighted_f1 = (
        float(sum(f1 * support for f1, support in zip(f1_scores, supports)) / support_sum)
        if support_sum > 0
        else 0.0
    )
    metrics = {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "weighted_f1": weighted_f1,
    }
    metrics.update(class_accuracies)
    return metrics


def multitask_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute prefixed metrics for multitask model outputs."""
    if "change_ratio_pred" not in outputs or "change_class_logits" not in outputs:
        raise KeyError("Multitask outputs must include change_ratio_pred and change_class_logits")
    reg = regression_metrics(outputs["change_ratio_pred"], batch["change_ratio"])
    cls = classification_metrics(outputs["change_class_logits"], batch["change_class_id"])
    combined = {f"reg_{key}": value for key, value in reg.items()}
    combined.update({f"cls_{key}": value for key, value in cls.items()})
    return combined


def compute_metrics(
    outputs: torch.Tensor | dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    output_mode: str,
) -> dict[str, float]:
    """Compute metrics for the selected model output mode."""
    if output_mode == "regression":
        pred = outputs["change_ratio_pred"] if isinstance(outputs, dict) else outputs
        return regression_metrics(pred, batch["change_ratio"])
    if output_mode == "classification":
        logits = outputs["change_class_logits"] if isinstance(outputs, dict) else outputs
        return classification_metrics(logits, batch["change_class_id"])
    if output_mode == "multitask":
        if not isinstance(outputs, dict):
            raise TypeError("Multitask outputs must be a dictionary")
        return multitask_metrics(outputs, batch)
    raise ValueError("output_mode must be one of: regression, classification, multitask")


def confusion_matrix_np(
    pred_classes: torch.Tensor | np.ndarray,
    target_classes: torch.Tensor | np.ndarray,
    num_classes: int = 3,
) -> np.ndarray:
    """Return confusion matrix with rows=true class and columns=predicted class."""
    pred_np = _to_numpy(pred_classes).reshape(-1).astype(int)
    target_np = _to_numpy(target_classes).reshape(-1).astype(int)
    if pred_np.shape != target_np.shape:
        raise ValueError(f"Confusion matrix shape mismatch: {pred_np.shape} vs {target_np.shape}")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (
        (target_np >= 0)
        & (target_np < num_classes)
        & (pred_np >= 0)
        & (pred_np < num_classes)
    )
    for true_class, pred_class in zip(target_np[valid], pred_np[valid]):
        matrix[true_class, pred_class] += 1
    return matrix


def summarize_metrics(metrics_dict: dict[str, float], prefix: str = "") -> str:
    """Return a compact one-line metric summary for console logging."""
    parts = []
    for key in sorted(metrics_dict):
        value = metrics_dict[key]
        metric_name = f"{prefix}{key}" if prefix else key
        if isinstance(value, float):
            parts.append(f"{metric_name}={value:.4f}" if np.isfinite(value) else f"{metric_name}=nan")
        else:
            parts.append(f"{metric_name}={value}")
    return " | ".join(parts)


def _classification_predictions(logits_or_pred: torch.Tensor | np.ndarray) -> np.ndarray:
    values = _to_numpy(logits_or_pred)
    if values.ndim == 2:
        return values.argmax(axis=1).astype(int)
    return values.reshape(-1).astype(int)


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0
