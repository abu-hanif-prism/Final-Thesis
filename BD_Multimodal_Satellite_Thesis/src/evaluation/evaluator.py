"""Model evaluation and prediction collection utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import compute_metrics
from src.training.losses import compute_loss
from src.training.train_utils import move_batch_to_device


CLASS_ID_TO_NAME = {0: "low", 1: "medium", 2: "high"}
METADATA_COLUMNS = ["patch_id", "pair_id", "district", "split", "change_class", "pair_type", "time_gap_group"]


class ModelEvaluator:
    """Evaluate a model on a DataLoader and collect per-sample predictions."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        output_mode: str = "regression",
        device: torch.device | str | None = None,
        loss_fn: torch.nn.Module | None = None,
    ) -> None:
        if output_mode not in {"regression", "classification", "multitask"}:
            raise ValueError("output_mode must be one of: regression, classification, multitask")
        self.model = model
        self.dataloader = dataloader
        self.output_mode = output_mode
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.loss_fn = loss_fn
        self.model.to(self.device)
        self._prediction_df: pd.DataFrame | None = None

    def evaluate(self) -> dict[str, Any]:
        """Run evaluation and return metrics, optional loss, and predictions."""
        prediction_df, metric_outputs, metric_batch, average_loss = self._run_inference()
        metrics = compute_metrics(metric_outputs, metric_batch, self.output_mode)
        result: dict[str, Any] = {"metrics": metrics, "predictions": prediction_df}
        if average_loss is not None:
            result["average_loss"] = average_loss
        self._prediction_df = prediction_df
        return result

    def collect_predictions(self) -> pd.DataFrame:
        """Return one prediction row per sample, running inference if needed."""
        if self._prediction_df is None:
            self._prediction_df, _, _, _ = self._run_inference()
        return self._prediction_df.copy()

    def summarize_by_groups(self, prediction_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Create group-level summaries for common metadata dimensions."""
        summaries: dict[str, pd.DataFrame] = {}
        for column in ["split", "district", "change_class", "pair_type", "time_gap_group"]:
            if column not in prediction_df.columns:
                continue
            rows = []
            for value, group in prediction_df.groupby(column, dropna=False):
                row: dict[str, Any] = {column: value, "count": int(len(group))}
                if {"abs_error", "squared_error", "y_true_change_ratio", "y_pred_change_ratio"}.issubset(group.columns):
                    row.update(
                        {
                            "mae": float(group["abs_error"].mean()),
                            "rmse": float(np.sqrt(group["squared_error"].mean())),
                            "mean_true_change_ratio": float(group["y_true_change_ratio"].mean()),
                            "mean_predicted_change_ratio": float(group["y_pred_change_ratio"].mean()),
                        }
                    )
                if "correct" in group.columns:
                    row["accuracy"] = float(group["correct"].mean())
                rows.append(row)
            summaries[column] = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
        return summaries

    def _run_inference(
        self,
    ) -> tuple[pd.DataFrame, torch.Tensor | dict[str, torch.Tensor], dict[str, torch.Tensor], float | None]:
        self.model.eval()
        rows: list[dict[str, Any]] = []
        reg_preds: list[torch.Tensor] = []
        reg_targets: list[torch.Tensor] = []
        cls_logits: list[torch.Tensor] = []
        cls_targets: list[torch.Tensor] = []
        loss_sum = 0.0
        loss_count = 0

        with torch.no_grad():
            for batch in self.dataloader:
                metadata_batch = batch
                batch = move_batch_to_device(batch, self.device)
                outputs = self.model(batch["image_t1"], batch["image_t2"], batch["tabular"])
                batch_size = int(batch["image_t1"].shape[0])

                if self.loss_fn is not None:
                    loss_dict = compute_loss(outputs, batch, self.output_mode, self.loss_fn)
                    loss_sum += float(loss_dict["total_loss"].detach().cpu().item()) * batch_size
                    loss_count += batch_size

                batch_rows = self._batch_prediction_rows(outputs, batch, metadata_batch)
                rows.extend(batch_rows)

                if self.output_mode in {"regression", "multitask"}:
                    pred = outputs["change_ratio_pred"] if isinstance(outputs, dict) else outputs
                    reg_preds.append(pred.detach().cpu().view(-1))
                    reg_targets.append(batch["change_ratio"].detach().cpu().view(-1))
                if self.output_mode in {"classification", "multitask"}:
                    logits = outputs["change_class_logits"] if isinstance(outputs, dict) else outputs
                    cls_logits.append(logits.detach().cpu())
                    cls_targets.append(batch["change_class_id"].detach().cpu().view(-1))

        prediction_df = pd.DataFrame(rows)
        metric_outputs, metric_batch = self._metric_payloads(reg_preds, reg_targets, cls_logits, cls_targets)
        average_loss = loss_sum / loss_count if loss_count else None
        return prediction_df, metric_outputs, metric_batch, average_loss

    def _metric_payloads(
        self,
        reg_preds: list[torch.Tensor],
        reg_targets: list[torch.Tensor],
        cls_logits: list[torch.Tensor],
        cls_targets: list[torch.Tensor],
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.output_mode == "regression":
            return torch.cat(reg_preds), {"change_ratio": torch.cat(reg_targets)}
        if self.output_mode == "classification":
            return torch.cat(cls_logits), {"change_class_id": torch.cat(cls_targets)}
        return (
            {"change_ratio_pred": torch.cat(reg_preds), "change_class_logits": torch.cat(cls_logits)},
            {"change_ratio": torch.cat(reg_targets), "change_class_id": torch.cat(cls_targets)},
        )

    def _batch_prediction_rows(
        self,
        outputs: torch.Tensor | dict[str, torch.Tensor],
        batch: dict[str, Any],
        metadata_batch: dict[str, Any],
    ) -> list[dict[str, Any]]:
        batch_size = int(batch["image_t1"].shape[0])
        base_rows = [{column: _metadata_value(metadata_batch, column, index) for column in METADATA_COLUMNS} for index in range(batch_size)]

        if self.output_mode in {"regression", "multitask"}:
            pred = outputs["change_ratio_pred"] if isinstance(outputs, dict) else outputs
            pred_np = pred.detach().cpu().view(-1).numpy()
            true_np = batch["change_ratio"].detach().cpu().view(-1).numpy()
            for index, row in enumerate(base_rows):
                error = float(pred_np[index] - true_np[index])
                row.update(
                    {
                        "y_true_change_ratio": float(true_np[index]),
                        "y_pred_change_ratio": float(pred_np[index]),
                        "abs_error": abs(error),
                        "squared_error": error * error,
                    }
                )

        if self.output_mode in {"classification", "multitask"}:
            logits = outputs["change_class_logits"] if isinstance(outputs, dict) else outputs
            probs = torch.softmax(logits.detach().cpu(), dim=1).numpy()
            pred_ids = probs.argmax(axis=1)
            true_ids = batch["change_class_id"].detach().cpu().view(-1).numpy()
            for index, row in enumerate(base_rows):
                pred_id = int(pred_ids[index])
                true_id = int(true_ids[index])
                row.update(
                    {
                        "y_true_class_id": true_id,
                        "y_pred_class_id": pred_id,
                        "y_pred_class_name": CLASS_ID_TO_NAME.get(pred_id, str(pred_id)),
                        "prob_low": float(probs[index, 0]),
                        "prob_medium": float(probs[index, 1]),
                        "prob_high": float(probs[index, 2]),
                        "correct": bool(pred_id == true_id),
                    }
                )
        return base_rows


def _metadata_value(batch: dict[str, Any], column: str, index: int) -> Any:
    if column not in batch:
        return None
    value = batch[column]
    if isinstance(value, torch.Tensor):
        return value[index].detach().cpu().item()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value
