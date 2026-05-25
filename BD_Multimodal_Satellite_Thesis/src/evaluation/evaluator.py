"""Model evaluation and prediction collection utilities."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
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
        progress_every: int = 10,
    ) -> None:
        if output_mode not in {"regression", "classification", "multitask"}:
            raise ValueError("output_mode must be one of: regression, classification, multitask")
        self.model = model
        self.dataloader = dataloader
        self.output_mode = output_mode
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.loss_fn = loss_fn
        self.progress_every = max(int(progress_every), 0)
        self.model.to(self.device)
        self._predictions: list[dict[str, Any]] | None = None

    def evaluate(self) -> dict[str, Any]:
        """Run evaluation and return metrics, optional loss, and predictions."""
        predictions, metric_outputs, metric_batch, average_loss = self._run_inference()
        metrics = compute_metrics(metric_outputs, metric_batch, self.output_mode)
        result: dict[str, Any] = {"metrics": metrics, "predictions": predictions}
        if average_loss is not None:
            result["average_loss"] = average_loss
        self._predictions = predictions
        return result

    def collect_predictions(self) -> list[dict[str, Any]]:
        """Return one prediction row per sample, running inference if needed."""
        if self._predictions is None:
            self._predictions, _, _, _ = self._run_inference()
        return [dict(row) for row in self._predictions]

    def summarize_by_groups(self, predictions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """Create group-level summaries for common metadata dimensions."""
        summaries: dict[str, list[dict[str, Any]]] = {}
        for column in ["split", "district", "change_class", "pair_type", "time_gap_group"]:
            if not any(column in row for row in predictions):
                continue
            groups: dict[Any, list[dict[str, Any]]] = {}
            for row in predictions:
                groups.setdefault(row.get(column), []).append(row)

            rows: list[dict[str, Any]] = []
            for value, group in groups.items():
                row: dict[str, Any] = {column: value, "count": int(len(group))}
                if all(
                    all(key in item for key in ["abs_error", "squared_error", "y_true_change_ratio", "y_pred_change_ratio"])
                    for item in group
                ):
                    abs_errors = [float(item["abs_error"]) for item in group]
                    squared_errors = [float(item["squared_error"]) for item in group]
                    true_values = [float(item["y_true_change_ratio"]) for item in group]
                    pred_values = [float(item["y_pred_change_ratio"]) for item in group]
                    row.update(
                        {
                            "mae": float(np.mean(abs_errors)),
                            "rmse": float(np.sqrt(np.mean(squared_errors))),
                            "mean_true_change_ratio": float(np.mean(true_values)),
                            "mean_predicted_change_ratio": float(np.mean(pred_values)),
                        }
                    )
                if all("correct" in item for item in group):
                    row["accuracy"] = float(np.mean([bool(item["correct"]) for item in group]))
                rows.append(row)
            summaries[column] = sorted(rows, key=lambda item: item.get("count", 0), reverse=True)
        return summaries

    def _run_inference(
        self,
    ) -> tuple[list[dict[str, Any]], torch.Tensor | dict[str, torch.Tensor], dict[str, torch.Tensor], float | None]:
        self.model.eval()
        rows: list[dict[str, Any]] = []
        reg_preds: list[torch.Tensor] = []
        reg_targets: list[torch.Tensor] = []
        cls_logits: list[torch.Tensor] = []
        cls_targets: list[torch.Tensor] = []
        loss_sum = 0.0
        loss_count = 0
        processed_samples = 0
        total_samples = _safe_dataset_len(self.dataloader)
        total_batches = len(self.dataloader) if hasattr(self.dataloader, "__len__") else None
        start_time = time.time()

        with torch.no_grad():
            for batch_index, batch in enumerate(self.dataloader, start=1):
                metadata_batch = batch
                batch = move_batch_to_device(batch, self.device)
                outputs = self.model(batch["image_t1"], batch["image_t2"], batch["tabular"])
                batch_size = int(batch["image_t1"].shape[0])
                processed_samples += batch_size

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

                if self._should_log_progress(batch_index, total_batches):
                    self._print_progress(batch_index, processed_samples, total_samples, start_time)

        metric_outputs, metric_batch = self._metric_payloads(reg_preds, reg_targets, cls_logits, cls_targets)
        average_loss = loss_sum / loss_count if loss_count else None
        return rows, metric_outputs, metric_batch, average_loss

    def _should_log_progress(self, batch_index: int, total_batches: int | None) -> bool:
        """Return whether to print progress after this batch."""
        if self.progress_every <= 0:
            return False
        if batch_index == 1:
            return True
        if batch_index % self.progress_every == 0:
            return True
        return total_batches is not None and batch_index == total_batches

    def _print_progress(
        self,
        batch_index: int,
        processed_samples: int,
        total_samples: int | None,
        start_time: float,
    ) -> None:
        """Print live evaluation progress."""
        elapsed = max(time.time() - start_time, 1e-9)
        samples_per_sec = processed_samples / elapsed
        batches_per_sec = batch_index / elapsed
        if total_samples:
            percent = min(100.0, processed_samples / total_samples * 100.0)
            remaining = max(total_samples - processed_samples, 0)
            eta_seconds = remaining / samples_per_sec if samples_per_sec > 0 else float("inf")
            total_text = str(total_samples)
            percent_text = f"{percent:.1f}%"
            eta_text = _format_seconds(eta_seconds)
        else:
            total_text = "unknown"
            percent_text = "n/a"
            eta_text = "n/a"

        print(
            "Evaluation progress: "
            f"batch={batch_index}, "
            f"samples={processed_samples}/{total_text}, "
            f"percent={percent_text}, "
            f"elapsed={_format_seconds(elapsed)}, "
            f"eta={eta_text}, "
            f"speed={samples_per_sec:.1f} samples/s, "
            f"{batches_per_sec:.2f} batches/s",
            flush=True,
        )

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


def _safe_dataset_len(dataloader: torch.utils.data.DataLoader) -> int | None:
    """Return DataLoader dataset length when available."""
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        return None
    try:
        return len(dataset)
    except TypeError:
        return None


def _format_seconds(seconds: float) -> str:
    """Format seconds as a compact duration."""
    if not np.isfinite(seconds):
        return "n/a"
    seconds_int = int(round(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
