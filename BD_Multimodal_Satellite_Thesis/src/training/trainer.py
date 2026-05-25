"""Generic trainer for multimodal Siamese models."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

from src.evaluation.metrics import compute_metrics
from src.training.losses import compute_loss
from src.training.train_utils import ensure_dir, move_batch_to_device, save_checkpoint


class Trainer:
    """Reusable train/validation loop for CNN and future Siamese backbones."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        output_mode: str = "regression",
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        device: torch.device | str | None = None,
        mixed_precision: bool = False,
        grad_clip_norm: float | None = None,
        checkpoint_dir: str | Path = "checkpoints",
        log_dir: str | Path = "logs",
        experiment_name: str = "siamese_cnn_baseline",
        save_best: bool = True,
        metric_for_best: str = "val_loss",
        lower_is_better: bool = True,
        start_epoch: int = 1,
        best_metric: float | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.output_mode = output_mode
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.mixed_precision = bool(mixed_precision and self.device.type == "cuda")
        self.grad_clip_norm = grad_clip_norm
        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.log_dir = ensure_dir(log_dir)
        self.experiment_name = experiment_name
        self.save_best = save_best
        self.metric_for_best = metric_for_best
        self.lower_is_better = lower_is_better
        self.start_epoch = max(1, int(start_epoch))
        self.best_metric = best_metric
        self.history = list(history or [])
        self.model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch and return averaged metrics."""
        if self.optimizer is None:
            raise ValueError("Trainer requires an optimizer for training.")
        self.model.train()
        total_samples = 0
        loss_sums: dict[str, float] = {}
        metric_sums: dict[str, float] = {}

        for batch in self.train_loader:
            batch = move_batch_to_device(batch, self.device)
            batch_size = int(batch["image_t1"].shape[0])
            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(batch["image_t1"], batch["image_t2"], batch["tabular"])
                loss_dict = compute_loss(outputs, batch, self.output_mode, self.loss_fn)
                loss = loss_dict["total_loss"]

            self.scaler.scale(loss).backward()
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            metrics = compute_metrics(outputs, batch, self.output_mode)
            _accumulate(loss_sums, _tensor_dict_to_float(loss_dict), batch_size)
            _accumulate(metric_sums, metrics, batch_size)
            total_samples += batch_size

        return _with_loss_alias(_prefix_metrics(_average_metrics(loss_sums | metric_sums, total_samples), "train"))

    def validate(self, epoch: int) -> dict[str, float]:
        """Run validation loop and return averaged metrics."""
        self.model.eval()
        total_samples = 0
        loss_sums: dict[str, float] = {}
        metric_sums: dict[str, float] = {}

        with torch.no_grad():
            for batch in self.val_loader:
                batch = move_batch_to_device(batch, self.device)
                batch_size = int(batch["image_t1"].shape[0])
                outputs = self.model(batch["image_t1"], batch["image_t2"], batch["tabular"])
                loss_dict = compute_loss(outputs, batch, self.output_mode, self.loss_fn)
                metrics = compute_metrics(outputs, batch, self.output_mode)
                _accumulate(loss_sums, _tensor_dict_to_float(loss_dict), batch_size)
                _accumulate(metric_sums, metrics, batch_size)
                total_samples += batch_size

        return _with_loss_alias(_prefix_metrics(_average_metrics(loss_sums | metric_sums, total_samples), "val"))

    def fit(self, num_epochs: int) -> list[dict[str, Any]]:
        """Train from start_epoch through num_epochs inclusive."""
        for epoch in range(self.start_epoch, int(num_epochs) + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)
            epoch_metrics: dict[str, Any] = {"epoch": epoch, **train_metrics, **val_metrics}
            self.history.append(epoch_metrics)
            self._step_scheduler(epoch_metrics)
            self._save_epoch_checkpoints(epoch, epoch_metrics)
            self.save_history()
            print(_format_epoch_summary(epoch_metrics))
        return self.history

    def save_history(self) -> Path:
        """Save training history CSV and return its path."""
        path = self.log_dir / f"{self.experiment_name}_history.csv"
        if not self.history:
            return path
        fieldnames = sorted({key for row in self.history for key in row})
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)
        return path

    def _is_best(self, current_metric: float | None) -> bool:
        """Return True if current metric improves over best metric."""
        if current_metric is None:
            return False
        if self.best_metric is None:
            return True
        if self.lower_is_better:
            return current_metric < self.best_metric
        return current_metric > self.best_metric

    def _step_scheduler(self, metrics: dict[str, Any]) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metrics.get(self.metric_for_best, metrics.get("val_total_loss", 0.0)))
        else:
            self.scheduler.step()

    def _save_epoch_checkpoints(self, epoch: int, metrics: dict[str, Any]) -> None:
        latest_path = self.checkpoint_dir / f"{self.experiment_name}_latest.pt"
        config = {
            "output_mode": self.output_mode,
            "experiment_name": self.experiment_name,
            "metric_for_best": self.metric_for_best,
        }
        current_metric = metrics.get(self.metric_for_best)
        if self.save_best and self._is_best(current_metric):
            self.best_metric = float(current_metric)
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
            save_checkpoint(
                best_path,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                metrics,
                config,
                best_metric=self.best_metric,
                history=self.history,
            )
        save_checkpoint(
            latest_path,
            self.model,
            self.optimizer,
            self.scheduler,
            epoch,
            metrics,
            config,
            best_metric=self.best_metric,
            history=self.history,
        )


def _accumulate(target: dict[str, float], values: dict[str, float], weight: int) -> None:
    for key, value in values.items():
        if value is None:
            continue
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if value_float != value_float:
            continue
        target[key] = target.get(key, 0.0) + value_float * weight


def _average_metrics(sums: dict[str, float], total_samples: int) -> dict[str, float]:
    denominator = max(1, int(total_samples))
    return {key: value / denominator for key, value in sums.items()}


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _tensor_dict_to_float(loss_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu().item()) for key, value in loss_dict.items()}


def _format_epoch_summary(metrics: dict[str, Any]) -> str:
    epoch = metrics.get("epoch", "?")
    keys = ["train_total_loss", "val_total_loss", "train_mae", "val_mae", "train_accuracy", "val_accuracy"]
    parts = [f"epoch={epoch}"]
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.6f}")
    return " | ".join(parts)


def _with_loss_alias(metrics: dict[str, float]) -> dict[str, float]:
    if "train_total_loss" in metrics:
        metrics["train_loss"] = metrics["train_total_loss"]
    if "val_total_loss" in metrics:
        metrics["val_loss"] = metrics["val_total_loss"]
    return metrics
