"""Training utility helpers for multimodal Siamese models."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, Torch, and CUDA when available."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return cuda when requested and available, otherwise cpu."""
    device = torch.device("cuda" if prefer_gpu and torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    return device


def move_batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    """Move tensor values in a batch dictionary to device."""
    device = torch.device(device)
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            moved[key] = {
                nested_key: nested_value.to(device, non_blocking=True)
                if isinstance(nested_value, torch.Tensor)
                else nested_value
                for nested_key, nested_value in value.items()
            }
        else:
            moved[key] = value
    return moved


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Create AdamW optimizer."""
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "plateau",
    **kwargs: Any,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """Create optional learning-rate scheduler."""
    if scheduler_type == "none":
        return None
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 3),
        )
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 30),
            eta_min=kwargs.get("eta_min", 0.0),
        )
    raise ValueError("scheduler_type must be one of: none, plateau, cosine")


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Save model, optimizer, scheduler, epoch, metrics, and config."""
    path = Path(path)
    ensure_dir(path.parent)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "metrics": metrics,
        "config": config,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load checkpoint into model and optionally optimizer/scheduler."""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if missing and return it as Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
