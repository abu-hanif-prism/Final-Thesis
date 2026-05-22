"""DataLoader helpers for final Siamese NPZ patch datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.training.npz_dataset import NPZSiameseDataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Create a standard PyTorch DataLoader for an existing dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_train_val_test_dataloaders(
    index_path: str | Path = "data/npz/final_npz_index.parquet",
    batch_size: int = 32,
    num_workers: int = 0,
    target_mode: str = "regression",
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders from one NPZ index."""
    train_dataset = NPZSiameseDataset(index_path, split="train", target_mode=target_mode)
    val_dataset = NPZSiameseDataset(index_path, split="val", target_mode=target_mode)
    test_dataset = NPZSiameseDataset(index_path, split="test", target_mode=target_mode)
    return (
        create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        create_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        create_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )


def inspect_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Return tensor shapes, dtypes, and batch size for a collated batch."""
    report: dict[str, Any] = {}
    for key in ["image_t1", "image_t2", "tabular", "target", "change_ratio", "change_class_id"]:
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, torch.Tensor):
            report[f"{key}_shape"] = tuple(value.shape)
            report[f"{key}_dtype"] = str(value.dtype)
        elif isinstance(value, dict):
            report[f"{key}_shape"] = {
                sub_key: tuple(sub_value.shape)
                for sub_key, sub_value in value.items()
                if isinstance(sub_value, torch.Tensor)
            }
            report[f"{key}_dtype"] = {
                sub_key: str(sub_value.dtype)
                for sub_key, sub_value in value.items()
                if isinstance(sub_value, torch.Tensor)
            }

    image_t1 = batch.get("image_t1")
    report["batch_size"] = int(image_t1.shape[0]) if isinstance(image_t1, torch.Tensor) else 0
    return report
