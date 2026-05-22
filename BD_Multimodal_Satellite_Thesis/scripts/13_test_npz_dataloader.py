"""Smoke-test PyTorch Dataset and DataLoader utilities for final NPZ patches."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.dataloaders import (  # noqa: E402
    create_dataloader,
    create_train_val_test_dataloaders,
    inspect_batch,
)
from src.training.npz_dataset import NPZSiameseDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test final NPZ PyTorch DataLoaders.")
    parser.add_argument(
        "--index_path",
        default="data/npz/final_npz_index.parquet",
        help="Path to final_npz_index.parquet.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--target_mode",
        choices=["regression", "classification", "both"],
        default="regression",
    )
    parser.add_argument("--pin_memory", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = Path(args.index_path)
    index_df = pd.read_parquet(index_path)
    print(f"Index path: {index_path}")
    print(f"Total index rows: {len(index_df)}")

    datasets = {
        split: NPZSiameseDataset(index_path, split=split, target_mode=args.target_mode)
        for split in ["train", "val", "test"]
    }
    print_dataset_summary(datasets)
    print_sample_structure(datasets["train"])

    print("\nDataLoader batch checks:")
    train_loader, val_loader, test_loader = create_train_val_test_dataloaders(
        index_path=index_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_mode=args.target_mode,
        pin_memory=args.pin_memory,
    )
    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        batch = next(iter(loader))
        print_batch_report(split, batch)

    print("\nTarget mode checks:")
    for target_mode in ["regression", "classification", "both"]:
        dataset = NPZSiameseDataset(index_path, split="train", target_mode=target_mode)
        loader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
        batch = next(iter(loader))
        target = batch["target"]
        if isinstance(target, torch.Tensor):
            print(f"  {target_mode}: target shape={tuple(target.shape)}, dtype={target.dtype}")
        else:
            shapes = {key: tuple(value.shape) for key, value in target.items()}
            dtypes = {key: str(value.dtype) for key, value in target.items()}
            print(f"  {target_mode}: target shapes={shapes}, dtypes={dtypes}")


def print_dataset_summary(datasets: dict[str, NPZSiameseDataset]) -> None:
    print("\nDataset sizes:")
    for split, dataset in datasets.items():
        print(f"  {split}: {len(dataset)}")

    train_dataset = datasets["train"]
    print(f"Image shape: {train_dataset.get_image_shape()}")
    print(f"Tabular dimension: {train_dataset.get_tabular_dim()}")

    print("\nClass counts by split:")
    for split, dataset in datasets.items():
        counts = dataset.get_class_counts()
        print(f"  {split}:")
        for change_class, count in counts.items():
            print(f"    {change_class}: {int(count)}")


def print_sample_structure(dataset: NPZSiameseDataset) -> None:
    sample = dataset[0]
    print("\nOne sample structure:")
    for key in ["image_t1", "image_t2", "tabular", "target", "change_ratio", "change_class_id"]:
        value = sample[key]
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        elif isinstance(value, dict):
            shapes = {sub_key: tuple(sub_value.shape) for sub_key, sub_value in value.items()}
            dtypes = {sub_key: str(sub_value.dtype) for sub_key, sub_value in value.items()}
            print(f"  {key}: shapes={shapes}, dtypes={dtypes}")

    print("  metadata:")
    for key in ["patch_id", "pair_id", "district", "split", "change_class", "pair_type", "time_gap_group"]:
        print(f"    {key}: {sample.get(key)}")


def print_batch_report(split: str, batch: dict[str, Any]) -> None:
    report = inspect_batch(batch)
    print(f"\n  {split}:")
    print(f"    batch size: {report['batch_size']}")
    for key, value in report.items():
        if key == "batch_size":
            continue
        print(f"    {key}: {value}")

    print("    NaN/inf checks:")
    for key in ["image_t1", "image_t2", "tabular", "target"]:
        result = finite_check(batch[key])
        print(f"      {key}: {result}")

    print("    first metadata values:")
    for key in ["patch_id", "pair_id", "district", "change_class"]:
        values = batch.get(key, [])
        shown = list(values[:3]) if isinstance(values, list) else values
        print(f"      {key}: {shown}")


def finite_check(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        has_nan = bool(torch.isnan(value).any().item()) if value.is_floating_point() else False
        has_inf = bool(torch.isinf(value).any().item()) if value.is_floating_point() else False
        return f"has_nan={has_nan}, has_inf={has_inf}"
    if isinstance(value, dict):
        return {
            key: finite_check(sub_value)
            for key, sub_value in value.items()
            if isinstance(sub_value, torch.Tensor)
        }
    return "not a tensor"


if __name__ == "__main__":
    main()
