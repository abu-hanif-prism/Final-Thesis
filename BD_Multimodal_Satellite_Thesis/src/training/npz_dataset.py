"""PyTorch Dataset for final Siamese NPZ patch files."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


VALID_SPLITS = {"train", "val", "test"}
REQUIRED_NPZ_KEYS = {
    "image_t1",
    "image_t2",
    "tabular",
    "change_ratio",
    "patch_id",
    "pair_id",
    "district",
    "split",
    "change_class",
    "pair_type",
    "time_gap_group",
}
DEFAULT_CLASS_MAPPING = {"low": 0, "medium": 1, "high": 2}
EXPECTED_IMAGE_SHAPE = (13, 128, 128)
REQUIRED_INDEX_COLUMNS = {
    "patch_id",
    "npz_path",
    "split",
    "change_class",
    "change_ratio",
    "pair_id",
    "district",
}


def load_npz_index(index_path: str | Path) -> list[dict[str, Any]] | Any:
    """Load NPZ index from CSV or parquet.

    CSV loading intentionally uses only the Python standard library so training
    can avoid importing pandas, pyarrow, or fastparquet.
    """
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ index file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        print(f"Loading NPZ index with Python csv.DictReader: {path}")
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if suffix != ".parquet":
        raise ValueError(f"Unsupported NPZ index file type {suffix!r}; use .csv or .parquet")

    import pandas as pd

    errors: list[str] = []
    if importlib.util.find_spec("fastparquet") is not None:
        try:
            print(f"Loading NPZ index with parquet engine=fastparquet: {path}")
            return pd.read_parquet(path, engine="fastparquet")
        except Exception as exc:
            errors.append(f"fastparquet failed: {exc}")
    else:
        errors.append("fastparquet is not installed")

    try:
        print(f"Loading NPZ index with parquet engine=pyarrow: {path}")
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:
        errors.append(f"pyarrow failed: {exc}")

    detail = "\n".join(f"- {error}" for error in errors)
    raise RuntimeError(
        "Could not load NPZ parquet index. Use a CSV index with --index_path "
        "data/npz/final_npz_index.csv or install/fix a parquet engine.\n"
        f"Tried:\n{detail}"
    )


def clean_npz_index(index: list[dict[str, Any]] | Any) -> list[dict[str, Any]]:
    """Validate required columns and drop rows without usable NPZ paths."""
    rows = _index_to_rows(index)
    available_columns = set(rows[0].keys()) if rows else set()
    if not rows and hasattr(index, "columns"):
        available_columns = set(index.columns)

    missing_columns = sorted(REQUIRED_INDEX_COLUMNS.difference(available_columns))
    if missing_columns:
        raise KeyError(f"NPZ index is missing required columns: {missing_columns}")

    cleaned_rows = [row for row in rows if not _is_bad_npz_path(row.get("npz_path"))]
    dropped_count = len(rows) - len(cleaned_rows)
    if dropped_count:
        print(f"Warning: dropped {dropped_count} NPZ index rows with missing/empty npz_path.")

    return cleaned_rows


def _index_to_rows(index: list[dict[str, Any]] | Any) -> list[dict[str, Any]]:
    """Convert a loaded index object to plain Python rows."""
    if isinstance(index, list):
        return [dict(row) for row in index]
    if hasattr(index, "to_dict"):
        return index.to_dict(orient="records")
    raise TypeError(f"Unsupported NPZ index table type: {type(index).__name__}")


def _is_bad_npz_path(value: Any) -> bool:
    """Return True for missing, empty, or placeholder path values."""
    if value is None:
        return True
    text = str(value).strip()
    return text in {"", "None", "none", "NULL", "null", "NaN", "nan"}


class NPZSiameseDataset(Dataset):
    """Generic Siamese NPZ dataset for CNN and transformer backbones."""

    def __init__(
        self,
        index_path: str | Path,
        split: str | None = None,
        transform: Any | None = None,
        return_metadata: bool = True,
        target_mode: str = "regression",
        class_mapping: dict[str, int] | None = None,
        validate_paths: bool = False,
    ) -> None:
        self.index_path = Path(index_path)
        self.split = split
        self.transform = transform
        self.return_metadata = return_metadata
        self.target_mode = target_mode
        self.class_mapping = class_mapping or DEFAULT_CLASS_MAPPING.copy()

        if target_mode not in {"regression", "classification", "both"}:
            raise ValueError(
                "target_mode must be one of: regression, classification, both. "
                f"Got {target_mode!r}."
            )
        if split is not None and split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}. Got {split!r}.")
        rows = clean_npz_index(load_npz_index(self.index_path))
        if split is not None:
            rows = [row for row in rows if row.get("split") == split]

        self.rows = rows
        if validate_paths:
            missing = [
                str(path)
                for path in (self._resolve_npz_path(row["npz_path"]) for row in self.rows)
                if not path.exists()
            ]
            if missing:
                examples = "\n".join(missing[:5])
                raise FileNotFoundError(
                    f"{len(missing)} NPZ files listed in the index are missing. Examples:\n{examples}"
                )

    def __len__(self) -> int:
        """Return number of indexed NPZ samples."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one NPZ sample and return tensors plus optional metadata."""
        row = self.rows[index]
        npz_path = self._resolve_npz_path(row["npz_path"])
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found for index {index}: {npz_path}")

        with np.load(npz_path, allow_pickle=False) as npz:
            missing_keys = sorted(REQUIRED_NPZ_KEYS - set(npz.files))
            if missing_keys:
                raise KeyError(f"{npz_path} is missing required keys: {missing_keys}")

            image_t1_np = npz["image_t1"]
            image_t2_np = npz["image_t2"]
            tabular_np = npz["tabular"]
            self._validate_arrays(npz_path, image_t1_np, image_t2_np, tabular_np)

            image_t1 = torch.as_tensor(image_t1_np, dtype=torch.float32)
            image_t2 = torch.as_tensor(image_t2_np, dtype=torch.float32)
            tabular = torch.as_tensor(tabular_np, dtype=torch.float32)
            change_ratio = torch.as_tensor(float(npz["change_ratio"].item()), dtype=torch.float32)
            change_class = _scalar_to_str(npz["change_class"])
            change_class_id = torch.as_tensor(self._class_id(change_class), dtype=torch.long)

            sample: dict[str, Any] = {
                "image_t1": image_t1,
                "image_t2": image_t2,
                "tabular": tabular,
                "target": self._target_tensor(change_ratio, change_class_id),
                "change_ratio": change_ratio,
                "change_class_id": change_class_id,
            }

            if self.return_metadata:
                sample.update(
                    {
                        "patch_id": _scalar_to_str(npz["patch_id"]),
                        "pair_id": _scalar_to_str(npz["pair_id"]),
                        "district": _scalar_to_str(npz["district"]),
                        "split": _scalar_to_str(npz["split"]),
                        "change_class": change_class,
                        "pair_type": _scalar_to_str(npz["pair_type"]),
                        "time_gap_group": _scalar_to_str(npz["time_gap_group"]),
                    }
                )

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_split_counts(self) -> dict[str, int]:
        """Return split counts from the loaded index rows."""
        return _value_counts(self.rows, "split")

    def get_class_counts(self) -> dict[str, int]:
        """Return change-class counts from the loaded index rows."""
        return _value_counts(self.rows, "change_class")

    def get_tabular_dim(self) -> int:
        """Return tabular feature dimension from the first sample."""
        if len(self) == 0:
            raise ValueError("Cannot inspect tabular dimension of an empty dataset.")
        sample = self[0]
        return int(sample["tabular"].shape[0])

    def get_image_shape(self) -> tuple[int, ...]:
        """Return image tensor shape from the first sample."""
        if len(self) == 0:
            raise ValueError("Cannot inspect image shape of an empty dataset.")
        sample = self[0]
        return tuple(sample["image_t1"].shape)

    def validate_sample(self, index: int = 0) -> dict[str, Any]:
        """Load and validate one sample, returning a compact structure report."""
        sample = self[index]
        return {
            "index": index,
            "image_t1_shape": tuple(sample["image_t1"].shape),
            "image_t2_shape": tuple(sample["image_t2"].shape),
            "tabular_shape": tuple(sample["tabular"].shape),
            "target_mode": self.target_mode,
            "target_type": type(sample["target"]).__name__,
            "change_ratio": float(sample["change_ratio"].item()),
            "change_class_id": int(sample["change_class_id"].item()),
            "patch_id": sample.get("patch_id"),
            "change_class": sample.get("change_class"),
        }

    def _resolve_npz_path(self, value: Any) -> Path:
        path = Path(str(value))
        if path.is_absolute():
            return path
        project_candidate = Path.cwd() / path
        if project_candidate.exists():
            return project_candidate
        return self.index_path.parent / path

    def _class_id(self, change_class: str) -> int:
        if change_class not in self.class_mapping:
            raise KeyError(
                f"Unknown change_class {change_class!r}. "
                f"Known classes: {sorted(self.class_mapping)}"
            )
        return int(self.class_mapping[change_class])

    def _target_tensor(
        self,
        change_ratio: torch.Tensor,
        change_class_id: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.target_mode == "regression":
            return change_ratio
        if self.target_mode == "classification":
            return change_class_id
        return {"change_ratio": change_ratio, "change_class_id": change_class_id}

    @staticmethod
    def _validate_arrays(
        npz_path: Path,
        image_t1: np.ndarray,
        image_t2: np.ndarray,
        tabular: np.ndarray,
    ) -> None:
        if tuple(image_t1.shape) != EXPECTED_IMAGE_SHAPE:
            raise ValueError(
                f"{npz_path} image_t1 shape must be {EXPECTED_IMAGE_SHAPE}; "
                f"got {tuple(image_t1.shape)}"
            )
        if tuple(image_t2.shape) != EXPECTED_IMAGE_SHAPE:
            raise ValueError(
                f"{npz_path} image_t2 shape must be {EXPECTED_IMAGE_SHAPE}; "
                f"got {tuple(image_t2.shape)}"
            )
        if tabular.ndim != 1:
            raise ValueError(f"{npz_path} tabular must be 1D; got shape {tuple(tabular.shape)}")


def _scalar_to_str(value: np.ndarray) -> str:
    """Convert a scalar numpy value to a plain Python string."""
    return str(value.item())


def _value_counts(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    """Return sorted value counts for a list-of-dict index."""
    counts: dict[str, int] = {}
    for row in rows:
        if column not in row:
            continue
        key = str(row[column])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
