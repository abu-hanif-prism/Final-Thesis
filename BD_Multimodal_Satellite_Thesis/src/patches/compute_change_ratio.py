"""Dynamic World patch-level change label computation.

The chunk labeler groups patches by Dynamic World raster pair so each t1/t2
raster is opened once per group. This is much faster on mounted Google Drive
than open/read/close per patch, because random drive metadata and directory
scans dominate small-window reads.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")

import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402


LABEL_COLUMNS = [
    "patch_id",
    "pair_id",
    "valid_pixel_count",
    "changed_pixel_count",
    "valid_pixel_ratio",
    "change_ratio",
    "change_class",
    "label_status",
    "label_error_message",
]


def read_dw_window(
    path: str | Path,
    x: int,
    y: int,
    patch_size: int,
    label_band: int = 10,
) -> tuple[np.ndarray, Any]:
    """Read one Dynamic World raster window from the selected label band.

    In this project export, Dynamic World bands 1-9 are class probability
    channels, band 10 is the modal land-cover class label
    (``dw_label_mode``), and band 11 is a forest mask. Change labels should
    normally use band 10 unless the caller explicitly overrides ``label_band``.
    """
    raster_path = Path(path)
    label_band = int(label_band)
    if label_band < 1:
        raise ValueError("label_band must be a 1-based positive integer.")

    with rasterio.open(raster_path) as dataset:
        if label_band > dataset.count:
            raise ValueError(
                f"Invalid Dynamic World label_band={label_band}; raster has "
                f"{dataset.count} bands. Expected band 10 for dw_label_mode in "
                f"this project export, unless intentionally overriding. Path: {raster_path}"
            )
        window = Window(
            col_off=int(x),
            row_off=int(y),
            width=int(patch_size),
            height=int(patch_size),
        )
        array = dataset.read(label_band, window=window, boundless=False)
        nodata = dataset.nodatavals[label_band - 1]
        if nodata is None:
            nodata = dataset.nodata

    return np.asarray(array), nodata


def compute_valid_mask(
    dw_t1: np.ndarray,
    dw_t2: np.ndarray,
    nodata_t1: Any = None,
    nodata_t2: Any = None,
) -> np.ndarray:
    """Return pixels valid in both Dynamic World arrays."""
    valid_mask = np.isfinite(dw_t1) & np.isfinite(dw_t2)
    if nodata_t1 is not None and not pd.isna(nodata_t1):
        valid_mask &= dw_t1 != nodata_t1
    if nodata_t2 is not None and not pd.isna(nodata_t2):
        valid_mask &= dw_t2 != nodata_t2
    return valid_mask


def compute_patch_change_ratio(
    dw_t1: np.ndarray,
    dw_t2: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float | int]:
    """Compute changed-pixel counts and change ratio for one patch."""
    if dw_t1.shape != dw_t2.shape:
        raise ValueError(f"Window shape mismatch: t1={dw_t1.shape}, t2={dw_t2.shape}")
    if valid_mask.shape != dw_t1.shape:
        raise ValueError(
            f"valid_mask shape {valid_mask.shape} does not match raster window {dw_t1.shape}."
        )

    valid_pixel_count = int(valid_mask.sum())
    total_pixel_count = int(valid_mask.size)
    valid_pixel_ratio = (
        float(valid_pixel_count / total_pixel_count) if total_pixel_count else 0.0
    )
    if valid_pixel_count == 0:
        return {
            "valid_pixel_count": 0,
            "changed_pixel_count": 0,
            "valid_pixel_ratio": valid_pixel_ratio,
            "change_ratio": float("nan"),
        }

    changed_pixel_count = int((dw_t1[valid_mask] != dw_t2[valid_mask]).sum())
    return {
        "valid_pixel_count": valid_pixel_count,
        "changed_pixel_count": changed_pixel_count,
        "valid_pixel_ratio": valid_pixel_ratio,
        "change_ratio": float(changed_pixel_count / valid_pixel_count),
    }


def assign_change_class(change_ratio: float) -> str:
    """Assign a low/medium/high change class from a change ratio."""
    if pd.isna(change_ratio):
        return "invalid"
    ratio = float(change_ratio)
    if 0.00 <= ratio < 0.05:
        return "low"
    if 0.05 <= ratio < 0.20:
        return "medium"
    if 0.20 <= ratio <= 1.00:
        return "high"
    return "invalid"


def compute_label_for_patch(
    row: pd.Series | dict[str, Any],
    patch_size: int = 128,
    min_valid_pixel_ratio: float = 0.80,
    label_band: int = 10,
) -> dict[str, Any]:
    """Compute one patch-level Dynamic World change label using DW band 10 by default."""
    patch_row = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    result = {
        "patch_id": patch_row.get("patch_id"),
        "pair_id": patch_row.get("pair_id"),
        "valid_pixel_count": None,
        "changed_pixel_count": None,
        "valid_pixel_ratio": None,
        "change_ratio": np.nan,
        "change_class": "invalid",
        "label_status": "failed",
        "label_error_message": None,
    }

    try:
        row_patch_size = int(patch_row.get("patch_size", patch_size))
        dw_t1, nodata_t1 = read_dw_window(
            patch_row["dw_path_t1"],
            patch_row["x"],
            patch_row["y"],
            row_patch_size,
            label_band=label_band,
        )
        dw_t2, nodata_t2 = read_dw_window(
            patch_row["dw_path_t2"],
            patch_row["x"],
            patch_row["y"],
            row_patch_size,
            label_band=label_band,
        )
        valid_mask = compute_valid_mask(dw_t1, dw_t2, nodata_t1, nodata_t2)
        metrics = compute_patch_change_ratio(dw_t1, dw_t2, valid_mask)
        change_class = assign_change_class(metrics["change_ratio"])
        label_status = (
            "invalid_low_valid_ratio"
            if metrics["valid_pixel_ratio"] < float(min_valid_pixel_ratio)
            else "success"
        )
        result.update(
            {
                **metrics,
                "change_class": change_class,
                "label_status": label_status,
                "label_error_message": None,
            }
        )
    except Exception as exc:  # noqa: BLE001 - one bad patch must not stop chunks.
        result["label_status"] = "failed"
        result["label_error_message"] = str(exc)

    return result


def compute_labels_for_raster_pair_group(
    group_df: pd.DataFrame,
    src_t1: Any,
    src_t2: Any,
    patch_size: int,
    min_valid_pixel_ratio: float,
    label_band: int,
    progress_callback: Any = None,
    processed_offset: int = 0,
    progress_interval: int = 500,
    raster_pair_group_index: int | None = None,
    raster_pair_group_total: int | None = None,
) -> pd.DataFrame:
    """Compute labels for one group using already-open rasterio datasets.

    The caller owns dataset open/close. This function reads only windowed
    band-10 class-label patches by default and never reopens rasters per patch.
    """
    _validate_label_band(src_t1, label_band, "dw_path_t1")
    _validate_label_band(src_t2, label_band, "dw_path_t2")
    nodata_t1 = _band_nodata(src_t1, label_band)
    nodata_t2 = _band_nodata(src_t2, label_band)
    records = []
    processed_count = int(processed_offset)

    for _, row in group_df.iterrows():
        try:
            row_patch_size = int(row.get("patch_size", patch_size))
            window = Window(
                col_off=int(row["x"]),
                row_off=int(row["y"]),
                width=row_patch_size,
                height=row_patch_size,
            )
            dw_t1 = src_t1.read(label_band, window=window, boundless=False)
            dw_t2 = src_t2.read(label_band, window=window, boundless=False)
            valid_mask = compute_valid_mask(dw_t1, dw_t2, nodata_t1, nodata_t2)
            metrics = compute_patch_change_ratio(dw_t1, dw_t2, valid_mask)
            change_class = assign_change_class(metrics["change_ratio"])
            label_status = (
                "invalid_low_valid_ratio"
                if metrics["valid_pixel_ratio"] < float(min_valid_pixel_ratio)
                else "success"
            )
            records.append(
                {
                    "patch_id": row.get("patch_id"),
                    "pair_id": row.get("pair_id"),
                    **metrics,
                    "change_class": change_class,
                    "label_status": label_status,
                    "label_error_message": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep the group moving.
            records.append(
                {
                    "patch_id": row.get("patch_id"),
                    "pair_id": row.get("pair_id"),
                    "valid_pixel_count": None,
                    "changed_pixel_count": None,
                    "valid_pixel_ratio": None,
                    "change_ratio": np.nan,
                    "change_class": "invalid",
                    "label_status": "failed",
                    "label_error_message": str(exc),
                }
            )

        processed_count += 1
        if (
            progress_callback is not None
            and progress_interval > 0
            and processed_count % progress_interval == 0
        ):
            progress_callback(
                processed_count=processed_count,
                raster_pair_group_index=raster_pair_group_index,
                raster_pair_group_total=raster_pair_group_total,
            )

    return pd.DataFrame(records, columns=LABEL_COLUMNS)


def compute_labels_for_chunk(
    chunk_df: pd.DataFrame,
    patch_size: int,
    min_valid_pixel_ratio: float,
    label_band: int,
    progress_callback: Any = None,
    progress_interval: int = 500,
) -> pd.DataFrame:
    """Compute Dynamic World labels for every patch row in one chunk.

    Patches are grouped by ``dw_path_t1``/``dw_path_t2``. Each raster pair is
    opened once, all windows in that group are read, then the datasets are
    closed. Existing chunk parquet outputs remain schema-compatible.
    """
    if chunk_df.empty:
        return pd.DataFrame(columns=LABEL_COLUMNS)

    required_columns = {"dw_path_t1", "dw_path_t2"}
    missing = required_columns - set(chunk_df.columns)
    if missing:
        raise ValueError(f"Chunk is missing required columns: {sorted(missing)}")

    label_tables = []
    group_keys = ["dw_path_t1", "dw_path_t2"]
    grouped = list(chunk_df.groupby(group_keys, sort=False, dropna=False))
    processed_offset = 0

    for group_position, ((dw_path_t1, dw_path_t2), group_df) in enumerate(grouped, start=1):
        try:
            with rasterio.open(dw_path_t1) as src_t1, rasterio.open(dw_path_t2) as src_t2:
                labels = compute_labels_for_raster_pair_group(
                    group_df=group_df,
                    src_t1=src_t1,
                    src_t2=src_t2,
                    patch_size=patch_size,
                    min_valid_pixel_ratio=min_valid_pixel_ratio,
                    label_band=label_band,
                    progress_callback=progress_callback,
                    processed_offset=processed_offset,
                    progress_interval=progress_interval,
                    raster_pair_group_index=group_position,
                    raster_pair_group_total=len(grouped),
                )
        except Exception as exc:  # noqa: BLE001 - mark the whole group failed.
            labels = _failed_labels_for_group(group_df, exc)
            if progress_callback is not None:
                progress_callback(
                    processed_count=processed_offset + len(group_df),
                    raster_pair_group_index=group_position,
                    raster_pair_group_total=len(grouped),
                )

        label_tables.append(labels)
        processed_offset += len(group_df)

    return pd.concat(label_tables, ignore_index=True)


def _validate_label_band(dataset: Any, label_band: int, label: str) -> None:
    """Validate that a requested 1-based label band exists in a dataset."""
    if int(label_band) < 1 or int(label_band) > int(dataset.count):
        raise ValueError(
            f"Invalid Dynamic World label_band={label_band} for {label}; "
            f"raster has {dataset.count} bands. Expected band 10 for dw_label_mode."
        )


def _band_nodata(dataset: Any, label_band: int) -> Any:
    """Return nodata for a selected 1-based band, falling back to dataset nodata."""
    nodata = dataset.nodatavals[int(label_band) - 1]
    return dataset.nodata if nodata is None else nodata


def _failed_labels_for_group(group_df: pd.DataFrame, exc: Exception) -> pd.DataFrame:
    """Return failed label rows for a raster-pair group that could not be opened."""
    records = [
        {
            "patch_id": row.get("patch_id"),
            "pair_id": row.get("pair_id"),
            "valid_pixel_count": None,
            "changed_pixel_count": None,
            "valid_pixel_ratio": None,
            "change_ratio": np.nan,
            "change_class": "invalid",
            "label_status": "failed",
            "label_error_message": str(exc),
        }
        for _, row in group_df.iterrows()
    ]
    return pd.DataFrame(records, columns=LABEL_COLUMNS)
