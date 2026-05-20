"""Utilities for saving final multimodal training patches as NPZ files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


SENTINEL_BAND_COUNT = 13


def read_sentinel_patch(path, x, y, patch_size=128):
    """Read a Sentinel patch window without loading the full raster.

    Returns an array with shape ``[13, patch_size, patch_size]`` as float32.
    The band order is preserved from the source raster.
    """
    raster_path = Path(path)
    window = Window(col_off=int(x), row_off=int(y), width=int(patch_size), height=int(patch_size))

    with rasterio.open(raster_path) as src:
        if src.count < SENTINEL_BAND_COUNT:
            raise ValueError(
                f"Expected at least {SENTINEL_BAND_COUNT} Sentinel bands, "
                f"found {src.count} in {raster_path}"
            )

        patch = src.read(
            indexes=list(range(1, SENTINEL_BAND_COUNT + 1)),
            window=window,
            boundless=False,
        )

    expected_shape = (SENTINEL_BAND_COUNT, int(patch_size), int(patch_size))
    if patch.shape != expected_shape:
        raise ValueError(f"Expected patch shape {expected_shape}, found {patch.shape} for {raster_path}")

    return patch.astype(np.float32, copy=False)


def normalize_sentinel_patch(patch):
    """Safely normalize a Sentinel patch to roughly the 0-1 range.

    Each band is robustly clipped to its 1st and 99th percentile within the
    patch, then scaled independently. NaN and infinite values are replaced with
    zero so saved training tensors are numerically stable.
    """
    patch = np.asarray(patch, dtype=np.float32)
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = np.zeros_like(patch, dtype=np.float32)

    for band_index in range(patch.shape[0]):
        band = patch[band_index]
        finite_mask = np.isfinite(band)
        if not finite_mask.any():
            continue

        values = band[finite_mask]
        low, high = np.percentile(values, [1, 99])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            continue

        scaled = (band - low) / (high - low)
        normalized[band_index] = np.clip(scaled, 0.0, 1.0)

    return normalized


def get_tabular_vector(pair_id, tabular_df, feature_columns):
    """Return the scaled tabular feature vector for a pair id."""
    if tabular_df.index.name == "pair_id":
        if pair_id not in tabular_df.index:
            raise KeyError(f"Pair id not found in tabular features: {pair_id}")
        row = tabular_df.loc[pair_id]
    else:
        matches = tabular_df.loc[tabular_df["pair_id"] == pair_id]
        if matches.empty:
            raise KeyError(f"Pair id not found in tabular features: {pair_id}")
        row = matches.iloc[0]

    missing_columns = [column for column in feature_columns if column not in row.index]
    if missing_columns:
        raise KeyError(f"Missing tabular feature columns: {missing_columns[:10]}")

    vector = pd.to_numeric(row.loc[list(feature_columns)], errors="coerce").to_numpy(dtype=np.float32)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def build_npz_sample(patch_row, tabular_df, feature_columns):
    """Build one multimodal training sample from a final patch metadata row."""
    image_t1 = normalize_sentinel_patch(
        read_sentinel_patch(
            patch_row["sentinel_path_t1"],
            patch_row["x"],
            patch_row["y"],
            patch_row.get("patch_size", 128),
        )
    )
    image_t2 = normalize_sentinel_patch(
        read_sentinel_patch(
            patch_row["sentinel_path_t2"],
            patch_row["x"],
            patch_row["y"],
            patch_row.get("patch_size", 128),
        )
    )
    tabular = get_tabular_vector(patch_row["pair_id"], tabular_df, feature_columns)

    return {
        "image_t1": image_t1,
        "image_t2": image_t2,
        "tabular": tabular,
        "change_ratio": np.float32(patch_row["change_ratio"]),
        "change_class": str(patch_row["change_class"]),
        "split": str(patch_row["split"]),
        "pair_type": str(patch_row.get("pair_type", "")),
        "time_gap_group": str(patch_row.get("time_gap_group", "")),
        "district": str(patch_row.get("district", "")),
        "pair_id": str(patch_row["pair_id"]),
        "patch_id": str(patch_row["patch_id"]),
    }


def save_npz_sample(sample_dict, output_path):
    """Save one multimodal sample as a compressed NPZ file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        image_t1=sample_dict["image_t1"].astype(np.float16),
        image_t2=sample_dict["image_t2"].astype(np.float16),
        tabular=sample_dict["tabular"].astype(np.float32),
        change_ratio=np.asarray(sample_dict["change_ratio"], dtype=np.float32),
        patch_id=np.asarray(sample_dict["patch_id"]),
        pair_id=np.asarray(sample_dict["pair_id"]),
        district=np.asarray(sample_dict["district"]),
        split=np.asarray(sample_dict["split"]),
        change_class=np.asarray(sample_dict["change_class"]),
        pair_type=np.asarray(sample_dict["pair_type"]),
        time_gap_group=np.asarray(sample_dict["time_gap_group"]),
    )


def safe_npz_filename(patch_id):
    """Create a Windows-friendly NPZ filename from a patch id."""
    unsafe_chars = '<>:"/\\|?*'
    filename = str(patch_id)
    for char in unsafe_chars:
        filename = filename.replace(char, "_")
    return f"{filename}.npz"


def process_patch_chunk(
    chunk_df,
    tabular_df,
    feature_columns,
    output_dir,
    force=False,
    progress_callback=None,
    progress_interval=100,
):
    """Save NPZ samples for a chunk of patch rows.

    Existing files are skipped unless ``force`` is true. Processing continues
    after per-row failures and returns index/error tables plus aggregate stats.
    """
    output_dir = Path(output_dir)
    index_rows = []
    error_rows = []
    saved_count = 0
    skipped_count = 0

    for row_number, (_, patch_row) in enumerate(chunk_df.iterrows(), start=1):
        patch_id = str(patch_row["patch_id"])
        split = str(patch_row["split"])
        output_path = output_dir / split / safe_npz_filename(patch_id)

        try:
            if output_path.exists() and not force:
                skipped_count += 1
            else:
                sample = build_npz_sample(patch_row, tabular_df, feature_columns)
                save_npz_sample(sample, output_path)
                saved_count += 1

            index_rows.append(
                {
                    "patch_id": patch_id,
                    "npz_path": str(output_path.resolve()),
                    "split": split,
                    "change_class": str(patch_row["change_class"]),
                    "change_ratio": float(patch_row["change_ratio"]),
                    "pair_id": str(patch_row["pair_id"]),
                    "district": str(patch_row.get("district", "")),
                }
            )
        except Exception as exc:  # noqa: BLE001 - row-level errors are reported and processing continues.
            error_rows.append(
                {
                    "patch_id": patch_id,
                    "pair_id": str(patch_row.get("pair_id", "")),
                    "split": split,
                    "error_message": str(exc),
                }
            )

        if progress_callback and (row_number % progress_interval == 0 or row_number == len(chunk_df)):
            progress_callback(
                {
                    "processed_count": int(row_number),
                    "saved_count": int(saved_count),
                    "skipped_count": int(skipped_count),
                    "failed_count": int(len(error_rows)),
                }
            )

    stats = {
        "processed_count": int(len(chunk_df)),
        "saved_count": int(saved_count),
        "skipped_count": int(skipped_count),
        "failed_count": int(len(error_rows)),
    }
    return pd.DataFrame(index_rows), pd.DataFrame(error_rows), stats


def estimate_npz_storage_bytes(index_rows: Iterable[dict]):
    """Estimate storage already written by summing NPZ file sizes."""
    total = 0
    for row in index_rows:
        path = Path(row["npz_path"])
        if path.exists():
            total += path.stat().st_size
    return total
