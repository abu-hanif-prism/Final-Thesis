"""Patch coordinate index creation for tabular-complete temporal pairs."""

from pathlib import Path
from typing import Any

import pandas as pd
import rasterio

from src.patches.create_patch_grid import create_patch_grid

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional local dependency.
    tqdm = None


PATCH_INDEX_COLUMNS = [
    "patch_id",
    "pair_id",
    "district",
    "split",
    "image_id_t1",
    "image_id_t2",
    "year_t1",
    "season_t1",
    "year_t2",
    "season_t2",
    "pair_type",
    "time_gap_group",
    "x",
    "y",
    "patch_size",
    "stride",
    "sentinel_path_t1",
    "sentinel_path_t2",
    "dw_path_t1",
    "dw_path_t2",
]

ERROR_COLUMNS = ["pair_id", "district", "split", "error_message"]


def get_raster_shape(path: str | Path) -> dict[str, Any]:
    """Read raster metadata shape without loading full raster arrays."""
    raster_path = Path(path)
    with rasterio.open(raster_path) as dataset:
        return {
            "width": int(dataset.width),
            "height": int(dataset.height),
            "band_count": int(dataset.count),
            "crs": str(dataset.crs) if dataset.crs else None,
            "transform": ",".join(f"{value:.12g}" for value in dataset.transform.to_gdal()),
        }


def create_patch_index_for_pair(
    pair_row: pd.Series | dict[str, Any],
    patch_size: int = 128,
    stride: int = 64,
) -> pd.DataFrame:
    """Create patch coordinate rows for one temporal pair."""
    row = _as_mapping(pair_row)
    shape_t1 = get_raster_shape(row["sentinel_path_t1"])
    shape_t2 = get_raster_shape(row["sentinel_path_t2"])
    _validate_same_shape(shape_t1, shape_t2)
    return _create_pair_rows(row, shape_t1["width"], shape_t1["height"], patch_size, stride)


def create_patch_index(
    pair_df: pd.DataFrame,
    patch_size: int = 128,
    stride: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create patch coordinate rows for all pairs, recording pair-level errors."""
    patch_tables = []
    errors = []
    shape_cache: dict[str, dict[str, Any]] = {}
    iterator = pair_df.iterrows()
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(pair_df), desc="Creating patch index")

    for _, pair_row in iterator:
        row = pair_row.to_dict()
        try:
            shape_t1 = _get_shape_cached(row["sentinel_path_t1"], shape_cache)
            shape_t2 = _get_shape_cached(row["sentinel_path_t2"], shape_cache)
            _validate_same_shape(shape_t1, shape_t2)
            patch_tables.append(
                _create_pair_rows(
                    row,
                    shape_t1["width"],
                    shape_t1["height"],
                    patch_size,
                    stride,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep processing other pairs.
            errors.append(
                {
                    "pair_id": row.get("pair_id"),
                    "district": row.get("district"),
                    "split": row.get("split"),
                    "error_message": str(exc),
                }
            )

    if patch_tables:
        patch_index = pd.concat(patch_tables, ignore_index=True)
    else:
        patch_index = pd.DataFrame(columns=PATCH_INDEX_COLUMNS)

    error_df = pd.DataFrame(errors, columns=ERROR_COLUMNS)
    return patch_index, error_df


def split_patch_index(
    patch_index_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a patch index into train, validation, and test DataFrames."""
    if "split" not in patch_index_df.columns:
        raise ValueError("Patch index must contain a 'split' column.")
    train_df = patch_index_df[patch_index_df["split"] == "train"].copy()
    val_df = patch_index_df[patch_index_df["split"] == "val"].copy()
    test_df = patch_index_df[patch_index_df["split"] == "test"].copy()
    return train_df, val_df, test_df


def _create_pair_rows(
    row: dict[str, Any],
    width: int,
    height: int,
    patch_size: int,
    stride: int,
) -> pd.DataFrame:
    """Create vectorized patch index rows for one pair."""
    grid = create_patch_grid(width, height, patch_size=patch_size, stride=stride)
    if grid.empty:
        return pd.DataFrame(columns=PATCH_INDEX_COLUMNS)

    output = grid.copy()
    output["patch_id"] = (
        row["pair_id"]
        + "_x"
        + output["x"].astype(str)
        + "_y"
        + output["y"].astype(str)
    )
    output["pair_id"] = row["pair_id"]
    output["district"] = row["district"]
    output["split"] = row["split"]
    output["image_id_t1"] = row["image_id_t1"]
    output["image_id_t2"] = row["image_id_t2"]
    output["year_t1"] = int(row["year_t1"])
    output["season_t1"] = row["season_t1"]
    output["year_t2"] = int(row["year_t2"])
    output["season_t2"] = row["season_t2"]
    output["pair_type"] = row["pair_type"]
    output["time_gap_group"] = row["time_gap_group"]
    output["sentinel_path_t1"] = row["sentinel_path_t1"]
    output["sentinel_path_t2"] = row["sentinel_path_t2"]
    output["dw_path_t1"] = row["dw_path_t1"]
    output["dw_path_t2"] = row["dw_path_t2"]
    return output[PATCH_INDEX_COLUMNS]


def _get_shape_cached(
    path: str | Path,
    shape_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return cached raster shape metadata for a path."""
    key = str(path)
    if key not in shape_cache:
        shape_cache[key] = get_raster_shape(path)
    return shape_cache[key]


def _validate_same_shape(shape_t1: dict[str, Any], shape_t2: dict[str, Any]) -> None:
    """Validate Sentinel t1 and t2 raster shapes match."""
    if shape_t1["width"] != shape_t2["width"] or shape_t1["height"] != shape_t2["height"]:
        raise ValueError(
            "Sentinel t1/t2 shape mismatch: "
            f"t1={shape_t1['width']}x{shape_t1['height']}, "
            f"t2={shape_t2['width']}x{shape_t2['height']}"
        )


def _as_mapping(pair_row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Convert a pair row to a plain dictionary."""
    if isinstance(pair_row, pd.Series):
        return pair_row.to_dict()
    return dict(pair_row)
