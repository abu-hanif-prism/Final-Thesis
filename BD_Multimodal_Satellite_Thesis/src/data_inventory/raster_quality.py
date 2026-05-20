"""Raster metadata inspection and alignment quality checks."""

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import rasterio


RASTER_METADATA_COLUMNS = [
    "full_path",
    "readable",
    "crs",
    "width",
    "height",
    "band_count",
    "dtype",
    "nodata",
    "transform",
    "bounds_left",
    "bounds_bottom",
    "bounds_right",
    "bounds_top",
    "resolution_x",
    "resolution_y",
    "file_size_mb",
    "error_message",
]

QUALITY_COLUMNS = [
    "image_id",
    "district",
    "year",
    "season",
    "filename",
    *RASTER_METADATA_COLUMNS,
    "expected_band_count",
    "band_count_ok",
]

ALIGNMENT_COLUMNS = [
    "image_id",
    "district",
    "year",
    "season",
    "sentinel_path",
    "dw_path",
    "sentinel_readable",
    "dw_readable",
    "same_crs",
    "same_width",
    "same_height",
    "same_resolution",
    "same_bounds_approx",
    "sentinel_band_count",
    "dw_band_count",
    "alignment_status",
]

DEFAULT_TOLERANCE = 1e-6


def inspect_raster(path: str | Path) -> dict[str, Any]:
    """Inspect GeoTIFF metadata without loading the full raster array."""
    raster_path = Path(path)
    result = _empty_raster_metadata(raster_path)

    try:
        result["file_size_mb"] = _file_size_mb(raster_path)
        with rasterio.open(raster_path) as dataset:
            bounds = dataset.bounds
            resolution_x, resolution_y = dataset.res
            result.update(
                {
                    "readable": True,
                    "crs": str(dataset.crs) if dataset.crs else None,
                    "width": int(dataset.width),
                    "height": int(dataset.height),
                    "band_count": int(dataset.count),
                    "dtype": _format_dtypes(dataset.dtypes),
                    "nodata": dataset.nodata,
                    "transform": _format_transform(dataset.transform),
                    "bounds_left": float(bounds.left),
                    "bounds_bottom": float(bounds.bottom),
                    "bounds_right": float(bounds.right),
                    "bounds_top": float(bounds.top),
                    "resolution_x": float(resolution_x),
                    "resolution_y": float(resolution_y),
                    "error_message": None,
                }
            )
    except Exception as exc:  # noqa: BLE001 - one bad raster must not stop checks.
        result["readable"] = False
        result["error_message"] = str(exc)

    return result


def build_raster_quality_table(
    inventory_df: pd.DataFrame,
    expected_band_count: int | None = None,
) -> pd.DataFrame:
    """Inspect successfully parsed inventory rows and return raster quality data."""
    records = []
    success_df = _success_rows(inventory_df)

    for _, row in success_df.iterrows():
        metadata = inspect_raster(row["full_path"])
        band_count = metadata["band_count"]
        band_count_ok = (
            None
            if expected_band_count is None
            else band_count == expected_band_count
        )
        records.append(
            {
                "image_id": row.get("image_id"),
                "district": row.get("district"),
                "year": row.get("year"),
                "season": row.get("season"),
                "filename": row.get("filename"),
                **metadata,
                "expected_band_count": expected_band_count,
                "band_count_ok": band_count_ok,
            }
        )

    return pd.DataFrame(records, columns=QUALITY_COLUMNS)


def compare_sentinel_dw_alignment(
    sentinel_quality_df: pd.DataFrame,
    dw_quality_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare Sentinel and Dynamic World raster alignment by image_id."""
    sentinel_by_id = _index_by_image_id(sentinel_quality_df)
    dw_by_id = _index_by_image_id(dw_quality_df)
    image_ids = sorted(set(sentinel_by_id) | set(dw_by_id))

    records = [
        _compare_one_image_id(image_id, sentinel_by_id.get(image_id), dw_by_id.get(image_id))
        for image_id in image_ids
    ]
    return pd.DataFrame(records, columns=ALIGNMENT_COLUMNS)


def _empty_raster_metadata(path: Path) -> dict[str, Any]:
    """Create a raster metadata result with a stable schema."""
    return {
        "full_path": str(path.resolve()),
        "readable": False,
        "crs": None,
        "width": None,
        "height": None,
        "band_count": None,
        "dtype": None,
        "nodata": None,
        "transform": None,
        "bounds_left": None,
        "bounds_bottom": None,
        "bounds_right": None,
        "bounds_top": None,
        "resolution_x": None,
        "resolution_y": None,
        "file_size_mb": None,
        "error_message": None,
    }


def _file_size_mb(path: Path) -> float | None:
    """Return file size in MiB if the file can be statted."""
    try:
        return round(path.stat().st_size / (1024 * 1024), 4)
    except OSError:
        return None


def _format_dtypes(dtypes: Iterable[str]) -> str | None:
    """Format raster band dtypes as a compact string."""
    dtype_list = list(dtypes)
    if not dtype_list:
        return None
    unique_dtypes = sorted(set(dtype_list))
    return unique_dtypes[0] if len(unique_dtypes) == 1 else ",".join(unique_dtypes)


def _format_transform(transform: Any) -> str:
    """Format an affine transform as six GDAL-style numeric values."""
    return ",".join(f"{value:.12g}" for value in transform.to_gdal())


def _success_rows(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with parse_status success, or an empty compatible DataFrame."""
    if inventory_df.empty or "parse_status" not in inventory_df:
        return inventory_df.iloc[0:0]
    return inventory_df[inventory_df["parse_status"] == "success"].copy()


def _index_by_image_id(quality_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Index quality rows by image_id, keeping the first row for duplicates."""
    if quality_df.empty or "image_id" not in quality_df:
        return {}

    indexed: dict[str, pd.Series] = {}
    for _, row in quality_df.dropna(subset=["image_id"]).iterrows():
        image_id = str(row["image_id"])
        indexed.setdefault(image_id, row)
    return indexed


def _compare_one_image_id(
    image_id: str,
    sentinel_row: pd.Series | None,
    dw_row: pd.Series | None,
) -> dict[str, Any]:
    """Compare one Sentinel/Dynamic World raster pair."""
    base_row = sentinel_row if sentinel_row is not None else dw_row
    record = {
        "image_id": image_id,
        "district": _value(base_row, "district"),
        "year": _value(base_row, "year"),
        "season": _value(base_row, "season"),
        "sentinel_path": _value(sentinel_row, "full_path"),
        "dw_path": _value(dw_row, "full_path"),
        "sentinel_readable": _value(sentinel_row, "readable"),
        "dw_readable": _value(dw_row, "readable"),
        "same_crs": None,
        "same_width": None,
        "same_height": None,
        "same_resolution": None,
        "same_bounds_approx": None,
        "sentinel_band_count": _value(sentinel_row, "band_count"),
        "dw_band_count": _value(dw_row, "band_count"),
        "alignment_status": None,
    }

    if sentinel_row is None:
        record["alignment_status"] = "missing_sentinel"
        return record
    if dw_row is None:
        record["alignment_status"] = "missing_dynamic_world"
        return record
    if not bool(record["sentinel_readable"]) or not bool(record["dw_readable"]):
        record["alignment_status"] = "unreadable"
        return record

    record.update(
        {
            "same_crs": _value(sentinel_row, "crs") == _value(dw_row, "crs"),
            "same_width": _value(sentinel_row, "width") == _value(dw_row, "width"),
            "same_height": _value(sentinel_row, "height") == _value(dw_row, "height"),
            "same_resolution": _same_resolution(sentinel_row, dw_row),
            "same_bounds_approx": _same_bounds(sentinel_row, dw_row),
        }
    )
    alignment_flags = [
        record["same_crs"],
        record["same_width"],
        record["same_height"],
        record["same_resolution"],
        record["same_bounds_approx"],
    ]
    record["alignment_status"] = (
        "aligned" if all(alignment_flags) else "mismatch"
    )
    return record


def _same_resolution(left: pd.Series, right: pd.Series) -> bool:
    """Compare x/y raster resolution using the default tolerance."""
    return _close(_value(left, "resolution_x"), _value(right, "resolution_x")) and _close(
        _value(left, "resolution_y"), _value(right, "resolution_y")
    )


def _same_bounds(left: pd.Series, right: pd.Series) -> bool:
    """Compare raster bounds using the default tolerance."""
    return all(
        _close(_value(left, column), _value(right, column))
        for column in [
            "bounds_left",
            "bounds_bottom",
            "bounds_right",
            "bounds_top",
        ]
    )


def _close(left: Any, right: Any, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """Return True when two numeric values are within tolerance."""
    if pd.isna(left) or pd.isna(right):
        return False
    return abs(float(left) - float(right)) <= tolerance


def _value(row: pd.Series | None, column: str) -> Any:
    """Safely read a value from a pandas Series."""
    if row is None or column not in row:
        return None
    value = row[column]
    return None if pd.isna(value) else value
