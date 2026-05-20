"""Sentinel and Dynamic World inventory matching utilities."""

from pathlib import Path
import warnings
from typing import Any

import pandas as pd


MATCHED_COLUMNS = [
    "image_id",
    "district",
    "year",
    "season",
    "sentinel_path",
    "sentinel_filename",
    "sentinel_parse_status",
    "dw_path",
    "dw_filename",
    "dw_parse_status",
    "has_sentinel",
    "has_dynamic_world",
    "is_matched",
    "alignment_status",
    "is_aligned",
    "is_valid_for_training",
    "sentinel_band_count",
    "dw_band_count",
    "sentinel_readable",
    "dw_readable",
    "same_crs",
    "same_width",
    "same_height",
    "same_resolution",
    "same_bounds_approx",
]

REQUIRED_INVENTORY_COLUMNS = {
    "image_id",
    "filename",
    "full_path",
    "parse_status",
}

OPTIONAL_ID_COLUMNS = ["district", "year", "season"]


def load_inventory_files(metadata_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load Sentinel, Dynamic World, and raster alignment inventory files."""
    inventory_dir = Path(metadata_dir) / "inventory"
    return {
        "sentinel": pd.read_parquet(inventory_dir / "sentinel_inventory.parquet"),
        "dynamic_world": pd.read_parquet(
            inventory_dir / "dynamic_world_inventory.parquet"
        ),
        "alignment": pd.read_parquet(inventory_dir / "raster_alignment_check.parquet"),
    }


def create_matched_inventory(
    sentinel_inventory_df: pd.DataFrame,
    dynamic_world_inventory_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create one matched Sentinel/Dynamic World inventory row per image_id."""
    _warn_missing_columns(
        sentinel_inventory_df,
        REQUIRED_INVENTORY_COLUMNS,
        "Sentinel inventory",
    )
    _warn_missing_columns(
        dynamic_world_inventory_df,
        REQUIRED_INVENTORY_COLUMNS,
        "Dynamic World inventory",
    )
    _warn_missing_columns(alignment_df, {"image_id", "alignment_status"}, "alignment")

    sentinel_by_id = _index_by_image_id(sentinel_inventory_df)
    dw_by_id = _index_by_image_id(dynamic_world_inventory_df)
    alignment_by_id = _index_by_image_id(alignment_df)
    image_ids = sorted(set(sentinel_by_id) | set(dw_by_id) | set(alignment_by_id))

    records = [
        _build_matched_record(
            image_id,
            sentinel_by_id.get(image_id),
            dw_by_id.get(image_id),
            alignment_by_id.get(image_id),
        )
        for image_id in image_ids
    ]
    return pd.DataFrame(records, columns=MATCHED_COLUMNS)


def summarize_matched_inventory(matched_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return summary tables for a matched inventory DataFrame."""
    return {
        "metrics": _metrics_summary(matched_df),
        "alignment_status": _count_summary(
            matched_df,
            "alignment_status",
            "alignment_status",
        ),
        "district": _count_summary(matched_df, "district", "district"),
        "year": _count_summary(matched_df, "year", "year"),
        "season": _count_summary(matched_df, "season", "season"),
        "district_year_season": _district_year_season_summary(matched_df),
    }


def flatten_summary_tables(summary_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flatten matched inventory summary tables into one CSV-friendly table."""
    rows: list[dict[str, object]] = []
    for summary_name, table in summary_tables.items():
        if table.empty:
            continue
        for _, row in table.iterrows():
            output = {"summary_type": summary_name}
            output.update(row.to_dict())
            rows.append(output)
    return pd.DataFrame(rows)


def _build_matched_record(
    image_id: str,
    sentinel_row: pd.Series | None,
    dw_row: pd.Series | None,
    alignment_row: pd.Series | None,
) -> dict[str, Any]:
    """Build one matched output record from optional source rows."""
    source_row = alignment_row if alignment_row is not None else sentinel_row
    if source_row is None:
        source_row = dw_row

    sentinel_parse_status = _value(sentinel_row, "parse_status")
    dw_parse_status = _value(dw_row, "parse_status")
    has_sentinel = sentinel_row is not None and sentinel_parse_status == "success"
    has_dynamic_world = dw_row is not None and dw_parse_status == "success"
    is_matched = has_sentinel and has_dynamic_world

    alignment_status = _value(alignment_row, "alignment_status")
    is_aligned = alignment_status == "aligned"
    sentinel_readable = _value(alignment_row, "sentinel_readable")
    dw_readable = _value(alignment_row, "dw_readable")
    is_valid_for_training = bool(
        is_matched
        and is_aligned
        and sentinel_readable == True  # noqa: E712
        and dw_readable == True  # noqa: E712
    )

    return {
        "image_id": image_id,
        "district": _first_value("district", alignment_row, sentinel_row, dw_row),
        "year": _first_value("year", alignment_row, sentinel_row, dw_row),
        "season": _first_value("season", alignment_row, sentinel_row, dw_row),
        "sentinel_path": _first_value("sentinel_path", alignment_row)
        or _value(sentinel_row, "full_path"),
        "sentinel_filename": _value(sentinel_row, "filename"),
        "sentinel_parse_status": sentinel_parse_status,
        "dw_path": _first_value("dw_path", alignment_row) or _value(dw_row, "full_path"),
        "dw_filename": _value(dw_row, "filename"),
        "dw_parse_status": dw_parse_status,
        "has_sentinel": bool(has_sentinel),
        "has_dynamic_world": bool(has_dynamic_world),
        "is_matched": bool(is_matched),
        "alignment_status": alignment_status,
        "is_aligned": bool(is_aligned),
        "is_valid_for_training": is_valid_for_training,
        "sentinel_band_count": _value(alignment_row, "sentinel_band_count"),
        "dw_band_count": _value(alignment_row, "dw_band_count"),
        "sentinel_readable": sentinel_readable,
        "dw_readable": dw_readable,
        "same_crs": _value(alignment_row, "same_crs"),
        "same_width": _value(alignment_row, "same_width"),
        "same_height": _value(alignment_row, "same_height"),
        "same_resolution": _value(alignment_row, "same_resolution"),
        "same_bounds_approx": _value(alignment_row, "same_bounds_approx"),
    }


def _index_by_image_id(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Index rows by image_id, keeping the first non-null image_id occurrence."""
    if df.empty or "image_id" not in df:
        return {}

    indexed: dict[str, pd.Series] = {}
    for _, row in df.dropna(subset=["image_id"]).iterrows():
        image_id = str(row["image_id"])
        indexed.setdefault(image_id, row)
    return indexed


def _metrics_summary(matched_df: pd.DataFrame) -> pd.DataFrame:
    """Create top-level matched inventory metric rows."""
    metrics = [
        ("total_images", len(matched_df)),
        ("matched_count", _true_count(matched_df, "is_matched")),
        ("valid_for_training_count", _true_count(matched_df, "is_valid_for_training")),
        ("missing_sentinel_count", _false_count(matched_df, "has_sentinel")),
        ("missing_dynamic_world_count", _false_count(matched_df, "has_dynamic_world")),
    ]
    return pd.DataFrame(metrics, columns=["metric", "count"])


def _count_summary(
    matched_df: pd.DataFrame,
    column: str,
    label_column: str,
) -> pd.DataFrame:
    """Create a count table for a single matched inventory column."""
    if matched_df.empty or column not in matched_df:
        return pd.DataFrame(columns=[label_column, "count"])

    counts = matched_df[column].dropna().value_counts().sort_index()
    return pd.DataFrame(
        [{label_column: value, "count": int(count)} for value, count in counts.items()]
    )


def _district_year_season_summary(matched_df: pd.DataFrame) -> pd.DataFrame:
    """Count images by district/year/season."""
    columns = ["district", "year", "season"]
    if matched_df.empty or not set(columns).issubset(matched_df.columns):
        return pd.DataFrame(columns=[*columns, "count"])

    return (
        matched_df.dropna(subset=columns)
        .groupby(columns, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(columns)
        .reset_index(drop=True)
    )


def _true_count(df: pd.DataFrame, column: str) -> int:
    """Count True values in a boolean-like column."""
    if df.empty or column not in df:
        return 0
    return int((df[column] == True).sum())  # noqa: E712


def _false_count(df: pd.DataFrame, column: str) -> int:
    """Count False values in a boolean-like column."""
    if df.empty or column not in df:
        return 0
    return int((df[column] != True).sum())  # noqa: E712


def _first_value(column: str, *rows: pd.Series | None) -> Any:
    """Return the first non-empty value for a column across optional rows."""
    for row in rows:
        value = _value(row, column)
        if value is not None:
            return value
    return None


def _value(row: pd.Series | None, column: str) -> Any:
    """Safely read a scalar value from a pandas Series."""
    if row is None or column not in row:
        return None
    value = row[column]
    return None if pd.isna(value) else value


def _warn_missing_columns(
    df: pd.DataFrame,
    expected_columns: set[str],
    label: str,
) -> None:
    """Warn when an input table is missing expected columns."""
    missing = sorted(expected_columns - set(df.columns))
    if missing:
        warnings.warn(
            f"{label} is missing expected columns: {missing}",
            stacklevel=2,
        )
