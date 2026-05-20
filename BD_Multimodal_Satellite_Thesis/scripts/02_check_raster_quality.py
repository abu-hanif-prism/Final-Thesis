"""Run raster metadata quality checks for Sentinel and Dynamic World rasters."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_all_configs  # noqa: E402
from src.data_inventory.raster_quality import (  # noqa: E402
    build_raster_quality_table,
    compare_sentinel_dw_alignment,
)
from src.utils.file_utils import ensure_dir  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - depends on optional local package.
    tqdm = None


def main() -> None:
    """Load inventories, inspect rasters, compare alignment, and save reports."""
    configs = load_all_configs()
    paths = configs.paths

    inventory_dir = ensure_dir(paths["metadata_dir"] / "inventory")
    reports_dir = ensure_dir(paths["output_dir"] / "reports")

    sentinel_inventory_path = inventory_dir / "sentinel_inventory.parquet"
    dw_inventory_path = inventory_dir / "dynamic_world_inventory.parquet"

    sentinel_inventory = _read_inventory(sentinel_inventory_path)
    dw_inventory = _read_inventory(dw_inventory_path)

    sentinel_quality_path = inventory_dir / "sentinel_raster_quality.parquet"
    dw_quality_path = inventory_dir / "dynamic_world_raster_quality.parquet"
    alignment_path = inventory_dir / "raster_alignment_check.parquet"

    print("Checking Sentinel raster metadata...")
    sentinel_quality = _build_quality_with_progress(
        sentinel_inventory,
        expected_band_count=13,
        label="Sentinel rasters",
    )
    sentinel_quality.to_parquet(sentinel_quality_path, index=False)

    print("Checking Dynamic World raster metadata...")
    dw_quality = _build_quality_with_progress(
        dw_inventory,
        expected_band_count=None,
        label="Dynamic World rasters",
    )
    dw_quality.to_parquet(dw_quality_path, index=False)

    print("Comparing Sentinel and Dynamic World alignment...")
    alignment = compare_sentinel_dw_alignment(sentinel_quality, dw_quality)
    alignment.to_parquet(alignment_path, index=False)

    summary = build_quality_summary(sentinel_quality, dw_quality, alignment)
    summary.to_csv(
        reports_dir / "raster_quality_summary.csv",
        index=False,
        encoding="utf-8",
    )

    print_quality_report(sentinel_quality, dw_quality, alignment)


def build_quality_summary(
    sentinel_quality: pd.DataFrame,
    dw_quality: pd.DataFrame,
    alignment: pd.DataFrame,
) -> pd.DataFrame:
    """Create a long-form CSV summary for raster quality outputs."""
    rows: list[dict[str, object]] = []

    rows.extend(_metric_rows("sentinel", sentinel_quality))
    rows.extend(_metric_rows("dynamic_world", dw_quality))
    rows.extend(_distribution_rows("sentinel_band_count", sentinel_quality, "band_count"))
    rows.extend(_distribution_rows("dynamic_world_band_count", dw_quality, "band_count"))
    rows.extend(_distribution_rows("sentinel_crs", sentinel_quality, "crs"))
    rows.extend(_distribution_rows("dynamic_world_crs", dw_quality, "crs"))
    rows.extend(_shape_distribution_rows("sentinel_shape", sentinel_quality))
    rows.extend(_shape_distribution_rows("dynamic_world_shape", dw_quality))
    rows.extend(_distribution_rows("alignment_status", alignment, "alignment_status"))

    return pd.DataFrame(rows, columns=["summary_type", "value", "count"])


def print_quality_report(
    sentinel_quality: pd.DataFrame,
    dw_quality: pd.DataFrame,
    alignment: pd.DataFrame,
) -> None:
    """Print the requested raster quality summary to the console."""
    print(f"Total Sentinel rasters checked: {len(sentinel_quality)}")
    print(f"Total Dynamic World rasters checked: {len(dw_quality)}")
    print(f"Unreadable Sentinel count: {_unreadable_count(sentinel_quality)}")
    print(f"Unreadable Dynamic World count: {_unreadable_count(dw_quality)}")
    print("Sentinel band count distribution:")
    _print_value_counts(sentinel_quality, "band_count")
    print("Dynamic World band count distribution:")
    _print_value_counts(dw_quality, "band_count")
    print(f"Sentinel files where band_count != 13: {_sentinel_bad_band_count(sentinel_quality)}")
    print("CRS distribution for Sentinel:")
    _print_value_counts(sentinel_quality, "crs")
    print("CRS distribution for Dynamic World:")
    _print_value_counts(dw_quality, "crs")
    print("Shape distribution for Sentinel:")
    _print_shape_counts(sentinel_quality)
    print("Shape distribution for Dynamic World:")
    _print_shape_counts(dw_quality)
    print("Alignment status counts:")
    _print_value_counts(alignment, "alignment_status")
    print("First 10 mismatched or unreadable examples:")
    _print_alignment_examples(alignment)


def _read_inventory(path: Path) -> pd.DataFrame:
    """Read an inventory parquet file with a useful missing-file error."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing inventory file: {path}. Run scripts/01_build_inventory.py first."
        )
    return pd.read_parquet(path)


def _build_quality_with_progress(
    inventory: pd.DataFrame,
    expected_band_count: int | None,
    label: str,
) -> pd.DataFrame:
    """Build quality data, using tqdm if it is installed."""
    if tqdm is None:
        return build_raster_quality_table(inventory, expected_band_count)

    success_count = int((inventory["parse_status"] == "success").sum())
    with tqdm(total=success_count, desc=label, unit="raster") as progress_bar:
        quality = build_raster_quality_table(inventory, expected_band_count)
        progress_bar.update(success_count)
    return quality


def _metric_rows(source: str, quality: pd.DataFrame) -> list[dict[str, object]]:
    """Return total and unreadable metric rows for one quality table."""
    return [
        {
            "summary_type": f"{source}_metric",
            "value": "total_rasters_checked",
            "count": int(len(quality)),
        },
        {
            "summary_type": f"{source}_metric",
            "value": "unreadable_count",
            "count": _unreadable_count(quality),
        },
    ]


def _distribution_rows(
    summary_type: str,
    df: pd.DataFrame,
    column: str,
) -> list[dict[str, object]]:
    """Return value-count rows for a DataFrame column."""
    if df.empty or column not in df:
        return []

    counts = df[column].dropna().value_counts().sort_index()
    return [
        {
            "summary_type": summary_type,
            "value": str(value),
            "count": int(count),
        }
        for value, count in counts.items()
    ]


def _shape_distribution_rows(
    summary_type: str,
    quality: pd.DataFrame,
) -> list[dict[str, object]]:
    """Return value-count rows for width x height shapes."""
    if quality.empty or "width" not in quality or "height" not in quality:
        return []

    shapes = quality.dropna(subset=["width", "height"]).copy()
    if shapes.empty:
        return []

    shape_labels = shapes["width"].astype(int).astype(str) + "x" + shapes[
        "height"
    ].astype(int).astype(str)
    counts = shape_labels.value_counts().sort_index()
    return [
        {
            "summary_type": summary_type,
            "value": str(value),
            "count": int(count),
        }
        for value, count in counts.items()
    ]


def _unreadable_count(quality: pd.DataFrame) -> int:
    """Count unreadable raster rows."""
    if quality.empty or "readable" not in quality:
        return 0
    return int((quality["readable"] != True).sum())  # noqa: E712


def _sentinel_bad_band_count(sentinel_quality: pd.DataFrame) -> int:
    """Count readable Sentinel files whose band count is not 13."""
    required_columns = {"readable", "band_count"}
    if sentinel_quality.empty or not required_columns.issubset(sentinel_quality.columns):
        return 0
    readable = sentinel_quality["readable"] == True  # noqa: E712
    has_band_count = sentinel_quality["band_count"].notna()
    bad_band_count = sentinel_quality["band_count"] != 13
    return int((readable & has_band_count & bad_band_count).sum())


def _print_value_counts(df: pd.DataFrame, column: str) -> None:
    """Print value counts for a DataFrame column."""
    if df.empty or column not in df:
        print("  none")
        return

    counts = df[column].dropna().value_counts().sort_index()
    if counts.empty:
        print("  none")
        return

    for value, count in counts.items():
        print(f"  {value}: {int(count)}")


def _print_shape_counts(quality: pd.DataFrame) -> None:
    """Print width x height distribution for a quality table."""
    rows = _shape_distribution_rows("shape", quality)
    if not rows:
        print("  none")
        return

    for row in rows:
        print(f"  {row['value']}: {row['count']}")


def _print_alignment_examples(alignment: pd.DataFrame) -> None:
    """Print the first 10 non-aligned alignment rows."""
    if alignment.empty or "alignment_status" not in alignment:
        print("  none")
        return

    examples = alignment[alignment["alignment_status"] != "aligned"].head(10)
    if examples.empty:
        print("  none")
        return

    for _, row in examples.iterrows():
        print(
            "  "
            f"{row['image_id']} | {row['alignment_status']} | "
            f"sentinel={row['sentinel_path']} | dw={row['dw_path']}"
        )


if __name__ == "__main__":
    main()
