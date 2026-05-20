"""Build Sentinel and Dynamic World GeoTIFF inventory files."""

from pathlib import Path
import importlib.util
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_all_configs  # noqa: E402
from src.data_inventory.scan_dynamic_world import scan_dynamic_world_files  # noqa: E402
from src.data_inventory.scan_sentinel import scan_sentinel_files  # noqa: E402
from src.utils.file_utils import ensure_dir  # noqa: E402


def build_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    """Build a long-form summary table for an inventory DataFrame."""
    rows = [
        {
            "summary_type": "metric",
            "value": "total_files_found",
            "count": int(len(inventory)),
        },
        {
            "summary_type": "metric",
            "value": "successfully_parsed_files",
            "count": _status_count(inventory, "success"),
        },
        {
            "summary_type": "metric",
            "value": "failed_parsed_files",
            "count": _status_count(inventory, "failed"),
        },
    ]

    rows.extend(_count_rows(inventory, "district", "count_by_district"))
    rows.extend(_count_rows(inventory, "year", "count_by_year"))
    rows.extend(_count_rows(inventory, "season", "count_by_season"))
    rows.extend(_count_rows(inventory, "parse_status", "count_by_parse_status"))

    return pd.DataFrame(rows, columns=["summary_type", "value", "count"])


def save_inventory_outputs(
    inventory: pd.DataFrame,
    inventory_path: Path,
    summary_path: Path,
) -> None:
    """Save inventory parquet and summary CSV outputs."""
    ensure_dir(inventory_path.parent)
    ensure_dir(summary_path.parent)

    _ensure_parquet_engine()
    inventory.to_parquet(inventory_path, index=False)
    build_summary(inventory).to_csv(summary_path, index=False, encoding="utf-8")


def print_inventory_report(
    sentinel_inventory: pd.DataFrame,
    dynamic_world_inventory: pd.DataFrame,
) -> None:
    """Print useful inventory summaries to the console."""
    print(f"Total Sentinel files found: {len(sentinel_inventory)}")
    print(f"Total Dynamic World files found: {len(dynamic_world_inventory)}")
    print(f"Sentinel successfully parsed count: {_status_count(sentinel_inventory, 'success')}")
    print(
        "Dynamic World successfully parsed count: "
        f"{_status_count(dynamic_world_inventory, 'success')}"
    )
    print(f"Sentinel failed parse count: {_status_count(sentinel_inventory, 'failed')}")
    print(
        "Dynamic World failed parse count: "
        f"{_status_count(dynamic_world_inventory, 'failed')}"
    )

    _print_top_districts("Sentinel", sentinel_inventory)
    _print_top_districts("Dynamic World", dynamic_world_inventory)
    _print_available_values("Sentinel years available", sentinel_inventory, "year")
    _print_available_values("Dynamic World years available", dynamic_world_inventory, "year")
    _print_available_values("Sentinel seasons available", sentinel_inventory, "season")
    _print_available_values("Dynamic World seasons available", dynamic_world_inventory, "season")


def main() -> None:
    """Build inventories from configured raw data directories."""
    configs = load_all_configs()
    paths = configs.paths

    sentinel_dir = paths["sentinel_drive_dir"]
    dynamic_world_dir = paths["dynamic_world_drive_dir"]
    metadata_dir = paths["metadata_dir"]
    output_dir = paths["output_dir"]

    inventory_dir = ensure_dir(metadata_dir / "inventory")
    reports_dir = ensure_dir(output_dir / "reports")

    sentinel_inventory = scan_sentinel_files(sentinel_dir)
    dynamic_world_inventory = scan_dynamic_world_files(dynamic_world_dir)

    save_inventory_outputs(
        sentinel_inventory,
        inventory_dir / "sentinel_inventory.parquet",
        reports_dir / "sentinel_inventory_summary.csv",
    )
    save_inventory_outputs(
        dynamic_world_inventory,
        inventory_dir / "dynamic_world_inventory.parquet",
        reports_dir / "dynamic_world_inventory_summary.csv",
    )

    print_inventory_report(sentinel_inventory, dynamic_world_inventory)


def _status_count(inventory: pd.DataFrame, status: str) -> int:
    """Count inventory rows with a given parse status."""
    if "parse_status" not in inventory:
        return 0
    return int((inventory["parse_status"] == status).sum())


def _ensure_parquet_engine() -> None:
    """Raise a clear error if pandas cannot write parquet files."""
    has_pyarrow = importlib.util.find_spec("pyarrow") is not None
    has_fastparquet = importlib.util.find_spec("fastparquet") is not None
    if not has_pyarrow and not has_fastparquet:
        raise RuntimeError(
            "Saving parquet inventory files requires pyarrow or fastparquet. "
            "Install one in the active environment, for example: pip install pyarrow"
        )


def _count_rows(
    inventory: pd.DataFrame,
    column: str,
    summary_type: str,
) -> list[dict[str, object]]:
    """Return count rows for non-empty values in a column."""
    if column not in inventory or inventory.empty:
        return []

    counts = inventory[column].dropna().value_counts().sort_index()
    return [
        {
            "summary_type": summary_type,
            "value": str(value),
            "count": int(count),
        }
        for value, count in counts.items()
    ]


def _print_top_districts(label: str, inventory: pd.DataFrame) -> None:
    """Print the top 10 districts by inventory row count."""
    print(f"{label} top 10 districts by file count:")
    if inventory.empty or "district" not in inventory:
        print("  none")
        return

    district_counts = inventory["district"].dropna().value_counts().head(10)
    if district_counts.empty:
        print("  none")
        return

    for district, count in district_counts.items():
        print(f"  {district}: {int(count)}")


def _print_available_values(label: str, inventory: pd.DataFrame, column: str) -> None:
    """Print sorted non-empty values available in an inventory column."""
    if inventory.empty or column not in inventory:
        values = []
    else:
        values = sorted(inventory[column].dropna().unique().tolist())

    print(f"{label}: {values}")


if __name__ == "__main__":
    main()
