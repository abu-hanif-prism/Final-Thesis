"""Create matched Sentinel-Dynamic World inventory files and reports."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_all_configs  # noqa: E402
from src.data_inventory.match_files import (  # noqa: E402
    create_matched_inventory,
    flatten_summary_tables,
    load_inventory_files,
    summarize_matched_inventory,
)
from src.utils.file_utils import ensure_dir  # noqa: E402


def main() -> None:
    """Load inventories, create matched outputs, and save summary reports."""
    configs = load_all_configs()
    metadata_dir = configs.paths["metadata_dir"]
    output_dir = configs.paths["output_dir"]

    inventory_dir = ensure_dir(metadata_dir / "inventory")
    reports_dir = ensure_dir(output_dir / "reports")

    inventories = load_inventory_files(metadata_dir)
    sentinel_inventory = inventories["sentinel"]
    dynamic_world_inventory = inventories["dynamic_world"]
    alignment = inventories["alignment"]

    matched = create_matched_inventory(
        sentinel_inventory,
        dynamic_world_inventory,
        alignment,
    )
    valid_matched = matched[matched["is_valid_for_training"] == True].copy()  # noqa: E712

    matched.to_parquet(inventory_dir / "matched_inventory.parquet", index=False)
    valid_matched.to_parquet(
        inventory_dir / "matched_inventory_valid.parquet",
        index=False,
    )

    missing_report = build_missing_report(matched)
    missing_report.to_csv(
        reports_dir / "missing_sentinel_or_dw.csv",
        index=False,
        encoding="utf-8",
    )

    summary = flatten_summary_tables(summarize_matched_inventory(matched))
    summary.to_csv(
        reports_dir / "matched_inventory_summary.csv",
        index=False,
        encoding="utf-8",
    )

    alignment_problem_report = build_alignment_problem_report(matched)
    alignment_problem_report.to_csv(
        reports_dir / "alignment_problem_examples.csv",
        index=False,
        encoding="utf-8",
    )

    print_match_report(sentinel_inventory, dynamic_world_inventory, matched)


def build_missing_report(matched: pd.DataFrame) -> pd.DataFrame:
    """Return rows missing either a Sentinel or Dynamic World counterpart."""
    if matched.empty:
        return matched.copy()
    missing = matched[
        (matched["has_sentinel"] != True)  # noqa: E712
        | (matched["has_dynamic_world"] != True)  # noqa: E712
    ].copy()
    return missing.sort_values(["district", "year", "season", "image_id"])


def build_alignment_problem_report(matched: pd.DataFrame) -> pd.DataFrame:
    """Return matched rows with non-aligned or unreadable raster quality status."""
    if matched.empty:
        return matched.copy()
    problems = matched[
        (matched["is_matched"] == True)  # noqa: E712
        & (
            (matched["is_aligned"] != True)  # noqa: E712
            | (matched["sentinel_readable"] != True)  # noqa: E712
            | (matched["dw_readable"] != True)  # noqa: E712
        )
    ].copy()
    return problems.sort_values(["alignment_status", "district", "year", "season"])


def print_match_report(
    sentinel_inventory: pd.DataFrame,
    dynamic_world_inventory: pd.DataFrame,
    matched: pd.DataFrame,
) -> None:
    """Print the requested matched inventory summary to the console."""
    print(f"Total Sentinel inventory rows: {len(sentinel_inventory)}")
    print(f"Total Dynamic World inventory rows: {len(dynamic_world_inventory)}")
    print(f"Total matched inventory rows: {len(matched)}")
    print(f"Matched count: {_true_count(matched, 'is_matched')}")
    print(f"Valid for training count: {_true_count(matched, 'is_valid_for_training')}")
    print(f"Missing Sentinel count: {_false_count(matched, 'has_sentinel')}")
    print(f"Missing Dynamic World count: {_false_count(matched, 'has_dynamic_world')}")
    print("Alignment status counts:")
    _print_value_counts(matched, "alignment_status")
    print("Valid image count by district top 20:")
    _print_valid_counts(matched, "district", top_n=20)
    print("Valid image count by year:")
    _print_valid_counts(matched, "year")
    print("Valid image count by season:")
    _print_valid_counts(matched, "season")
    print("First 10 invalid examples:")
    _print_invalid_examples(matched)


def _valid_rows(matched: pd.DataFrame) -> pd.DataFrame:
    """Return rows valid for future training."""
    if matched.empty or "is_valid_for_training" not in matched:
        return matched.iloc[0:0]
    return matched[matched["is_valid_for_training"] == True].copy()  # noqa: E712


def _true_count(df: pd.DataFrame, column: str) -> int:
    """Count True values in a boolean-like column."""
    if df.empty or column not in df:
        return 0
    return int((df[column] == True).sum())  # noqa: E712


def _false_count(df: pd.DataFrame, column: str) -> int:
    """Count non-True values in a boolean-like column."""
    if df.empty or column not in df:
        return 0
    return int((df[column] != True).sum())  # noqa: E712


def _print_value_counts(df: pd.DataFrame, column: str) -> None:
    """Print value counts for a column."""
    if df.empty or column not in df:
        print("  none")
        return

    counts = df[column].dropna().value_counts().sort_index()
    if counts.empty:
        print("  none")
        return

    for value, count in counts.items():
        print(f"  {value}: {int(count)}")


def _print_valid_counts(
    matched: pd.DataFrame,
    column: str,
    top_n: int | None = None,
) -> None:
    """Print counts for valid training rows grouped by a column."""
    valid = _valid_rows(matched)
    if valid.empty or column not in valid:
        print("  none")
        return

    counts = valid[column].dropna().value_counts()
    if top_n is None:
        counts = counts.sort_index()
    else:
        counts = counts.head(top_n)

    if counts.empty:
        print("  none")
        return

    for value, count in counts.items():
        print(f"  {value}: {int(count)}")


def _print_invalid_examples(matched: pd.DataFrame) -> None:
    """Print the first 10 rows not valid for future training."""
    if matched.empty or "is_valid_for_training" not in matched:
        print("  none")
        return

    invalid = matched[matched["is_valid_for_training"] != True].head(10)  # noqa: E712
    if invalid.empty:
        print("  none")
        return

    for _, row in invalid.iterrows():
        print(
            "  "
            f"{row['image_id']} | matched={row['is_matched']} | "
            f"alignment={row['alignment_status']} | "
            f"has_sentinel={row['has_sentinel']} | "
            f"has_dynamic_world={row['has_dynamic_world']}"
        )


if __name__ == "__main__":
    main()
