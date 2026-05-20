"""Create district-level train/validation/test splits."""

from pathlib import Path
import json
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_all_configs  # noqa: E402
from src.splits.create_district_split import (  # noqa: E402
    assign_split_to_inventory,
    create_district_split,
    get_unique_districts,
    validate_split_integrity,
)
from src.utils.file_utils import ensure_dir  # noqa: E402


IMAGE_SPLIT_COLUMNS = [
    "image_id",
    "district",
    "year",
    "season",
    "sentinel_path",
    "dw_path",
    "split",
]


def main() -> None:
    """Create and save district-level splits from valid matched inventory only."""
    configs = load_all_configs()
    metadata_dir = configs.paths["metadata_dir"]
    output_dir = configs.paths["output_dir"]

    inventory_dir = ensure_dir(metadata_dir / "inventory")
    split_dir = ensure_dir(metadata_dir / "splits")
    reports_dir = ensure_dir(output_dir / "reports")

    matched_valid_path = inventory_dir / "matched_inventory_valid.parquet"
    if not matched_valid_path.exists():
        raise FileNotFoundError(
            f"Missing valid matched inventory: {matched_valid_path}. "
            "Run scripts/03_match_files.py first."
        )

    inventory = pd.read_parquet(matched_valid_path)
    inventory["year"] = pd.to_numeric(inventory["year"], errors="raise").astype(int)

    districts = get_unique_districts(inventory)
    train_districts, val_districts, test_districts = create_district_split(
        districts,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )
    split_inventory = assign_split_to_inventory(
        inventory,
        train_districts,
        val_districts,
        test_districts,
    )
    summary = validate_split_integrity(split_inventory)

    image_split = split_inventory[IMAGE_SPLIT_COLUMNS].copy()
    image_split.to_parquet(split_dir / "image_split.parquet", index=False)

    split_payload = {
        "random_seed": 42,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "train_districts": train_districts,
        "val_districts": val_districts,
        "test_districts": test_districts,
        "summary": summary,
    }
    save_json(split_payload, split_dir / "district_split.json")
    save_district_list(train_districts, split_dir / "train_districts.txt")
    save_district_list(val_districts, split_dir / "val_districts.txt")
    save_district_list(test_districts, split_dir / "test_districts.txt")

    build_summary_table(summary, len(districts)).to_csv(
        reports_dir / "district_split_summary.csv",
        index=False,
        encoding="utf-8",
    )

    print_split_report(
        total_district_count=len(districts),
        summary=summary,
        train_districts=train_districts,
        val_districts=val_districts,
        test_districts=test_districts,
    )


def save_json(data: dict[str, object], path: Path) -> None:
    """Save JSON data with UTF-8 encoding."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_district_list(districts: list[str], path: Path) -> None:
    """Save one district name per line."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(districts))
        file.write("\n")


def build_summary_table(summary: dict[str, int], total_district_count: int) -> pd.DataFrame:
    """Build a CSV-friendly split summary table."""
    rows = [{"metric": "total_district_count", "count": total_district_count}]
    rows.extend({"metric": metric, "count": count} for metric, count in summary.items())
    return pd.DataFrame(rows)


def print_split_report(
    total_district_count: int,
    summary: dict[str, int],
    train_districts: list[str],
    val_districts: list[str],
    test_districts: list[str],
) -> None:
    """Print the requested split summary to the console."""
    print(f"Total district count: {total_district_count}")
    print(f"Train district count: {summary['train_district_count']}")
    print(f"Validation district count: {summary['val_district_count']}")
    print(f"Test district count: {summary['test_district_count']}")
    print(f"Train image count: {summary['train_image_count']}")
    print(f"Validation image count: {summary['val_image_count']}")
    print(f"Test image count: {summary['test_image_count']}")
    print(f"Train district names: {train_districts}")
    print(f"Validation district names: {val_districts}")
    print(f"Test district names: {test_districts}")
    print(f"Overlap count: {summary['overlap_count']}")


if __name__ == "__main__":
    main()
