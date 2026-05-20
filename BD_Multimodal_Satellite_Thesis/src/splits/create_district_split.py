"""District-level train/validation/test split utilities."""

import random
from typing import Iterable

import pandas as pd


VALID_SPLITS = {"train", "val", "test"}


def get_unique_districts(df: pd.DataFrame) -> list[str]:
    """Return sorted unique district names from an inventory DataFrame."""
    if "district" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'district' column.")

    districts = df["district"].dropna().astype(str).unique().tolist()
    return sorted(districts)


def create_district_split(
    districts: Iterable[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle districts reproducibly and split them into train/val/test lists."""
    district_list = sorted({str(district) for district in districts})
    if not district_list:
        raise ValueError("No districts were provided for splitting.")
    if len(district_list) < 3:
        raise ValueError("At least three districts are required for train/val/test splits.")

    ratio_total = train_ratio + val_ratio + test_ratio
    if abs(ratio_total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0.")

    shuffled = district_list.copy()
    random.Random(random_seed).shuffle(shuffled)

    total = len(shuffled)
    val_count = max(1, round(total * val_ratio))
    test_count = max(1, round(total * test_ratio))
    train_count = total - val_count - test_count
    if train_count < 1:
        raise ValueError("Split ratios leave no districts for training.")

    train_districts = sorted(shuffled[:train_count])
    val_districts = sorted(shuffled[train_count:train_count + val_count])
    test_districts = sorted(shuffled[train_count + val_count:])

    _validate_district_sets(train_districts, val_districts, test_districts, total)
    return train_districts, val_districts, test_districts


def assign_split_to_inventory(
    inventory_df: pd.DataFrame,
    train_districts: Iterable[str],
    val_districts: Iterable[str],
    test_districts: Iterable[str],
) -> pd.DataFrame:
    """Assign a split label to every inventory row based on district membership."""
    if "district" not in inventory_df.columns:
        raise ValueError("Input inventory must contain a 'district' column.")

    train_set = set(train_districts)
    val_set = set(val_districts)
    test_set = set(test_districts)
    split_lookup = {
        **{district: "train" for district in train_set},
        **{district: "val" for district in val_set},
        **{district: "test" for district in test_set},
    }

    output = inventory_df.copy()
    output["split"] = output["district"].map(split_lookup)

    missing = sorted(output.loc[output["split"].isna(), "district"].dropna().unique())
    if missing:
        raise ValueError(f"Districts missing from split assignment: {missing}")
    if output["split"].isna().any():
        raise ValueError("Some rows have missing district values and cannot be split.")

    return output


def validate_split_integrity(split_df: pd.DataFrame) -> dict[str, int]:
    """Validate split integrity and return district/image count summary."""
    required_columns = {"district", "split"}
    missing_columns = required_columns - set(split_df.columns)
    if missing_columns:
        raise ValueError(f"Split DataFrame is missing columns: {sorted(missing_columns)}")

    if split_df["split"].isna().any():
        raise ValueError("Split DataFrame contains missing split values.")

    invalid_splits = sorted(set(split_df["split"].dropna()) - VALID_SPLITS)
    if invalid_splits:
        raise ValueError(f"Invalid split labels found: {invalid_splits}")

    district_sets = {
        split: set(split_df.loc[split_df["split"] == split, "district"].dropna())
        for split in VALID_SPLITS
    }
    overlap_count = _overlap_count(district_sets)
    if overlap_count:
        raise ValueError(f"District overlap detected across splits: {overlap_count}")

    return {
        "train_district_count": len(district_sets["train"]),
        "val_district_count": len(district_sets["val"]),
        "test_district_count": len(district_sets["test"]),
        "train_image_count": int((split_df["split"] == "train").sum()),
        "val_image_count": int((split_df["split"] == "val").sum()),
        "test_image_count": int((split_df["split"] == "test").sum()),
        "overlap_count": overlap_count,
    }


def _validate_district_sets(
    train_districts: list[str],
    val_districts: list[str],
    test_districts: list[str],
    expected_total: int,
) -> None:
    """Validate generated split district sets."""
    if not val_districts or not test_districts:
        raise ValueError("Validation and test splits must each contain at least one district.")

    district_sets = {
        "train": set(train_districts),
        "val": set(val_districts),
        "test": set(test_districts),
    }
    overlap_count = _overlap_count(district_sets)
    if overlap_count:
        raise ValueError(f"District overlap detected across splits: {overlap_count}")

    combined_count = len(set().union(*district_sets.values()))
    if combined_count != expected_total:
        raise ValueError(
            "District split did not assign every district exactly once. "
            f"Expected {expected_total}, got {combined_count}."
        )


def _overlap_count(district_sets: dict[str, set[str]]) -> int:
    """Count districts that appear in more than one split set."""
    train_val = district_sets["train"] & district_sets["val"]
    train_test = district_sets["train"] & district_sets["test"]
    val_test = district_sets["val"] & district_sets["test"]
    return len(train_val | train_test | val_test)
