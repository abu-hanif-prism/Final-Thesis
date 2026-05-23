"""Repair the final NPZ CSV index from the official parquet index.

Default mode uses data/npz/final_npz_index.parquet as the source of truth so
old test NPZ files on disk are not accidentally added back to training.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


NPZ_ROOT = Path("data/npz")
SPLITS = ("train", "val", "test")
OFFICIAL_INDEX = NPZ_ROOT / "final_npz_index.parquet"
OUTPUT_CSV = NPZ_ROOT / "final_npz_index.csv"
REPORT_CSV = Path("outputs/reports/npz_index_repair_summary.csv")
METADATA_KEYS = ("patch_id", "pair_id", "district", "split", "change_class", "change_ratio")


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def scalar_to_python(value: np.ndarray) -> object:
    """Convert a numpy scalar or single-value array to a Python value."""
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return array.tolist()


def load_npz_metadata(npz_path: Path) -> dict[str, object]:
    """Load required metadata fields from one NPZ file."""
    with np.load(npz_path, allow_pickle=False) as data:
        missing_keys = [key for key in METADATA_KEYS if key not in data.files]
        if missing_keys:
            raise KeyError(f"missing metadata keys: {missing_keys}")
        return {key: scalar_to_python(data[key]) for key in METADATA_KEYS}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Repair final_npz_index.csv safely.")
    parser.add_argument(
        "--scan_files",
        action="store_true",
        help="Build the CSV by scanning actual NPZ files instead of using the official parquet.",
    )
    return parser.parse_args()


def load_official_index() -> pd.DataFrame:
    """Load the official parquet index, preferring fastparquet over pyarrow."""
    if not OFFICIAL_INDEX.exists():
        raise FileNotFoundError(f"Official NPZ index not found: {OFFICIAL_INDEX}")
    if importlib.util.find_spec("fastparquet") is not None:
        print(f"Loading official index with parquet engine=fastparquet: {OFFICIAL_INDEX}")
        return pd.read_parquet(OFFICIAL_INDEX, engine="fastparquet")

    if importlib.util.find_spec("pyarrow") is not None:
        print("Warning: fastparquet is not installed; falling back to parquet engine=pyarrow.")
        print(f"Loading official index with parquet engine=pyarrow: {OFFICIAL_INDEX}")
        return pd.read_parquet(OFFICIAL_INDEX, engine="pyarrow")

    raise RuntimeError(
        "No parquet engine is available for official mode. Install fastparquet "
        "or run with --scan_files for filesystem scan mode."
    )


def invalid_npz_path_mask(df: pd.DataFrame) -> pd.Series:
    """Return mask for missing, empty, or placeholder NPZ paths."""
    npz_path_text = df["npz_path"].astype("string").str.strip()
    return (
        df["npz_path"].isna()
        | npz_path_text.isna()
        | npz_path_text.isin(["", "None", "none", "NULL", "null", "NaN", "nan"])
    )


def resolve_index_path(path_value: object) -> Path:
    """Resolve an index path value relative to the current project if needed."""
    path = Path(str(path_value))
    return path.resolve()


def reconstruct_missing_npz_paths(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fill missing NPZ paths from official split and patch_id when possible."""
    if not {"patch_id", "split", "npz_path"}.issubset(df.columns):
        return df, 0

    repaired_df = df.copy()
    bad_path_mask = invalid_npz_path_mask(repaired_df)
    repaired_count = 0
    for row_index, row in repaired_df.loc[bad_path_mask].iterrows():
        patch_id = str(row["patch_id"]).strip()
        split = str(row["split"]).strip()
        if not patch_id or not split or patch_id in {"None", "nan"} or split not in SPLITS:
            continue

        candidate = (NPZ_ROOT / split / f"{patch_id}.npz").resolve()
        if candidate.exists():
            repaired_df.at[row_index, "npz_path"] = str(candidate)
            repaired_count += 1

    return repaired_df, repaired_count


def list_disk_npz_files() -> list[Path]:
    """List NPZ files currently present under train/val/test folders."""
    files: list[Path] = []
    for split in SPLITS:
        split_dir = NPZ_ROOT / split
        if split_dir.exists():
            files.extend(sorted(split_dir.glob("*.npz")))
    return files


def build_index_from_scan() -> tuple[pd.DataFrame, dict[str, object], list[str]]:
    """Scan NPZ files and return a clean index plus scan-specific details."""
    rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    disk_files = list_disk_npz_files()

    for npz_path in disk_files:
        try:
            metadata = load_npz_metadata(npz_path)
            rows.append(
                {
                    "patch_id": metadata["patch_id"],
                    "npz_path": str(npz_path.resolve()),
                    "split": metadata["split"],
                    "change_class": metadata["change_class"],
                    "change_ratio": metadata["change_ratio"],
                    "pair_id": metadata["pair_id"],
                    "district": metadata["district"],
                }
            )
        except Exception as exc:  # Continue repairing if one NPZ is malformed.
            error_rows.append(
                {
                    "file": str(npz_path),
                    "error": str(exc),
                }
            )

    index_df = pd.DataFrame(
        rows,
        columns=["patch_id", "npz_path", "split", "change_class", "change_ratio", "pair_id", "district"],
    )
    details = {
        "official_index_row_count": "",
        "missing_npz_path_rows_dropped": 0,
        "missing_metadata_count": len(error_rows),
        "total_npz_files_found": len(disk_files),
        "extra_npz_files_on_disk_count": 0,
    }
    error_values = [f"{row['file']} | {row['error']}" for row in error_rows[:50]]
    return index_df, details, error_values


def build_index_from_official() -> tuple[pd.DataFrame, dict[str, object], list[str]]:
    """Build the clean CSV index from the official parquet index."""
    official_df = load_official_index()
    official_row_count = len(official_df)
    if "npz_path" not in official_df.columns:
        raise KeyError("Official index is missing required column: npz_path")

    official_df, reconstructed_npz_path_count = reconstruct_missing_npz_paths(official_df)
    bad_path_mask = invalid_npz_path_mask(official_df)
    if int(bad_path_mask.sum()) and importlib.util.find_spec("pyarrow") is not None:
        print(
            "Warning: fastparquet output still has incomplete official paths; "
            "falling back to pyarrow for this repair run."
        )
        official_df = pd.read_parquet(OFFICIAL_INDEX, engine="pyarrow")
        official_row_count = len(official_df)
        official_df, reconstructed_npz_path_count = reconstruct_missing_npz_paths(official_df)
        bad_path_mask = invalid_npz_path_mask(official_df)

    clean_df = official_df.loc[~bad_path_mask].copy()

    resolved_paths = clean_df["npz_path"].map(resolve_index_path)
    missing_files = [str(path) for path in resolved_paths if not path.exists()]

    disk_files = list_disk_npz_files()
    official_path_set = {str(path) for path in resolved_paths}
    extra_files = [path for path in disk_files if str(path.resolve()) not in official_path_set]

    details = {
        "official_index_row_count": official_row_count,
        "missing_npz_path_rows_dropped": int(bad_path_mask.sum()),
        "missing_files_count": len(missing_files),
        "total_npz_files_found": len(disk_files),
        "extra_npz_files_on_disk_count": len(extra_files),
        "reconstructed_npz_path_count": reconstructed_npz_path_count,
    }
    error_values = [f"missing_file_{idx}: {path}" for idx, path in enumerate(missing_files[:50], start=1)]
    extra_values = [path.name for path in extra_files[:20]]
    return clean_df, details, error_values + [f"extra_npz_file_{idx}: {name}" for idx, name in enumerate(extra_values, start=1)]


def build_summary(index_df: pd.DataFrame, details: dict[str, object], notes: list[str]) -> pd.DataFrame:
    """Create a summary report for the repaired CSV index."""
    split_counts = index_df["split"].value_counts().to_dict() if "split" in index_df else {}
    class_counts = index_df["change_class"].value_counts().to_dict() if "change_class" in index_df else {}
    duplicate_patch_id_count = (
        int(index_df["patch_id"].duplicated(keep=False).sum()) if "patch_id" in index_df else 0
    )

    summary_rows = [
        {"metric": "official_index_row_count", "value": details.get("official_index_row_count", "")},
        {"metric": "cleaned_csv_row_count", "value": len(index_df)},
        {
            "metric": "missing_npz_path_rows_dropped",
            "value": details.get("missing_npz_path_rows_dropped", 0),
        },
        {"metric": "missing_files_count", "value": details.get("missing_files_count", 0)},
        {"metric": "reconstructed_npz_path_count", "value": details.get("reconstructed_npz_path_count", 0)},
        {"metric": "total_npz_files_found", "value": details.get("total_npz_files_found", "")},
        {"metric": "train_count", "value": int(split_counts.get("train", 0))},
        {"metric": "val_count", "value": int(split_counts.get("val", 0))},
        {"metric": "test_count", "value": int(split_counts.get("test", 0))},
        {"metric": "low_count", "value": int(class_counts.get("low", 0))},
        {"metric": "medium_count", "value": int(class_counts.get("medium", 0))},
        {"metric": "high_count", "value": int(class_counts.get("high", 0))},
        {
            "metric": "extra_npz_files_on_disk_count",
            "value": details.get("extra_npz_files_on_disk_count", 0),
        },
        {"metric": "missing_metadata_count", "value": details.get("missing_metadata_count", 0)},
        {"metric": "duplicate_patch_id_count", "value": duplicate_patch_id_count},
        {"metric": "output_csv_path", "value": str(OUTPUT_CSV)},
    ]
    for index, note in enumerate(notes[:70], start=1):
        summary_rows.append({"metric": f"note_{index}", "value": note})

    return pd.DataFrame(summary_rows, columns=["metric", "value"])


def print_summary(summary_df: pd.DataFrame) -> None:
    """Print the repair summary in the required console format."""
    summary = dict(zip(summary_df["metric"], summary_df["value"]))
    print("NPZ index repair complete")
    print(f"Official index row count: {summary.get('official_index_row_count', '')}")
    print(f"Cleaned CSV row count: {summary.get('cleaned_csv_row_count', 0)}")
    print(f"Missing npz_path rows dropped: {summary.get('missing_npz_path_rows_dropped', 0)}")
    print(f"Missing files count: {summary.get('missing_files_count', 0)}")
    print(f"Train count: {summary.get('train_count', 0)}")
    print(f"Val count: {summary.get('val_count', 0)}")
    print(f"Test count: {summary.get('test_count', 0)}")
    print(f"Low count: {summary.get('low_count', 0)}")
    print(f"Medium count: {summary.get('medium_count', 0)}")
    print(f"High count: {summary.get('high_count', 0)}")
    print(f"Extra NPZ files on disk not in official index count: {summary.get('extra_npz_files_on_disk_count', 0)}")

    extra_notes = [
        str(value).replace("extra_npz_file_", "", 1)
        for metric, value in zip(summary_df["metric"], summary_df["value"])
        if str(value).startswith("extra_npz_file_")
    ]
    if extra_notes:
        print("First extra NPZ file names:")
        for note in extra_notes[:20]:
            print(f"  {note}")

    print(f"Output CSV path: {OUTPUT_CSV}")
    print(f"Repair summary path: {REPORT_CSV}")


def main() -> None:
    """Create data/npz/final_npz_index.csv."""
    args = parse_args()
    ensure_dir(OUTPUT_CSV.parent)
    ensure_dir(REPORT_CSV.parent)

    if args.scan_files:
        print("Running optional scan mode from actual NPZ files.")
        index_df, details, notes = build_index_from_scan()
    else:
        print("Running default official mode from final_npz_index.parquet.")
        index_df, details, notes = build_index_from_official()

    summary_df = build_summary(index_df, details, notes)
    index_df.to_csv(OUTPUT_CSV, index=False)
    summary_df.to_csv(REPORT_CSV, index=False)
    print_summary(summary_df)


if __name__ == "__main__":
    main()
