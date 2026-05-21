"""Select the final balanced patch metadata dataset."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_all_configs  # noqa: E402
from src.patches.select_final_patches import (  # noqa: E402
    create_final_patch_selection,
    create_selection_reports,
    load_valid_labeled_patches,
    summarize_success_label_filter,
)
from src.utils.file_utils import ensure_dir  # noqa: E402


SPLIT_TARGETS = {
    "train": 120000,
    "val": 20000,
    "test": 20000,
}
RANDOM_SEED = 42


def main() -> None:
    """Create and save final selected patch metadata files."""
    configs = load_all_configs()
    metadata_dir = configs.paths["metadata_dir"]
    output_dir = configs.paths["output_dir"]

    patch_dir = metadata_dir / "patches"
    final_dir = ensure_dir(metadata_dir / "final")
    reports_dir = ensure_dir(output_dir / "reports")

    labeled_path = patch_dir / "patch_index_labeled.parquet"
    if not labeled_path.exists():
        raise FileNotFoundError(
            f"Missing labeled patch index: {labeled_path}. "
            "Run scripts/10_compute_patch_labels.py first."
        )

    labeled = pd.read_parquet(labeled_path)
    valid = load_valid_labeled_patches(labeled)
    print_true_success_label_report(labeled, valid)
    final = create_final_patch_selection(
        labeled,
        split_targets=SPLIT_TARGETS,
        random_seed=RANDOM_SEED,
    )
    validate_final_selection(final)

    final.to_parquet(final_dir / "final_patch_dataset.parquet", index=False)
    for split in ["train", "val", "test"]:
        final[final["split"] == split].to_parquet(
            final_dir / f"final_patch_{split}.parquet",
            index=False,
        )

    create_selection_reports(labeled, final, reports_dir)
    print_final_selection_report(valid, final)


def print_final_selection_report(valid: pd.DataFrame, final: pd.DataFrame) -> None:
    """Print the requested final patch selection summary."""
    pair_counts = final.groupby("pair_id").size() if not final.empty else pd.Series(dtype=int)
    class_pair_counts = (
        final.groupby(["change_class", "pair_id"]).size()
        if not final.empty
        else pd.Series(dtype=int)
    )
    print(f"Original valid patch count: {len(valid)}")
    print(f"Final selected patch count: {len(final)}")
    print("Selected count by split:")
    _print_counts(final, "split")
    print("Selected count by change_class:")
    _print_counts(final, "change_class")
    print("Selected count by pair_type:")
    _print_counts(final, "pair_type")
    print("Selected count by time_gap_group:")
    _print_counts(final, "time_gap_group")
    print("Selected count by district top 20:")
    _print_counts(final, "district", top_n=20)
    print(f"Max patches per pair after balancing: {0 if pair_counts.empty else int(pair_counts.max())}")
    print("Max patches per pair by change_class:")
    if class_pair_counts.empty:
        print("  none")
    else:
        for change_class, max_count in class_pair_counts.groupby(level=0).max().sort_index().items():
            print(f"  {change_class}: {int(max_count)}")
    print(
        "Mean patches per pair after balancing: "
        f"{0.0 if pair_counts.empty else pair_counts.mean():.4f}"
    )
    reduction_ratio = 0.0 if len(valid) == 0 else len(final) / len(valid)
    print(f"Reduction ratio: {reduction_ratio:.6f}")
    print("First 5 final selected patches:")
    _print_patch_examples(final)


def _print_counts(df: pd.DataFrame, column: str, top_n: int | None = None) -> None:
    """Print value counts for one column."""
    if df.empty or column not in df:
        print("  none")
        return
    counts = df[column].value_counts()
    if top_n is None:
        counts = counts.sort_index()
    else:
        counts = counts.head(top_n)
    if counts.empty:
        print("  none")
        return
    for value, count in counts.items():
        print(f"  {value}: {int(count)}")


def print_true_success_label_report(labeled: pd.DataFrame, valid: pd.DataFrame) -> None:
    """Print the true success-only label distribution before selection."""
    print("True success label filter report:")
    for _, row in summarize_success_label_filter(labeled).iterrows():
        print(f"  {row['metric']}: {int(row['value'])}")
    print("True successful class distribution:")
    _print_counts(valid, "change_class")


def validate_final_selection(final: pd.DataFrame) -> None:
    """Validate required final-selection integrity constraints."""
    duplicate_count = int(final["patch_id"].duplicated().sum()) if "patch_id" in final else 0
    invalid_status = int((final["label_status"] != "success").sum()) if "label_status" in final else len(final)
    invalid_class = int((~final["change_class"].isin({"low", "medium", "high"})).sum())
    nan_ratio = int(final["change_ratio"].isna().sum())

    print("Integrity checks:")
    print(f"  duplicate patch_id count: {duplicate_count}")
    print(f"  non-success label_status count: {invalid_status}")
    print(f"  invalid change_class count: {invalid_class}")
    print(f"  NaN change_ratio count: {nan_ratio}")

    if duplicate_count:
        raise ValueError(f"Final selection contains duplicate patch_id rows: {duplicate_count}")
    if invalid_status:
        raise ValueError(f"Final selection contains non-success labels: {invalid_status}")
    if invalid_class:
        raise ValueError(f"Final selection contains invalid change_class rows: {invalid_class}")
    if nan_ratio:
        raise ValueError(f"Final selection contains NaN change_ratio rows: {nan_ratio}")


def _print_patch_examples(df: pd.DataFrame) -> None:
    """Print first five selected patch rows."""
    if df.empty:
        print("  none")
        return
    columns = ["patch_id", "pair_id", "split", "change_class", "change_ratio"]
    for _, row in df[columns].head(5).iterrows():
        print(
            "  "
            f"{row['patch_id']} | pair={row['pair_id']} | "
            f"split={row['split']} | class={row['change_class']} | "
            f"ratio={row['change_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
