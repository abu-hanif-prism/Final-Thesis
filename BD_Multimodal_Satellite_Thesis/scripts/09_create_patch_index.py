"""Create patch coordinate index metadata for tabular-complete pairs."""

from pathlib import Path
import os
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid expensive sibling-file scans when opening GeoTIFFs from mounted drives.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

from src.config.settings import load_all_configs  # noqa: E402
from src.patches.create_patch_index import create_patch_index, split_patch_index  # noqa: E402
from src.utils.file_utils import ensure_dir  # noqa: E402


def main() -> None:
    """Build and save patch coordinate indexes for tabular-complete pairs."""
    configs = load_all_configs()
    metadata_dir = configs.paths["metadata_dir"]
    output_dir = configs.paths["output_dir"]
    patch_size = int(configs.patching.get("patch_size", 128))
    stride = int(configs.patching.get("stride", 64))

    pair_dir = ensure_dir(metadata_dir / "pairs")
    patch_dir = ensure_dir(metadata_dir / "patches")
    reports_dir = ensure_dir(output_dir / "reports")

    pair_path = pair_dir / "constrained_pair_index_tabular_complete.parquet"
    if not pair_path.exists():
        raise FileNotFoundError(
            f"Missing tabular-complete pair index: {pair_path}. "
            "Run scripts/08b_filter_tabular_complete_pairs.py first."
        )

    pairs = pd.read_parquet(pair_path)
    for column in ["year_t1", "year_t2"]:
        pairs[column] = pd.to_numeric(pairs[column], errors="raise").astype(int)

    patch_index, errors = create_patch_index(
        pairs,
        patch_size=patch_size,
        stride=stride,
    )
    train_index, val_index, test_index = split_patch_index(patch_index)

    patch_index.to_parquet(patch_dir / "patch_index_all.parquet", index=False)
    train_index.to_parquet(patch_dir / "patch_index_train.parquet", index=False)
    val_index.to_parquet(patch_dir / "patch_index_val.parquet", index=False)
    test_index.to_parquet(patch_dir / "patch_index_test.parquet", index=False)

    build_patch_summary(pairs, patch_index, errors, patch_size).to_csv(
        reports_dir / "patch_index_summary.csv",
        index=False,
        encoding="utf-8",
    )
    errors.to_csv(
        reports_dir / "patch_index_errors.csv",
        index=False,
        encoding="utf-8",
    )

    print_patch_report(pairs, patch_index, errors, patch_size)


def build_patch_summary(
    pairs: pd.DataFrame,
    patch_index: pd.DataFrame,
    errors: pd.DataFrame,
    patch_size: int,
) -> pd.DataFrame:
    """Build a long-form patch index summary report."""
    rows = [
        {"summary_type": "metric", "value": "total_input_pairs", "count": len(pairs)},
        {"summary_type": "metric", "value": "total_patch_rows", "count": len(patch_index)},
        {"summary_type": "metric", "value": "pair_error_count", "count": len(errors)},
    ]
    if not patch_index.empty:
        patch_counts = patch_index.groupby("pair_id").size()
        rows.extend(
            [
                {
                    "summary_type": "metric",
                    "value": "min_patches_per_pair",
                    "count": int(patch_counts.min()),
                },
                {
                    "summary_type": "metric",
                    "value": "max_patches_per_pair",
                    "count": int(patch_counts.max()),
                },
                {
                    "summary_type": "metric",
                    "value": "mean_patches_per_pair",
                    "count": float(patch_counts.mean()),
                },
                {
                    "summary_type": "metric",
                    "value": "estimated_float16_image_pair_storage_gb",
                    "count": _estimate_storage_gb(len(patch_index), patch_size),
                },
            ]
        )
    rows.extend(_count_rows(patch_index, "split", "patch_rows_by_split"))
    rows.extend(_count_rows(patch_index, "district", "patch_rows_by_district"))
    rows.extend(_count_rows(patch_index, "pair_type", "patch_rows_by_pair_type"))
    rows.extend(_count_rows(patch_index, "time_gap_group", "patch_rows_by_time_gap_group"))
    return pd.DataFrame(rows, columns=["summary_type", "value", "count"])


def print_patch_report(
    pairs: pd.DataFrame,
    patch_index: pd.DataFrame,
    errors: pd.DataFrame,
    patch_size: int,
) -> None:
    """Print the requested patch index summary."""
    print(f"Total input pairs: {len(pairs)}")
    print(f"Total patch rows: {len(patch_index)}")
    print("Patch rows by split:")
    _print_value_counts(patch_index, "split")
    print("Patch rows by district top 20:")
    _print_value_counts(patch_index, "district", top_n=20)
    print("Patch rows by pair_type:")
    _print_value_counts(patch_index, "pair_type")
    print("Patch rows by time_gap_group:")
    _print_value_counts(patch_index, "time_gap_group")
    _print_patch_count_stats(patch_index)
    print(
        "Estimated saved image-pair storage as float16: "
        f"{_estimate_storage_gb(len(patch_index), patch_size):.4f} GB"
    )
    print("First 5 patch rows:")
    _print_patch_examples(patch_index)
    print(f"Number of pair errors: {len(errors)}")


def _count_rows(
    df: pd.DataFrame,
    column: str,
    summary_type: str,
) -> list[dict[str, object]]:
    """Return value-count rows for one patch index column."""
    if df.empty or column not in df:
        return []
    counts = df[column].dropna().value_counts().sort_index()
    return [
        {"summary_type": summary_type, "value": str(value), "count": int(count)}
        for value, count in counts.items()
    ]


def _print_value_counts(
    df: pd.DataFrame,
    column: str,
    top_n: int | None = None,
) -> None:
    """Print value counts for one patch index column."""
    if df.empty or column not in df:
        print("  none")
        return
    counts = df[column].dropna().value_counts()
    if top_n is None:
        counts = counts.sort_index()
    else:
        counts = counts.head(top_n)
    if counts.empty:
        print("  none")
        return
    for value, count in counts.items():
        print(f"  {value}: {int(count)}")


def _print_patch_count_stats(patch_index: pd.DataFrame) -> None:
    """Print min/max/mean patch count per pair."""
    if patch_index.empty:
        print("Min patches per pair: 0")
        print("Max patches per pair: 0")
        print("Mean patches per pair: 0.0000")
        return
    patch_counts = patch_index.groupby("pair_id").size()
    print(f"Min patches per pair: {int(patch_counts.min())}")
    print(f"Max patches per pair: {int(patch_counts.max())}")
    print(f"Mean patches per pair: {patch_counts.mean():.4f}")


def _estimate_storage_gb(patch_count: int, patch_size: int) -> float:
    """Estimate Sentinel Siamese image-pair storage in decimal GB as float16."""
    one_sentinel_patch_bytes = 13 * int(patch_size) * int(patch_size) * 2
    one_siamese_sample_bytes = 2 * one_sentinel_patch_bytes
    return float(patch_count * one_siamese_sample_bytes / 1_000_000_000)


def _print_patch_examples(patch_index: pd.DataFrame) -> None:
    """Print first five patch rows."""
    if patch_index.empty:
        print("  none")
        return
    columns = ["patch_id", "pair_id", "split", "x", "y"]
    for _, row in patch_index[columns].head(5).iterrows():
        print(
            "  "
            f"{row['patch_id']} | pair={row['pair_id']} | "
            f"split={row['split']} | x={row['x']} | y={row['y']}"
        )


if __name__ == "__main__":
    main()
