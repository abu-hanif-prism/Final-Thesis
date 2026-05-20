"""Sample spatially diverse patch coordinates before label calculation."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_all_configs  # noqa: E402
from src.patches.sample_patch_index import (  # noqa: E402
    sample_patch_index,
    summarize_patch_sampling,
)
from src.utils.file_utils import ensure_dir  # noqa: E402


MAX_PATCHES_PER_PAIR = 50
RANDOM_SEED = 42


def main() -> None:
    """Sample patch coordinates and save split-specific sampled indexes."""
    configs = load_all_configs()
    metadata_dir = configs.paths["metadata_dir"]
    output_dir = configs.paths["output_dir"]

    patch_dir = ensure_dir(metadata_dir / "patches")
    reports_dir = ensure_dir(output_dir / "reports")

    full_patch_path = patch_dir / "patch_index_all.parquet"
    if not full_patch_path.exists():
        raise FileNotFoundError(
            f"Missing full patch index: {full_patch_path}. "
            "Run scripts/09_create_patch_index.py first."
        )

    full_patch_df = pd.read_parquet(full_patch_path)
    sampled_patch_df = sample_patch_index(
        full_patch_df,
        max_patches_per_pair=MAX_PATCHES_PER_PAIR,
        random_seed=RANDOM_SEED,
    )

    sampled_patch_df.to_parquet(patch_dir / "patch_index_sampled.parquet", index=False)
    for split in ["train", "val", "test"]:
        sampled_patch_df[sampled_patch_df["split"] == split].to_parquet(
            patch_dir / f"patch_index_{split}_sampled.parquet",
            index=False,
        )

    summaries = summarize_patch_sampling(full_patch_df, sampled_patch_df)
    flatten_summary_tables(summaries).to_csv(
        reports_dir / "patch_sampling_summary.csv",
        index=False,
        encoding="utf-8",
    )
    summaries["per_pair"].to_csv(
        reports_dir / "per_pair_patch_sampling_stats.csv",
        index=False,
        encoding="utf-8",
    )

    print_sampling_report(full_patch_df, sampled_patch_df, summaries["per_pair"])


def flatten_summary_tables(summary_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flatten sampling summary tables into a single report CSV."""
    rows = []
    for summary_type, table in summary_tables.items():
        if summary_type == "per_pair" or table.empty:
            continue
        for _, row in table.iterrows():
            output = {"summary_type": summary_type}
            output.update(row.to_dict())
            rows.append(output)
    return pd.DataFrame(rows)


def print_sampling_report(
    full_patch_df: pd.DataFrame,
    sampled_patch_df: pd.DataFrame,
    per_pair_stats: pd.DataFrame,
) -> None:
    """Print requested patch sampling summary."""
    original_count = len(full_patch_df)
    sampled_count = len(sampled_patch_df)
    reduction_ratio = 0.0 if original_count == 0 else sampled_count / original_count
    sampled_per_pair = sampled_patch_df.groupby("pair_id").size()

    print(f"Original patch count: {original_count}")
    print(f"Sampled patch count: {sampled_count}")
    print(f"Reduction ratio: {reduction_ratio:.6f}")
    print("Sampled count by split:")
    _print_counts(sampled_patch_df, "split")
    print("Sampled count by pair_type:")
    _print_counts(sampled_patch_df, "pair_type")
    print("Sampled count by time_gap_group:")
    _print_counts(sampled_patch_df, "time_gap_group")
    print(f"Min sampled patches per pair: {int(sampled_per_pair.min())}")
    print(f"Max sampled patches per pair: {int(sampled_per_pair.max())}")
    print(f"Mean sampled patches per pair: {sampled_per_pair.mean():.4f}")
    print("Top 10 pairs by original patch count:")
    _print_pair_counts(per_pair_stats, "original_patch_count")
    print("Top 10 pairs by sampled patch count:")
    _print_pair_counts(
        per_pair_stats.sort_values("sampled_patch_count", ascending=False),
        "sampled_patch_count",
    )
    print("First 5 sampled patch rows:")
    _print_patch_examples(sampled_patch_df)


def _print_counts(df: pd.DataFrame, column: str) -> None:
    """Print value counts for one column."""
    if df.empty or column not in df:
        print("  none")
        return
    counts = df[column].dropna().value_counts().sort_index()
    for value, count in counts.items():
        print(f"  {value}: {int(count)}")


def _print_pair_counts(per_pair_stats: pd.DataFrame, count_column: str) -> None:
    """Print top ten pair count rows."""
    if per_pair_stats.empty or count_column not in per_pair_stats:
        print("  none")
        return
    for _, row in per_pair_stats.head(10).iterrows():
        print(f"  {row['pair_id']}: {int(row[count_column])}")


def _print_patch_examples(sampled_patch_df: pd.DataFrame) -> None:
    """Print first five sampled patch rows."""
    if sampled_patch_df.empty:
        print("  none")
        return
    columns = ["patch_id", "pair_id", "split", "x", "y"]
    for _, row in sampled_patch_df[columns].head(5).iterrows():
        print(
            "  "
            f"{row['patch_id']} | pair={row['pair_id']} | "
            f"split={row['split']} | x={row['x']} | y={row['y']}"
        )


if __name__ == "__main__":
    main()
