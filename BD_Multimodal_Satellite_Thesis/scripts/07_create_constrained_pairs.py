"""Create the final constrained pair index for patch generation and training."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_all_configs  # noqa: E402
from src.pairing.constrained_pair_sampler import constrained_sample_all_splits  # noqa: E402
from src.pairing.pair_statistics import (  # noqa: E402
    save_pair_statistics,
    summarize_image_reuse,
)
from src.utils.file_utils import ensure_dir  # noqa: E402


SPLIT_TARGETS = {
    "train": 12000,
    "val": 2500,
    "test": 2500,
}


def main() -> None:
    """Run constrained pair sampling and save final pair indexes."""
    configs = load_all_configs()
    metadata_dir = configs.paths["metadata_dir"]
    output_dir = configs.paths["output_dir"]

    pair_dir = ensure_dir(metadata_dir / "pairs")
    reports_dir = ensure_dir(output_dir / "reports")

    candidate_path = pair_dir / "all_candidate_pairs.parquet"
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Missing candidate pairs: {candidate_path}. "
            "Run scripts/06_create_candidate_pairs.py first."
        )

    candidate_pairs = pd.read_parquet(candidate_path)
    selected_pairs = constrained_sample_all_splits(
        candidate_pairs,
        split_targets=SPLIT_TARGETS,
        max_image_use_global=3,
        fallback_max_image_use=5,
        random_seed=42,
    )

    selected_pairs.to_parquet(pair_dir / "constrained_pair_index.parquet", index=False)
    for split in ["train", "val", "test"]:
        split_pairs = selected_pairs[selected_pairs["split"] == split].copy()
        split_pairs.to_parquet(pair_dir / f"{split}_pairs.parquet", index=False)

    save_pair_statistics(selected_pairs, reports_dir)
    print_constrained_pair_report(candidate_pairs, selected_pairs)


def print_constrained_pair_report(
    candidate_pairs: pd.DataFrame,
    selected_pairs: pd.DataFrame,
) -> None:
    """Print the requested constrained-pair summary."""
    print(f"Candidate pair count before sampling: {len(candidate_pairs)}")
    print(f"Selected pair count after sampling: {len(selected_pairs)}")
    print("Selected count by split:")
    _print_value_counts(selected_pairs, "split")
    print("Selected count by pair_type:")
    _print_value_counts(selected_pairs, "pair_type")
    print("Selected count by time_gap_group:")
    _print_value_counts(selected_pairs, "time_gap_group")
    print("Selected count by time_gap_years:")
    _print_value_counts(selected_pairs, "time_gap_years")
    print(f"Same-season count: {_true_count(selected_pairs, 'same_season')}")
    print(f"Cross-season count: {_true_count(selected_pairs, 'cross_season')}")

    reuse_summary = summarize_image_reuse(selected_pairs)
    max_reuse = _metric_value(reuse_summary, "max_image_reuse")
    mean_reuse = _metric_value(reuse_summary, "mean_image_reuse")
    print(f"Max image reuse count: {max_reuse}")
    print(f"Mean image reuse count: {mean_reuse:.4f}")
    print(
        "Fallback max_image_use was used: "
        f"{bool(selected_pairs.get('fallback_max_image_use_used', pd.Series(dtype=bool)).any())}"
    )
    print("First 5 selected pairs:")
    _print_pair_examples(selected_pairs)


def _print_value_counts(df: pd.DataFrame, column: str) -> None:
    """Print value counts for a selected-pair column."""
    if df.empty or column not in df:
        print("  none")
        return

    counts = df[column].dropna().value_counts().sort_index()
    if counts.empty:
        print("  none")
        return

    for value, count in counts.items():
        print(f"  {value}: {int(count)}")


def _true_count(df: pd.DataFrame, column: str) -> int:
    """Count True values in a boolean-like column."""
    if df.empty or column not in df:
        return 0
    return int((df[column] == True).sum())  # noqa: E712


def _metric_value(summary: pd.DataFrame, value: str) -> float:
    """Read one numeric metric from an image reuse summary table."""
    match = summary[(summary["summary_type"] == "metric") & (summary["value"] == value)]
    if match.empty:
        return 0.0
    return float(match["count"].iloc[0])


def _print_pair_examples(selected_pairs: pd.DataFrame) -> None:
    """Print the first five selected pair examples."""
    if selected_pairs.empty:
        print("  none")
        return

    columns = ["pair_id", "split", "image_id_t1", "image_id_t2", "pair_type"]
    for _, row in selected_pairs[columns].head(5).iterrows():
        print(
            "  "
            f"{row['pair_id']} | {row['split']} | "
            f"{row['image_id_t1']} -> {row['image_id_t2']} | "
            f"{row['pair_type']}"
        )


if __name__ == "__main__":
    main()
