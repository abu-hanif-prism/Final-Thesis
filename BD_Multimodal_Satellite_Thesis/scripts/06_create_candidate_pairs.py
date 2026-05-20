"""Create all forward-time candidate image pairs for Siamese modeling."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_all_configs  # noqa: E402
from src.pairing.create_candidate_pairs import generate_all_candidate_pairs  # noqa: E402
from src.utils.file_utils import ensure_dir  # noqa: E402


def main() -> None:
    """Load image splits, generate candidate pairs, and save reports."""
    configs = load_all_configs()
    metadata_dir = configs.paths["metadata_dir"]
    output_dir = configs.paths["output_dir"]
    season_order = configs.data["season_order"]

    split_dir = ensure_dir(metadata_dir / "splits")
    pair_dir = ensure_dir(metadata_dir / "pairs")
    reports_dir = ensure_dir(output_dir / "reports")

    image_split_path = split_dir / "image_split.parquet"
    if not image_split_path.exists():
        raise FileNotFoundError(
            f"Missing image split file: {image_split_path}. "
            "Run scripts/05_create_splits.py first."
        )

    image_split = pd.read_parquet(image_split_path)
    image_split["year"] = pd.to_numeric(image_split["year"], errors="raise").astype(int)

    candidate_pairs = generate_all_candidate_pairs(image_split, season_order)
    candidate_pairs.to_parquet(
        pair_dir / "all_candidate_pairs.parquet",
        index=False,
    )

    summary = build_candidate_pair_summary(image_split, candidate_pairs)
    summary.to_csv(
        reports_dir / "all_candidate_pair_summary.csv",
        index=False,
        encoding="utf-8",
    )

    print_candidate_pair_report(image_split, candidate_pairs)


def build_candidate_pair_summary(
    image_split: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Build a long-form candidate-pair summary table."""
    rows = [
        {
            "summary_type": "metric",
            "value": "total_input_images",
            "count": int(len(image_split)),
        },
        {
            "summary_type": "metric",
            "value": "total_candidate_pairs",
            "count": int(len(candidate_pairs)),
        },
        {
            "summary_type": "metric",
            "value": "same_season_count",
            "count": _true_count(candidate_pairs, "same_season"),
        },
        {
            "summary_type": "metric",
            "value": "cross_season_count",
            "count": _true_count(candidate_pairs, "cross_season"),
        },
    ]
    rows.extend(_count_rows(candidate_pairs, "split", "pairs_by_split"))
    rows.extend(_count_rows(candidate_pairs, "district", "pairs_by_district"))
    rows.extend(_count_rows(candidate_pairs, "pair_type", "pairs_by_pair_type"))
    rows.extend(_count_rows(candidate_pairs, "time_gap_years", "pairs_by_time_gap_years"))
    return pd.DataFrame(rows, columns=["summary_type", "value", "count"])


def print_candidate_pair_report(
    image_split: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
) -> None:
    """Print the requested candidate-pair summary to the console."""
    print(f"Total input images: {len(image_split)}")
    print(f"Total candidate pairs: {len(candidate_pairs)}")
    print("Candidate pairs by split:")
    _print_value_counts(candidate_pairs, "split")
    print("Candidate pairs by district top 20:")
    _print_value_counts(candidate_pairs, "district", top_n=20)
    print("Candidate pairs by pair_type:")
    _print_value_counts(candidate_pairs, "pair_type")
    print("Candidate pairs by time_gap_years:")
    _print_value_counts(candidate_pairs, "time_gap_years")
    print(f"Same-season count: {_true_count(candidate_pairs, 'same_season')}")
    print(f"Cross-season count: {_true_count(candidate_pairs, 'cross_season')}")
    _print_year_ranges(candidate_pairs)
    print("First 5 candidate pair examples:")
    _print_pair_examples(candidate_pairs)


def _count_rows(
    df: pd.DataFrame,
    column: str,
    summary_type: str,
) -> list[dict[str, object]]:
    """Return value-count rows for a candidate-pair column."""
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


def _true_count(df: pd.DataFrame, column: str) -> int:
    """Count True values in a boolean-like column."""
    if df.empty or column not in df:
        return 0
    return int((df[column] == True).sum())  # noqa: E712


def _print_value_counts(
    df: pd.DataFrame,
    column: str,
    top_n: int | None = None,
) -> None:
    """Print value counts for a candidate-pair column."""
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


def _print_year_ranges(candidate_pairs: pd.DataFrame) -> None:
    """Print min/max years for t1 and t2."""
    if candidate_pairs.empty:
        print("Min year_t1/year_t2: none")
        print("Max year_t1/year_t2: none")
        return

    print(
        "Min year_t1/year_t2: "
        f"{int(candidate_pairs['year_t1'].min())}/"
        f"{int(candidate_pairs['year_t2'].min())}"
    )
    print(
        "Max year_t1/year_t2: "
        f"{int(candidate_pairs['year_t1'].max())}/"
        f"{int(candidate_pairs['year_t2'].max())}"
    )


def _print_pair_examples(candidate_pairs: pd.DataFrame) -> None:
    """Print the first five candidate pair examples."""
    if candidate_pairs.empty:
        print("  none")
        return

    example_columns = [
        "pair_id",
        "split",
        "image_id_t1",
        "image_id_t2",
        "pair_type",
    ]
    for _, row in candidate_pairs[example_columns].head(5).iterrows():
        print(
            "  "
            f"{row['pair_id']} | {row['split']} | "
            f"{row['image_id_t1']} -> {row['image_id_t2']} | "
            f"{row['pair_type']}"
        )


if __name__ == "__main__":
    main()
