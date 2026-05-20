"""Filter constrained pairs to rows with complete tabular features."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_all_configs  # noqa: E402
from src.utils.file_utils import ensure_dir  # noqa: E402


def main() -> None:
    """Create tabular-complete pair and feature files without changing originals."""
    configs = load_all_configs()
    paths = configs.paths

    pair_dir = ensure_dir(paths["metadata_dir"] / "pairs")
    processed_dir = ensure_dir(paths["local_project_root"] / "data" / "tabular" / "processed")
    reports_dir = ensure_dir(paths["output_dir"] / "reports")

    pairs = pd.read_parquet(pair_dir / "constrained_pair_index.parquet")
    scaled_features = pd.read_parquet(
        processed_dir / "pair_tabular_features_scaled.parquet"
    )
    raw_features = pd.read_parquet(processed_dir / "pair_tabular_features.parquet")

    _validate_inputs(pairs, scaled_features, raw_features)

    complete_pair_ids = set(
        scaled_features.loc[
            scaled_features["tabular_any_missing"] == False,  # noqa: E712
            "pair_id",
        ]
    )
    complete_pairs = pairs[pairs["pair_id"].isin(complete_pair_ids)].copy()
    complete_scaled = scaled_features[
        scaled_features["pair_id"].isin(complete_pair_ids)
    ].copy()
    complete_raw = raw_features[raw_features["pair_id"].isin(complete_pair_ids)].copy()
    removed_pairs = pairs[~pairs["pair_id"].isin(complete_pair_ids)].copy()

    complete_pairs.to_parquet(
        pair_dir / "constrained_pair_index_tabular_complete.parquet",
        index=False,
    )
    for split in ["train", "val", "test"]:
        complete_pairs[complete_pairs["split"] == split].to_parquet(
            pair_dir / f"{split}_pairs_tabular_complete.parquet",
            index=False,
        )

    complete_scaled.to_parquet(
        processed_dir / "pair_tabular_features_scaled_tabular_complete.parquet",
        index=False,
    )
    complete_raw.to_parquet(
        processed_dir / "pair_tabular_features_tabular_complete.parquet",
        index=False,
    )

    summary = build_summary(pairs, complete_pairs, removed_pairs)
    summary.to_csv(
        reports_dir / "tabular_complete_pair_filter_summary.csv",
        index=False,
        encoding="utf-8",
    )

    print_filter_report(pairs, complete_pairs, removed_pairs)


def build_summary(
    original_pairs: pd.DataFrame,
    kept_pairs: pd.DataFrame,
    removed_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Build a CSV-friendly summary for tabular-complete filtering."""
    rows = [
        {"summary_type": "metric", "value": "original_pair_count", "count": len(original_pairs)},
        {"summary_type": "metric", "value": "kept_pair_count", "count": len(kept_pairs)},
        {"summary_type": "metric", "value": "removed_pair_count", "count": len(removed_pairs)},
    ]
    rows.extend(_count_rows(kept_pairs, "split", "kept_by_split"))
    rows.extend(_count_rows(removed_pairs, "split", "removed_by_split"))
    rows.extend(_count_rows(kept_pairs, "pair_type", "kept_by_pair_type"))
    rows.extend(_count_rows(removed_pairs, "pair_type", "removed_by_pair_type"))
    rows.extend(_count_rows(removed_pairs, "district", "removed_by_district"))
    return pd.DataFrame(rows, columns=["summary_type", "value", "count"])


def print_filter_report(
    original_pairs: pd.DataFrame,
    kept_pairs: pd.DataFrame,
    removed_pairs: pd.DataFrame,
) -> None:
    """Print the requested filter summary to the console."""
    print(f"Original pair count: {len(original_pairs)}")
    print(f"Kept pair count: {len(kept_pairs)}")
    print(f"Removed pair count: {len(removed_pairs)}")
    print("Kept count by split:")
    _print_counts(kept_pairs, "split")
    print("Removed count by split:")
    _print_counts(removed_pairs, "split")
    print("Kept count by pair_type:")
    _print_counts(kept_pairs, "pair_type")
    print("Removed count by pair_type:")
    _print_counts(removed_pairs, "pair_type")
    removed_districts = sorted(removed_pairs["district"].dropna().unique().tolist())
    print(f"Removed districts: {removed_districts}")


def _validate_inputs(
    pairs: pd.DataFrame,
    scaled_features: pd.DataFrame,
    raw_features: pd.DataFrame,
) -> None:
    """Validate required columns before filtering."""
    for name, df in [
        ("constrained pairs", pairs),
        ("scaled pair tabular features", scaled_features),
        ("raw pair tabular features", raw_features),
    ]:
        if "pair_id" not in df.columns:
            raise ValueError(f"{name} is missing required column: pair_id")

    if "tabular_any_missing" not in scaled_features.columns:
        raise ValueError(
            "scaled pair tabular features are missing required column: "
            "tabular_any_missing"
        )


def _count_rows(
    df: pd.DataFrame,
    column: str,
    summary_type: str,
) -> list[dict[str, object]]:
    """Return count rows for a DataFrame column."""
    if df.empty or column not in df:
        return []
    counts = df[column].dropna().value_counts().sort_index()
    return [
        {"summary_type": summary_type, "value": str(value), "count": int(count)}
        for value, count in counts.items()
    ]


def _print_counts(df: pd.DataFrame, column: str) -> None:
    """Print count rows for a DataFrame column."""
    rows = _count_rows(df, column, column)
    if not rows:
        print("  none")
        return
    for row in rows:
        print(f"  {row['value']}: {row['count']}")


if __name__ == "__main__":
    main()
