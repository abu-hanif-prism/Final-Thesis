"""Create pair-level tabular features for constrained temporal pairs."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_all_configs  # noqa: E402
from src.tabular.create_pair_features import (  # noqa: E402
    MISSING_FLAG_COLUMNS,
    create_missing_tabular_report,
    create_pair_tabular_features,
    identify_base_tabular_features,
    impute_and_scale_pair_features,
    normalize_join_key_district,
    prepare_tabular_for_join,
    save_json,
)
from src.utils.file_utils import ensure_dir  # noqa: E402


def main() -> None:
    """Load constrained pairs and seasonal tabular data, then create pair features."""
    configs = load_all_configs()
    paths = configs.paths

    pair_dir = ensure_dir(paths["metadata_dir"] / "pairs")
    processed_dir = ensure_dir(paths["local_project_root"] / "data" / "tabular" / "processed")
    reports_dir = ensure_dir(paths["local_project_root"] / "data" / "tabular" / "reports")

    pair_path = pair_dir / "constrained_pair_index.parquet"
    tabular_path = processed_dir / "district_seasonal_features.parquet"
    if not pair_path.exists():
        raise FileNotFoundError(
            f"Missing constrained pair index: {pair_path}. "
            "Run scripts/07_create_constrained_pairs.py first."
        )
    if not tabular_path.exists():
        raise FileNotFoundError(
            f"Missing seasonal tabular features: {tabular_path}. "
            "Run scripts/04_process_tabular.py first."
        )

    pairs = pd.read_parquet(pair_path)
    tabular = pd.read_parquet(tabular_path)
    prepared_tabular = prepare_tabular_for_join(tabular)
    base_features = identify_base_tabular_features(prepared_tabular)

    pair_features = create_pair_tabular_features(pairs, tabular)
    scaled_features, imputer_values, scaler_stats, column_info = (
        impute_and_scale_pair_features(pair_features, split_column="split")
    )

    pair_features.to_parquet(
        processed_dir / "pair_tabular_features.parquet",
        index=False,
    )
    scaled_features.to_parquet(
        processed_dir / "pair_tabular_features_scaled.parquet",
        index=False,
    )
    save_json(
        column_info,
        processed_dir / "pair_tabular_feature_columns.json",
    )
    save_json(
        imputer_values,
        processed_dir / "tabular_imputer_values.json",
    )
    save_json(
        scaler_stats,
        processed_dir / "tabular_scaler_stats.json",
    )

    missing_report = create_missing_tabular_report(pairs, tabular, pair_features)
    missing_report.to_csv(
        reports_dir / "pair_tabular_missing_report.csv",
        index=False,
        encoding="utf-8",
    )
    build_feature_summary(pair_features, scaled_features, base_features, column_info).to_csv(
        reports_dir / "pair_tabular_feature_summary.csv",
        index=False,
        encoding="utf-8",
    )
    missing_districts = build_missing_district_report(pairs, tabular)
    missing_districts.to_csv(
        reports_dir / "missing_tabular_districts.csv",
        index=False,
        encoding="utf-8",
    )

    print_pair_tabular_report(
        pairs,
        pair_features,
        scaled_features,
        base_features,
        column_info,
        missing_districts,
    )


def build_feature_summary(
    pair_features: pd.DataFrame,
    scaled_features: pd.DataFrame,
    base_features: list[str],
    column_info: dict[str, list[str]],
) -> pd.DataFrame:
    """Build a compact pair-tabular feature summary report."""
    rows = [
        {"metric": "pair_feature_rows", "value": len(pair_features)},
        {"metric": "scaled_feature_rows", "value": len(scaled_features)},
        {"metric": "base_tabular_feature_count", "value": len(base_features)},
        {
            "metric": "generated_raw_pair_feature_count",
            "value": len(column_info["raw_feature_columns"]),
        },
        {
            "metric": "scaled_feature_column_count",
            "value": len(column_info["scaled_feature_columns"]),
        },
        {
            "metric": "missing_flag_column_count",
            "value": len(column_info["missing_flag_columns"]),
        },
    ]
    for split, count in pair_features["split"].value_counts().sort_index().items():
        rows.append({"metric": f"rows_split_{split}", "value": int(count)})
    return pd.DataFrame(rows)


def build_missing_district_report(
    pairs: pd.DataFrame,
    tabular: pd.DataFrame,
) -> pd.DataFrame:
    """Return pair districts that are not present in the tabular district table."""
    pair_districts = (
        pairs[["district"]]
        .drop_duplicates()
        .assign(district_join_key=lambda df: df["district"].map(normalize_join_key_district))
    )
    tabular_district_keys = set(
        tabular["district"].map(normalize_join_key_district).dropna().unique()
    )
    missing = pair_districts[
        ~pair_districts["district_join_key"].isin(tabular_district_keys)
    ].copy()
    if missing.empty:
        return pd.DataFrame(columns=["district", "district_join_key", "pair_count"])

    counts = pairs["district"].value_counts()
    missing["pair_count"] = missing["district"].map(counts).astype(int)
    return missing.sort_values("district").reset_index(drop=True)


def print_pair_tabular_report(
    pairs: pd.DataFrame,
    pair_features: pd.DataFrame,
    scaled_features: pd.DataFrame,
    base_features: list[str],
    column_info: dict[str, list[str]],
    missing_districts: pd.DataFrame,
) -> None:
    """Print the requested pair-tabular feature summary."""
    print(f"Number of input pairs: {len(pairs)}")
    print(f"Number of output pair feature rows: {len(pair_features)}")
    print(f"Number of base tabular features: {len(base_features)}")
    print(
        "Number of generated raw pair features: "
        f"{len(column_info['raw_feature_columns'])}"
    )
    print(f"Number of scaled feature columns: {len(column_info['scaled_feature_columns'])}")
    print(f"Number of missing tabular districts: {len(missing_districts)}")
    print(
        "Missing tabular districts list: "
        f"{missing_districts['district'].tolist() if not missing_districts.empty else []}"
    )
    print(f"T1 missing count: {_true_count(pair_features, 'tabular_t1_missing')}")
    print(f"T2 missing count: {_true_count(pair_features, 'tabular_t2_missing')}")
    print(f"Any missing count: {_true_count(pair_features, 'tabular_any_missing')}")
    print("Missing count by split:")
    _print_missing_counts(pair_features, "split")
    print("Missing count by pair_type:")
    _print_missing_counts(pair_features, "pair_type")
    print("Train/val/test row counts:")
    for split, count in scaled_features["split"].value_counts().sort_index().items():
        print(f"  {split}: {int(count)}")


def _true_count(df: pd.DataFrame, column: str) -> int:
    """Count True values in a boolean-like column."""
    if df.empty or column not in df:
        return 0
    return int((df[column] == True).sum())  # noqa: E712


def _print_missing_counts(df: pd.DataFrame, column: str) -> None:
    """Print tabular_any_missing counts grouped by a column."""
    if df.empty or column not in df or "tabular_any_missing" not in df:
        print("  none")
        return

    missing = df[df["tabular_any_missing"] == True]  # noqa: E712
    if missing.empty:
        print("  none")
        return

    counts = missing[column].dropna().value_counts().sort_index()
    for value, count in counts.items():
        print(f"  {value}: {int(count)}")


if __name__ == "__main__":
    main()
