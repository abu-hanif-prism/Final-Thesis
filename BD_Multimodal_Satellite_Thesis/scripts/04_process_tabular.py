"""Clean monthly tabular data and aggregate it to seasonal district features."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_all_configs  # noqa: E402
from src.tabular.clean_tabular import (  # noqa: E402
    clean_tabular_dataframe,
    detect_key_columns,
    normalize_district_values,
    normalize_month_values,
)
from src.tabular.feature_schema import (  # noqa: E402
    build_tabular_feature_schema,
    save_feature_schema,
    schema_to_column_dictionary,
)
from src.tabular.monthly_to_seasonal import (  # noqa: E402
    KEY_COLUMNS,
    identify_count_columns,
    identify_numeric_feature_columns,
    monthly_to_seasonal,
)
from src.utils.file_utils import ensure_dir  # noqa: E402


def main() -> None:
    """Run local tabular CSV cleaning and seasonal aggregation."""
    configs = load_all_configs()
    paths = configs.paths

    raw_path = paths["tabular_raw_path"]
    processed_dir = ensure_dir(paths["local_project_root"] / "data" / "tabular" / "processed")
    reports_dir = ensure_dir(paths["local_project_root"] / "data" / "tabular" / "reports")

    raw_df = pd.read_csv(raw_path)
    cleaned_df = clean_tabular_dataframe(raw_df)
    key_columns = detect_key_columns(cleaned_df)

    cleaned_output = prepare_clean_monthly_output(cleaned_df, key_columns)
    seasonal_df = monthly_to_seasonal(
        cleaned_output,
        key_columns["district"],
        key_columns["year"],
        key_columns["month"],
    )

    schema = build_tabular_feature_schema(seasonal_df)
    feature_summary = build_feature_summary(schema)
    missing_values = build_missing_values_report(seasonal_df)

    cleaned_output.to_parquet(
        processed_dir / "district_monthly_clean.parquet",
        index=False,
    )
    seasonal_df.to_parquet(
        processed_dir / "district_seasonal_features.parquet",
        index=False,
    )
    missing_values.to_csv(
        reports_dir / "tabular_missing_values.csv",
        index=False,
        encoding="utf-8",
    )
    feature_summary.to_csv(
        reports_dir / "tabular_feature_summary.csv",
        index=False,
        encoding="utf-8",
    )
    save_feature_schema(
        schema,
        reports_dir / "tabular_column_dictionary.csv",
    )

    print_tabular_report(
        raw_df,
        cleaned_output,
        seasonal_df,
        key_columns,
        schema,
        missing_values,
    )


def prepare_clean_monthly_output(
    cleaned_df: pd.DataFrame,
    key_columns: dict[str, str],
) -> pd.DataFrame:
    """Normalize detected district and month columns in the cleaned monthly table."""
    output = cleaned_df.copy()
    district_col = key_columns["district"]
    year_col = key_columns["year"]
    month_col = key_columns["month"]

    output[district_col] = normalize_district_values(output[district_col])
    output[month_col] = normalize_month_values(output[month_col]).astype(int)
    output[year_col] = pd.to_numeric(output[year_col], errors="raise").astype(int)
    return output


def build_missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """Create a missing value report for each seasonal output column."""
    row_count = len(df)
    rows = []
    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        missing_percent = 0.0 if row_count == 0 else missing_count / row_count * 100
        rows.append(
            {
                "column": column,
                "missing_count": missing_count,
                "missing_percent": missing_percent,
                "dtype": str(df[column].dtype),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["missing_percent", "missing_count", "column"],
        ascending=[False, False, True],
    )


def build_feature_summary(schema: dict[str, object]) -> pd.DataFrame:
    """Create a compact feature summary CSV from the schema."""
    column_dictionary = schema_to_column_dictionary(schema)
    summary_rows = [
        {
            "summary_type": "metric",
            "value": "numeric_feature_count",
            "count": len(schema["numeric_feature_columns"]),
        },
        {
            "summary_type": "metric",
            "value": "count_sum_feature_count",
            "count": len(schema["count_sum_feature_columns"]),
        },
        {
            "summary_type": "metric",
            "value": "mean_feature_count",
            "count": len(schema["mean_feature_columns"]),
        },
    ]

    aggregation_counts = (
        column_dictionary["aggregation"]
        .fillna("none")
        .value_counts()
        .sort_index()
    )
    summary_rows.extend(
        {
            "summary_type": "aggregation",
            "value": aggregation,
            "count": int(count),
        }
        for aggregation, count in aggregation_counts.items()
    )

    return pd.DataFrame(summary_rows)


def print_tabular_report(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    seasonal_df: pd.DataFrame,
    key_columns: dict[str, str],
    schema: dict[str, object],
    missing_values: pd.DataFrame,
) -> None:
    """Print the requested tabular processing summary to the console."""
    print(f"Original shape: {raw_df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Seasonal shape: {seasonal_df.shape}")
    print(f"Detected district column: {key_columns['district']}")
    print(f"Detected year column: {key_columns['year']}")
    print(f"Detected month column: {key_columns['month']}")
    print(f"Numeric feature count: {len(schema['numeric_feature_columns'])}")
    print(f"Count/sum feature count: {len(schema['count_sum_feature_columns'])}")
    print(f"Mean feature count: {len(schema['mean_feature_columns'])}")
    print("Missing value summary top 20:")
    _print_missing_top_20(missing_values)

    if seasonal_df.empty:
        print("Year range in seasonal output: none")
        print("Seasons found: []")
        print("District count: 0")
        return

    print(
        "Year range in seasonal output: "
        f"{int(seasonal_df['year'].min())}-{int(seasonal_df['year'].max())}"
    )
    print(f"Seasons found: {sorted(seasonal_df['season'].dropna().unique().tolist())}")
    print(f"District count: {int(seasonal_df['district'].nunique())}")


def _print_missing_top_20(missing_values: pd.DataFrame) -> None:
    """Print top missing-value rows."""
    if missing_values.empty:
        print("  none")
        return

    for _, row in missing_values.head(20).iterrows():
        print(
            "  "
            f"{row['column']}: {int(row['missing_count'])} "
            f"({row['missing_percent']:.2f}%)"
        )


if __name__ == "__main__":
    main()
