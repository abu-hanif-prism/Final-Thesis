"""Feature schema helpers for seasonal tabular features."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.tabular.monthly_to_seasonal import identify_count_columns
from src.utils.file_utils import ensure_dir


KEY_COLUMNS = ["district", "year", "season"]


def build_tabular_feature_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Build a schema dictionary for seasonal tabular feature data."""
    numeric_feature_columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in KEY_COLUMNS
    ]
    count_columns = identify_count_columns(numeric_feature_columns)
    mean_columns = [
        column for column in numeric_feature_columns if column not in count_columns
    ]

    return {
        "key_columns": KEY_COLUMNS,
        "numeric_feature_columns": numeric_feature_columns,
        "count_sum_feature_columns": count_columns,
        "mean_feature_columns": mean_columns,
        "missing_value_percentages": {
            column: float(df[column].isna().mean() * 100) for column in df.columns
        },
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }


def save_feature_schema(schema: dict[str, Any], path: str | Path) -> Path:
    """Save a feature schema as JSON or a CSV-style column dictionary."""
    output_path = Path(path)
    ensure_dir(output_path.parent)

    if output_path.suffix.lower() == ".json":
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(schema, file, indent=2, ensure_ascii=False)
    else:
        schema_to_column_dictionary(schema).to_csv(
            output_path,
            index=False,
            encoding="utf-8",
        )

    return output_path


def schema_to_column_dictionary(schema: dict[str, Any]) -> pd.DataFrame:
    """Convert a schema dictionary into one row per column."""
    key_columns = set(schema["key_columns"])
    count_columns = set(schema["count_sum_feature_columns"])
    mean_columns = set(schema["mean_feature_columns"])
    missing = schema["missing_value_percentages"]
    dtypes = schema["dtypes"]

    rows = []
    for column, dtype in dtypes.items():
        if column in key_columns:
            role = "key"
            aggregation = None
        elif column in count_columns:
            role = "numeric_feature"
            aggregation = "sum"
        elif column in mean_columns:
            role = "numeric_feature"
            aggregation = "mean"
        else:
            role = "non_numeric"
            aggregation = None

        rows.append(
            {
                "column": column,
                "role": role,
                "aggregation": aggregation,
                "dtype": dtype,
                "missing_percent": missing.get(column),
            }
        )

    return pd.DataFrame(rows)
