"""Utilities for cleaning local district-level monthly tabular data."""

import re
from typing import Iterable

import pandas as pd


DISTRICT_COLUMN_CANDIDATES = {
    "district",
    "district_name",
    "dist_name",
    "name",
    "adm2",
    "admin2",
}
YEAR_COLUMN_CANDIDATES = {"year", "yr"}
MONTH_COLUMN_CANDIDATES = {"month", "mon", "month_id", "month_number"}

MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def standardize_column_name(col: object) -> str:
    """Standardize a column name to lowercase snake_case."""
    standardized = str(col).strip().lower()
    standardized = standardized.replace("-", "_")
    standardized = re.sub(r"\s+", "_", standardized)
    standardized = re.sub(r"[^a-z0-9_]+", "_", standardized)
    standardized = re.sub(r"_+", "_", standardized)
    return standardized.strip("_")


def clean_tabular_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names, remove empty rows, and drop exact duplicates."""
    cleaned = df.copy()
    cleaned.columns = [standardize_column_name(column) for column in cleaned.columns]
    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.drop_duplicates()
    return cleaned.reset_index(drop=True)


def detect_key_columns(df: pd.DataFrame) -> dict[str, str]:
    """Detect district, year, and month columns from standardized names."""
    return {
        "district": _detect_one_column(df, DISTRICT_COLUMN_CANDIDATES, "district"),
        "year": _detect_one_column(df, YEAR_COLUMN_CANDIDATES, "year"),
        "month": _detect_one_column(df, MONTH_COLUMN_CANDIDATES, "month"),
    }


def normalize_month_values(series: pd.Series) -> pd.Series:
    """Normalize numeric or named month values to integer month numbers 1-12."""
    normalized = series.map(_normalize_single_month)
    failed_mask = normalized.isna() & series.notna()
    if failed_mask.any():
        examples = series[failed_mask].drop_duplicates().head(10).tolist()
        raise ValueError(f"Failed to parse month values. Examples: {examples}")

    return normalized.astype("Int64")


def normalize_district_values(series: pd.Series) -> pd.Series:
    """Normalize district names to the satellite-style readable underscore format."""
    return series.map(_normalize_single_district)


def _detect_one_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    label: str,
) -> str:
    """Detect one required key column and raise a clear error on failure."""
    matches = [column for column in df.columns if column in candidates]
    if len(matches) == 1:
        return matches[0]

    print(f"Available columns: {list(df.columns)}")
    if not matches:
        raise ValueError(
            f"Could not detect {label} column. Expected one of: {sorted(candidates)}."
        )

    raise ValueError(
        f"Ambiguous {label} column detection. Matched columns: {matches}."
    )


def _normalize_single_month(value: object) -> int | None:
    """Normalize one month value to an integer, returning None for missing values."""
    if pd.isna(value):
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if re.fullmatch(r"\d+(\.0+)?", text):
            month = int(float(text))
        else:
            month = MONTH_NAME_TO_NUMBER.get(text.lower())
    else:
        try:
            month = int(value)
        except (TypeError, ValueError):
            return None

    if month is None or month < 1 or month > 12:
        return None
    return month


def _normalize_single_district(value: object) -> str | None:
    """Normalize one district value."""
    if pd.isna(value):
        return None

    text = str(value).strip().replace("_", " ")
    text = text.replace("'", "")
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None

    text = text.title()
    text = text.replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    return text.strip("_")
