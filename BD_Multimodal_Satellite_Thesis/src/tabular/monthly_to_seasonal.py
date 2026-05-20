"""Convert district-level monthly tabular data into seasonal features."""

import pandas as pd

from src.tabular.clean_tabular import normalize_district_values, normalize_month_values


KEY_COLUMNS = {"district", "year", "month", "season", "season_year"}
COUNT_COLUMN_KEYWORDS = (
    "count",
    "cyclone",
    "flood",
    "drought",
    "event",
    "disaster",
    "storm",
)
SEASON_ORDER = ["Winter", "PreMonsoon", "Monsoon", "PostMonsoon"]


def assign_season_year_month(year: int, month: int) -> tuple[int, str]:
    """Assign a calendar year/month to a project season year and season."""
    year_int = int(year)
    month_int = int(month)

    if month_int == 12:
        return year_int + 1, "Winter"
    if month_int in {1, 2}:
        return year_int, "Winter"
    if month_int in {3, 4, 5}:
        return year_int, "PreMonsoon"
    if month_int in {6, 7, 8, 9}:
        return year_int, "Monsoon"
    if month_int in {10, 11}:
        return year_int, "PostMonsoon"

    raise ValueError(f"Invalid month {month}. Expected 1-12.")


def identify_numeric_feature_columns(
    df: pd.DataFrame,
    exclude_columns: set[str] | list[str] | tuple[str, ...],
) -> list[str]:
    """Return numeric feature columns excluding key columns."""
    excluded = set(exclude_columns)
    numeric_columns = df.select_dtypes(include="number").columns
    return [column for column in numeric_columns if column not in excluded]


def identify_count_columns(columns: list[str] | pd.Index) -> list[str]:
    """Identify numeric columns that should use sum aggregation."""
    return [
        column
        for column in columns
        if any(keyword in str(column).lower() for keyword in COUNT_COLUMN_KEYWORDS)
    ]


def monthly_to_seasonal(
    df: pd.DataFrame,
    district_col: str,
    year_col: str,
    month_col: str,
) -> pd.DataFrame:
    """Aggregate monthly district records into seasonal district-level features."""
    working = df.copy()
    working["district"] = normalize_district_values(working[district_col])
    working["month"] = normalize_month_values(working[month_col])
    working["year"] = pd.to_numeric(working[year_col], errors="raise").astype(int)

    assigned = working.apply(
        lambda row: assign_season_year_month(row["year"], row["month"]),
        axis=1,
        result_type="expand",
    )
    working["season_year"] = assigned[0].astype(int)
    working["season"] = assigned[1]

    exclude_columns = KEY_COLUMNS | {district_col, year_col, month_col}
    numeric_columns = identify_numeric_feature_columns(working, exclude_columns)
    count_columns = identify_count_columns(numeric_columns)
    mean_columns = [column for column in numeric_columns if column not in count_columns]

    aggregations = {
        **{column: "mean" for column in mean_columns},
        **{column: "sum" for column in count_columns},
    }
    grouped = (
        working.groupby(["district", "season_year", "season"], dropna=False)
        .agg(aggregations)
        .reset_index()
    )
    grouped = grouped.rename(columns={"season_year": "year"})
    grouped["year"] = grouped["year"].astype(int)
    grouped["season"] = pd.Categorical(
        grouped["season"],
        categories=SEASON_ORDER,
        ordered=True,
    )

    output_columns = ["district", "year", "season", *mean_columns, *count_columns]
    seasonal = grouped[output_columns].sort_values(
        ["district", "year", "season"]
    )
    seasonal["season"] = seasonal["season"].astype(str)
    return seasonal.reset_index(drop=True)
