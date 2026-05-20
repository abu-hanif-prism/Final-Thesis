"""Create pair-level tabular features for constrained image pairs."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover - depends on local environment.
    StandardScaler = None


SEASON_ORDER = {
    "Winter": 0,
    "PreMonsoon": 1,
    "Monsoon": 2,
    "PostMonsoon": 3,
}
SEASON_ALIASES = {
    "winter": "Winter",
    "premonsoon": "PreMonsoon",
    "pre_monsoon": "PreMonsoon",
    "monsoon": "Monsoon",
    "postmonsoon": "PostMonsoon",
    "post_monsoon": "PostMonsoon",
}
COUNT_KEYWORDS = (
    "count",
    "cyclone",
    "flood",
    "drought",
    "event",
    "disaster",
    "storm",
)
PAIR_METADATA_COLUMNS = [
    "pair_id",
    "split",
    "district",
    "year_t1",
    "season_t1",
    "year_t2",
    "season_t2",
    "pair_type",
    "time_gap_group",
]
MISSING_FLAG_COLUMNS = [
    "tabular_t1_missing",
    "tabular_t2_missing",
    "tabular_any_missing",
    "tabular_district_missing",
]
JOIN_KEY_COLUMNS = ["district_join_key", "year", "season"]


def normalize_join_key_district(name: object) -> str | None:
    """Normalize district names for satellite-tabular joins."""
    if pd.isna(name):
        return None
    text = str(name).strip().lower()
    text = re.sub(r"[\s_]+", "", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text or None


def prepare_tabular_for_join(tabular_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare seasonal tabular records for pair joins."""
    required = {"district", "year", "season"}
    missing = sorted(required - set(tabular_df.columns))
    if missing:
        raise ValueError(f"Tabular data is missing required columns: {missing}")

    prepared = tabular_df.copy()
    prepared["district_original"] = prepared["district"]
    prepared["district_join_key"] = prepared["district"].map(normalize_join_key_district)
    prepared["year"] = pd.to_numeric(prepared["year"], errors="raise").astype(int)
    prepared["season"] = prepared["season"].map(_normalize_season_name)
    prepared["season_order"] = prepared["season"].map(SEASON_ORDER).astype(int)
    prepared["time_index"] = prepared["year"] * 4 + prepared["season_order"]
    return prepared


def prepare_pairs_for_join(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare constrained pair records for tabular joins."""
    required = {
        "pair_id",
        "split",
        "district",
        "year_t1",
        "season_t1",
        "year_t2",
        "season_t2",
        "pair_type",
        "time_gap_group",
    }
    missing = sorted(required - set(pair_df.columns))
    if missing:
        raise ValueError(f"Pair data is missing required columns: {missing}")

    prepared = pair_df.copy()
    prepared["district_join_key"] = prepared["district"].map(normalize_join_key_district)
    prepared["year_t1"] = pd.to_numeric(prepared["year_t1"], errors="raise").astype(int)
    prepared["year_t2"] = pd.to_numeric(prepared["year_t2"], errors="raise").astype(int)
    prepared["season_t1"] = prepared["season_t1"].map(_normalize_season_name)
    prepared["season_t2"] = prepared["season_t2"].map(_normalize_season_name)
    prepared["season_order_t1"] = prepared["season_t1"].map(SEASON_ORDER).astype(int)
    prepared["season_order_t2"] = prepared["season_t2"].map(SEASON_ORDER).astype(int)
    prepared["time_index_t1"] = prepared["year_t1"] * 4 + prepared["season_order_t1"]
    prepared["time_index_t2"] = prepared["year_t2"] * 4 + prepared["season_order_t2"]
    return prepared


def identify_base_tabular_features(tabular_df: pd.DataFrame) -> list[str]:
    """Return base numeric tabular features, excluding key columns."""
    excluded = {
        "district",
        "district_original",
        "district_join_key",
        "year",
        "season",
        "season_order",
        "time_index",
    }
    return [
        column
        for column in tabular_df.select_dtypes(include="number").columns
        if column not in excluded
    ]


def create_pair_tabular_features(
    pair_df: pd.DataFrame,
    tabular_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create endpoint, delta, ratio, between-window, and metadata features."""
    pairs = prepare_pairs_for_join(pair_df)
    tabular = prepare_tabular_for_join(tabular_df)
    base_features = identify_base_tabular_features(tabular)
    count_features = _identify_count_features(base_features)
    tabular_lookup = _build_tabular_lookup(tabular)
    district_groups = _build_district_groups(tabular)
    tabular_district_keys = set(tabular["district_join_key"].dropna().unique())

    rows = []
    for _, pair in pairs.iterrows():
        row = _metadata_row(pair)
        t1_key = (pair["district_join_key"], pair["year_t1"], pair["season_t1"])
        t2_key = (pair["district_join_key"], pair["year_t2"], pair["season_t2"])
        t1_values = tabular_lookup.get(t1_key)
        t2_values = tabular_lookup.get(t2_key)

        district_missing = pair["district_join_key"] not in tabular_district_keys
        t1_missing = t1_values is None
        t2_missing = t2_values is None
        row.update(
            {
                "tabular_t1_missing": bool(t1_missing),
                "tabular_t2_missing": bool(t2_missing),
                "tabular_any_missing": bool(t1_missing or t2_missing),
                "tabular_district_missing": bool(district_missing),
            }
        )

        for feature in base_features:
            value_t1 = np.nan if t1_values is None else t1_values.get(feature, np.nan)
            value_t2 = np.nan if t2_values is None else t2_values.get(feature, np.nan)
            row[f"{feature}_t1"] = value_t1
            row[f"{feature}_t2"] = value_t2
            row[f"{feature}_diff"] = value_t2 - value_t1
            row[f"{feature}_ratio"] = _safe_ratio(value_t2, value_t1)

        between = _between_records(
            district_groups,
            pair["district_join_key"],
            pair["time_index_t1"],
            pair["time_index_t2"],
        )
        for feature in base_features:
            row[f"{feature}_mean_between"] = (
                np.nan if between.empty else between[feature].mean()
            )
        for feature in count_features:
            row[f"{feature}_sum_between"] = (
                np.nan if between.empty else between[feature].sum()
            )

        row.update(_one_hot("season_t1", pair["season_t1"], list(SEASON_ORDER)))
        row.update(_one_hot("season_t2", pair["season_t2"], list(SEASON_ORDER)))
        row.update(
            _one_hot(
                "pair_type",
                pair["pair_type"],
                [
                    "same_season_multiyear",
                    "cross_season_sameyear",
                    "cross_season_multiyear",
                ],
            )
        )
        row.update(
            _one_hot(
                "time_gap_group",
                pair["time_gap_group"],
                ["same_year", "short", "medium", "long", "very_long", "other"],
            )
        )
        rows.append(row)

    return pd.DataFrame(rows)


def impute_and_scale_pair_features(
    pair_feature_df: pd.DataFrame,
    split_column: str = "split",
) -> tuple[pd.DataFrame, dict[str, float], dict[str, dict[str, float]], dict[str, list[str]]]:
    """Impute and scale numeric features using train split statistics only."""
    if split_column not in pair_feature_df.columns:
        raise ValueError(f"Missing split column: {split_column}")

    scaled = pair_feature_df.copy()
    metadata_columns = [column for column in PAIR_METADATA_COLUMNS if column in scaled]
    missing_flag_columns = [column for column in MISSING_FLAG_COLUMNS if column in scaled]
    target_excluded_columns = metadata_columns + missing_flag_columns + ["district_join_key"]
    numeric_columns = [
        column
        for column in scaled.select_dtypes(include="number").columns
        if column not in target_excluded_columns
    ]

    train_mask = scaled[split_column] == "train"
    if not train_mask.any():
        raise ValueError("Train split has no rows; cannot fit tabular imputer/scaler.")

    imputer_values: dict[str, float] = {}
    for column in numeric_columns:
        train_values = scaled.loc[train_mask, column]
        median = train_values.median(skipna=True)
        if pd.isna(median):
            warnings.warn(
                f"All train values missing for {column}; filling with 0.",
                stacklevel=2,
            )
            median = 0.0
        imputer_values[column] = float(median)
        scaled[column] = scaled[column].fillna(median)

    scaler_stats: dict[str, dict[str, float]] = {}
    if numeric_columns:
        train_values = scaled.loc[train_mask, numeric_columns].astype(float)
        if StandardScaler is not None:
            scaler = StandardScaler()
            scaler.fit(train_values)
            scaled[numeric_columns] = scaler.transform(scaled[numeric_columns].astype(float))
            scaler_stats = {
                column: {
                    "mean": float(mean),
                    "scale": float(scale),
                    "method": "sklearn_standard_scaler",
                }
                for column, mean, scale in zip(
                    numeric_columns,
                    scaler.mean_,
                    scaler.scale_,
                )
            }
        else:
            means = train_values.mean()
            stds = train_values.std(ddof=0).replace(0, 1.0)
            scaled[numeric_columns] = (
                scaled[numeric_columns].astype(float) - means
            ) / stds
            scaler_stats = {
                column: {
                    "mean": float(means[column]),
                    "scale": float(stds[column]),
                    "method": "manual_train_mean_std",
                }
                for column in numeric_columns
            }

    column_info = {
        "metadata_columns": metadata_columns,
        "raw_feature_columns": numeric_columns,
        "scaled_feature_columns": numeric_columns,
        "missing_flag_columns": missing_flag_columns,
        "target_excluded_columns": target_excluded_columns,
    }
    return scaled, imputer_values, scaler_stats, column_info


def create_missing_tabular_report(
    pair_df: pd.DataFrame,
    tabular_df: pd.DataFrame,
    pair_feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a long-form report describing missing tabular matches."""
    pairs = prepare_pairs_for_join(pair_df)
    tabular = prepare_tabular_for_join(tabular_df)
    tabular_keys = set(
        tabular[["district_join_key", "year", "season"]]
        .dropna()
        .itertuples(index=False, name=None)
    )
    tabular_districts = set(tabular["district_join_key"].dropna().unique())

    rows: list[dict[str, Any]] = []
    pair_districts = pairs[["district", "district_join_key"]].drop_duplicates()
    for _, district_row in pair_districts.iterrows():
        if district_row["district_join_key"] not in tabular_districts:
            rows.append(
                {
                    "report_type": "missing_tabular_district",
                    "split": None,
                    "pair_type": None,
                    "district": district_row["district"],
                    "year": None,
                    "season": None,
                    "timepoint": None,
                    "count": 1,
                }
            )

    missing_t1 = pairs[
        ~pairs.apply(
            lambda row: (
                row["district_join_key"],
                row["year_t1"],
                row["season_t1"],
            )
            in tabular_keys,
            axis=1,
        )
    ]
    missing_t2 = pairs[
        ~pairs.apply(
            lambda row: (
                row["district_join_key"],
                row["year_t2"],
                row["season_t2"],
            )
            in tabular_keys,
            axis=1,
        )
    ]
    rows.extend(_missing_combo_rows(missing_t1, "t1", "year_t1", "season_t1"))
    rows.extend(_missing_combo_rows(missing_t2, "t2", "year_t2", "season_t2"))

    if not pair_feature_df.empty:
        rows.extend(_missing_count_rows(pair_feature_df, "district", "missing_by_district"))
        rows.extend(_missing_count_rows(pair_feature_df, "split", "missing_by_split"))
        rows.extend(_missing_count_rows(pair_feature_df, "pair_type", "missing_by_pair_type"))

    return pd.DataFrame(
        rows,
        columns=[
            "report_type",
            "split",
            "pair_type",
            "district",
            "year",
            "season",
            "timepoint",
            "count",
        ],
    )


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Save JSON data with UTF-8 encoding."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    return output_path


def _normalize_season_name(value: object) -> str:
    """Normalize project season names."""
    key = str(value).strip().replace(" ", "_").lower()
    key = re.sub(r"_+", "_", key)
    compact_key = key.replace("_", "")
    if key in SEASON_ALIASES:
        return SEASON_ALIASES[key]
    if compact_key in SEASON_ALIASES:
        return SEASON_ALIASES[compact_key]
    raise ValueError(f"Invalid season value: {value}")


def _identify_count_features(features: list[str]) -> list[str]:
    """Identify count/event/disaster features for sum-between aggregation."""
    return [
        feature
        for feature in features
        if any(keyword in feature.lower() for keyword in COUNT_KEYWORDS)
    ]


def _build_tabular_lookup(tabular: pd.DataFrame) -> dict[tuple[str, int, str], dict[str, Any]]:
    """Build a lookup for district/year/season tabular rows."""
    lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for _, row in tabular.iterrows():
        key = (row["district_join_key"], int(row["year"]), row["season"])
        lookup[key] = row.to_dict()
    return lookup


def _build_district_groups(tabular: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build sorted tabular groups by district join key."""
    return {
        district_key: group.sort_values("time_index").reset_index(drop=True)
        for district_key, group in tabular.groupby("district_join_key", dropna=True)
    }


def _metadata_row(pair: pd.Series) -> dict[str, Any]:
    """Create metadata and pair-derived feature values for one pair."""
    row = {column: pair[column] for column in PAIR_METADATA_COLUMNS if column in pair}
    row["time_gap_years"] = int(pair.get("time_gap_years", pair["year_t2"] - pair["year_t1"]))
    row["same_season"] = bool(pair["season_t1"] == pair["season_t2"])
    row["cross_season"] = bool(pair["season_t1"] != pair["season_t2"])
    return row


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    """Return numerator / denominator, or NaN for missing/zero denominators."""
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0.0:
        return float("nan")
    return float(numerator) / float(denominator)


def _between_records(
    district_groups: dict[str, pd.DataFrame],
    district_join_key: str,
    time_index_t1: int,
    time_index_t2: int,
) -> pd.DataFrame:
    """Return same-district seasonal records from t1 to t2 inclusive."""
    group = district_groups.get(district_join_key)
    if group is None:
        return pd.DataFrame()
    return group[
        (group["time_index"] >= int(time_index_t1))
        & (group["time_index"] <= int(time_index_t2))
    ]


def _one_hot(prefix: str, value: str, categories: list[str]) -> dict[str, int]:
    """Create one-hot columns for a categorical pair metadata value."""
    return {
        f"{prefix}_{category}": int(value == category)
        for category in categories
    }


def _missing_combo_rows(
    missing_df: pd.DataFrame,
    timepoint: str,
    year_column: str,
    season_column: str,
) -> list[dict[str, Any]]:
    """Summarize missing district/year/season combinations."""
    if missing_df.empty:
        return []

    group_columns = ["split", "pair_type", "district", year_column, season_column]
    grouped = missing_df.groupby(group_columns, dropna=False).size().reset_index(name="count")
    return [
        {
            "report_type": "missing_district_year_season",
            "split": row["split"],
            "pair_type": row["pair_type"],
            "district": row["district"],
            "year": int(row[year_column]),
            "season": row[season_column],
            "timepoint": timepoint,
            "count": int(row["count"]),
        }
        for _, row in grouped.iterrows()
    ]


def _missing_count_rows(
    pair_feature_df: pd.DataFrame,
    column: str,
    report_type: str,
) -> list[dict[str, Any]]:
    """Summarize tabular_any_missing counts by a selected column."""
    missing = pair_feature_df[pair_feature_df["tabular_any_missing"] == True]  # noqa: E712
    if missing.empty or column not in missing:
        return []

    counts = missing[column].dropna().value_counts().sort_index()
    return [
        {
            "report_type": report_type,
            "split": value if column == "split" else None,
            "pair_type": value if column == "pair_type" else None,
            "district": value if column == "district" else None,
            "year": None,
            "season": None,
            "timepoint": None,
            "count": int(count),
        }
        for value, count in counts.items()
    ]
