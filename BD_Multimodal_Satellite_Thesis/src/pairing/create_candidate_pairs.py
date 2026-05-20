"""Candidate temporal pair creation for Siamese change detection."""

import warnings
from typing import Any

import pandas as pd


PAIR_COLUMNS = [
    "pair_id",
    "district",
    "split",
    "image_id_t1",
    "image_id_t2",
    "year_t1",
    "season_t1",
    "year_t2",
    "season_t2",
    "season_order_t1",
    "season_order_t2",
    "time_index_t1",
    "time_index_t2",
    "time_gap_years",
    "season_gap",
    "same_season",
    "cross_season",
    "pair_type",
    "sentinel_path_t1",
    "sentinel_path_t2",
    "dw_path_t1",
    "dw_path_t2",
]


def add_time_columns(
    df: pd.DataFrame,
    season_order: dict[str, int],
) -> pd.DataFrame:
    """Add integer year, season_order, and time_index columns to image rows."""
    output = df.copy()
    output["year"] = pd.to_numeric(output["year"], errors="raise").astype(int)
    output["season_order"] = output["season"].map(season_order)

    missing_seasons = sorted(
        output.loc[output["season_order"].isna(), "season"].dropna().unique()
    )
    if missing_seasons:
        raise ValueError(f"Unknown seasons found: {missing_seasons}")

    output["season_order"] = output["season_order"].astype(int)
    output["time_index"] = output["year"] * 4 + output["season_order"]
    return output


def generate_candidate_pairs_for_district(district_df: pd.DataFrame) -> pd.DataFrame:
    """Generate all forward-time candidate pairs for one district and split."""
    if len(district_df) < 2:
        district = _first_value(district_df, "district")
        split = _first_value(district_df, "split")
        warnings.warn(
            f"Skipping district={district}, split={split}: fewer than 2 images.",
            stacklevel=2,
        )
        return pd.DataFrame(columns=PAIR_COLUMNS)

    sorted_df = district_df.sort_values(
        ["time_index", "image_id"],
    ).reset_index(drop=True)
    records: list[dict[str, Any]] = []

    rows = list(sorted_df.to_dict(orient="records"))
    for idx_t1, row_t1 in enumerate(rows[:-1]):
        for row_t2 in rows[idx_t1 + 1:]:
            if row_t2["time_index"] <= row_t1["time_index"]:
                continue

            pair_type = _determine_pair_type(row_t1, row_t2)
            if pair_type is None:
                continue

            record = _build_pair_record(row_t1, row_t2, pair_type)
            record["pair_id"] = build_pair_id(record)
            records.append(record)

    return pd.DataFrame(records, columns=PAIR_COLUMNS)


def generate_all_candidate_pairs(
    image_split_df: pd.DataFrame,
    season_order: dict[str, int],
) -> pd.DataFrame:
    """Generate candidate pairs inside each district/split group."""
    timed_df = add_time_columns(image_split_df, season_order)
    pair_tables = []

    for (split, district), group in timed_df.groupby(["split", "district"], sort=True):
        if len(group) < 2:
            warnings.warn(
                f"Skipping district={district}, split={split}: fewer than 2 images.",
                stacklevel=2,
            )
            continue
        pair_tables.append(generate_candidate_pairs_for_district(group))

    if not pair_tables:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    pairs = pd.concat(pair_tables, ignore_index=True)
    pairs["pair_id"] = _make_unique_pair_ids(pairs)
    return pairs[PAIR_COLUMNS]


def build_pair_id(row: pd.Series | dict[str, Any]) -> str:
    """Build a readable candidate pair identifier."""
    return (
        f"{row['district']}_{int(row['year_t1'])}_{row['season_t1']}"
        f"_to_{int(row['year_t2'])}_{row['season_t2']}"
    )


def _determine_pair_type(
    row_t1: dict[str, Any],
    row_t2: dict[str, Any],
) -> str | None:
    """Classify a forward-time pair according to project pair-type rules."""
    same_season = row_t1["season"] == row_t2["season"]
    same_year = row_t1["year"] == row_t2["year"]
    later_year = row_t2["year"] > row_t1["year"]

    if same_season and later_year:
        return "same_season_multiyear"
    if same_year and not same_season:
        return "cross_season_sameyear"
    if later_year and not same_season:
        return "cross_season_multiyear"

    return None


def _build_pair_record(
    row_t1: dict[str, Any],
    row_t2: dict[str, Any],
    pair_type: str,
) -> dict[str, Any]:
    """Build one candidate-pair output record."""
    same_season = row_t1["season"] == row_t2["season"]
    return {
        "pair_id": None,
        "district": row_t1["district"],
        "split": row_t1["split"],
        "image_id_t1": row_t1["image_id"],
        "image_id_t2": row_t2["image_id"],
        "year_t1": int(row_t1["year"]),
        "season_t1": row_t1["season"],
        "year_t2": int(row_t2["year"]),
        "season_t2": row_t2["season"],
        "season_order_t1": int(row_t1["season_order"]),
        "season_order_t2": int(row_t2["season_order"]),
        "time_index_t1": int(row_t1["time_index"]),
        "time_index_t2": int(row_t2["time_index"]),
        "time_gap_years": int(row_t2["year"]) - int(row_t1["year"]),
        "season_gap": int(row_t2["season_order"]) - int(row_t1["season_order"]),
        "same_season": bool(same_season),
        "cross_season": bool(not same_season),
        "pair_type": pair_type,
        "sentinel_path_t1": row_t1["sentinel_path"],
        "sentinel_path_t2": row_t2["sentinel_path"],
        "dw_path_t1": row_t1["dw_path"],
        "dw_path_t2": row_t2["dw_path"],
    }


def _make_unique_pair_ids(pairs: pd.DataFrame) -> list[str]:
    """Return pair IDs with numeric suffixes added for duplicate IDs."""
    seen: dict[str, int] = {}
    unique_ids = []

    for pair_id in pairs["pair_id"].astype(str):
        count = seen.get(pair_id, 0)
        if count == 0:
            unique_ids.append(pair_id)
        else:
            unique_ids.append(f"{pair_id}_{count + 1}")
        seen[pair_id] = count + 1

    return unique_ids


def _first_value(df: pd.DataFrame, column: str) -> Any:
    """Return the first value in a column, if available."""
    if df.empty or column not in df:
        return None
    return df[column].iloc[0]
