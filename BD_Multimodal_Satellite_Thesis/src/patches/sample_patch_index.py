"""Spatially diverse sampling for patch coordinate indexes."""

import math

import numpy as np
import pandas as pd


def add_patch_grid_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add patch center coordinate metadata to a patch index DataFrame."""
    output = df.copy()
    if "patch_size" not in output.columns:
        raise ValueError("Patch index must contain a 'patch_size' column.")
    output["patch_center_x"] = output["x"].astype(float) + output["patch_size"].astype(float) / 2
    output["patch_center_y"] = output["y"].astype(float) + output["patch_size"].astype(float) / 2
    return output


def spatially_diverse_sample(
    group_df: pd.DataFrame,
    max_patches: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Sample up to max_patches from one pair using coarse spatial bins."""
    if len(group_df) <= max_patches:
        return group_df.copy()
    if max_patches <= 0:
        raise ValueError("max_patches must be a positive integer.")

    working = add_patch_grid_metadata(group_df)
    rng = np.random.default_rng(_stable_seed(working["pair_id"].iloc[0], random_seed))

    bin_count = max(1, int(math.ceil(math.sqrt(max_patches))))
    working["_x_bin"] = _bin_series(working["patch_center_x"], bin_count)
    working["_y_bin"] = _bin_series(working["patch_center_y"], bin_count)
    working["_bin"] = working["_x_bin"].astype(str) + "_" + working["_y_bin"].astype(str)
    working = working.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))

    selected_indices = []
    grouped_bins = {
        bin_id: group.index.tolist()
        for bin_id, group in working.groupby("_bin", sort=True)
    }
    active_bins = sorted(grouped_bins)

    while len(selected_indices) < max_patches and active_bins:
        next_active_bins = []
        for bin_id in active_bins:
            indices = grouped_bins[bin_id]
            if not indices:
                continue
            selected_indices.append(indices.pop(0))
            if indices:
                next_active_bins.append(bin_id)
            if len(selected_indices) >= max_patches:
                break
        active_bins = next_active_bins

    sampled = working.loc[selected_indices].drop(columns=["_x_bin", "_y_bin", "_bin"])
    return sampled.sort_values(["y", "x", "patch_id"]).reset_index(drop=True)


def sample_patch_index(
    patch_df: pd.DataFrame,
    max_patches_per_pair: int = 50,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Sample patch coordinates independently for every pair_id."""
    if "pair_id" not in patch_df.columns:
        raise ValueError("Patch index must contain a 'pair_id' column.")

    sampled_tables = []
    pair_counts = patch_df.groupby("pair_id", sort=False).size()
    for pair_id, group in patch_df.groupby("pair_id", sort=False):
        sampled = spatially_diverse_sample(
            group,
            max_patches=max_patches_per_pair,
            random_seed=random_seed,
        )
        original_count = int(pair_counts[pair_id])
        sampled["sampled_from_total_pair_patches"] = original_count
        sampled["pair_sampling_ratio"] = len(sampled) / original_count
        sampled_tables.append(sampled)

    if not sampled_tables:
        return pd.DataFrame(columns=list(patch_df.columns))

    sampled_index = pd.concat(sampled_tables, ignore_index=True)
    duplicate_count = int(sampled_index["patch_id"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Sampled patch index has {duplicate_count} duplicate patch IDs.")
    return sampled_index


def summarize_patch_sampling(
    full_patch_df: pd.DataFrame,
    sampled_patch_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return summary tables for patch-index sampling."""
    full_count = len(full_patch_df)
    sampled_count = len(sampled_patch_df)
    reduction_ratio = 0.0 if full_count == 0 else sampled_count / full_count
    sampled_per_pair = sampled_patch_df.groupby("pair_id").size()
    full_per_pair = full_patch_df.groupby("pair_id").size()

    metric_rows = [
        {"metric": "total_full_patches", "value": full_count},
        {"metric": "total_sampled_patches", "value": sampled_count},
        {"metric": "reduction_ratio", "value": reduction_ratio},
        {
            "metric": "min_sampled_patches_per_pair",
            "value": 0 if sampled_per_pair.empty else int(sampled_per_pair.min()),
        },
        {
            "metric": "max_sampled_patches_per_pair",
            "value": 0 if sampled_per_pair.empty else int(sampled_per_pair.max()),
        },
        {
            "metric": "mean_sampled_patches_per_pair",
            "value": 0.0 if sampled_per_pair.empty else float(sampled_per_pair.mean()),
        },
        {
            "metric": "min_original_patches_per_pair",
            "value": 0 if full_per_pair.empty else int(full_per_pair.min()),
        },
        {
            "metric": "max_original_patches_per_pair",
            "value": 0 if full_per_pair.empty else int(full_per_pair.max()),
        },
        {
            "metric": "mean_original_patches_per_pair",
            "value": 0.0 if full_per_pair.empty else float(full_per_pair.mean()),
        },
    ]

    return {
        "metrics": pd.DataFrame(metric_rows),
        "sampled_by_split": _count_table(sampled_patch_df, "split"),
        "sampled_by_district": _count_table(sampled_patch_df, "district"),
        "sampled_by_pair_type": _count_table(sampled_patch_df, "pair_type"),
        "sampled_by_time_gap_group": _count_table(sampled_patch_df, "time_gap_group"),
        "per_pair": _per_pair_stats(full_patch_df, sampled_patch_df),
    }


def _bin_series(values: pd.Series, bin_count: int) -> pd.Series:
    """Convert numeric values into deterministic integer bin IDs."""
    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        return pd.Series(0, index=values.index, dtype="int64")

    scaled = (values - min_value) / (max_value - min_value)
    bins = np.floor(scaled * bin_count).astype(int)
    return bins.clip(lower=0, upper=bin_count - 1)


def _stable_seed(value: object, random_seed: int) -> int:
    """Create a deterministic per-pair seed without relying on Python hash randomization."""
    text = str(value)
    total = int(random_seed)
    for character in text:
        total = (total * 131 + ord(character)) % (2**32 - 1)
    return total


def _count_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Build a value-count table for one column."""
    if df.empty or column not in df:
        return pd.DataFrame(columns=[column, "count"])
    return (
        df[column]
        .dropna()
        .value_counts()
        .sort_index()
        .rename_axis(column)
        .reset_index(name="count")
    )


def _per_pair_stats(full_patch_df: pd.DataFrame, sampled_patch_df: pd.DataFrame) -> pd.DataFrame:
    """Build original-vs-sampled per-pair count statistics."""
    full_counts = full_patch_df.groupby("pair_id").size().rename("original_patch_count")
    sampled_counts = sampled_patch_df.groupby("pair_id").size().rename("sampled_patch_count")
    metadata_columns = [
        column
        for column in ["pair_id", "split", "district", "pair_type", "time_gap_group"]
        if column in full_patch_df.columns
    ]
    metadata = full_patch_df[metadata_columns].drop_duplicates("pair_id")
    stats = metadata.merge(full_counts, on="pair_id", how="left").merge(
        sampled_counts,
        on="pair_id",
        how="left",
    )
    stats["sampled_patch_count"] = stats["sampled_patch_count"].fillna(0).astype(int)
    stats["sampling_ratio"] = (
        stats["sampled_patch_count"] / stats["original_patch_count"]
    )
    return stats.sort_values("original_patch_count", ascending=False).reset_index(drop=True)
