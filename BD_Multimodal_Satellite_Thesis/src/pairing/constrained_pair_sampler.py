"""Constrained candidate-pair sampling with image reuse control."""

import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.pairing.image_reuse_limiter import ImageReuseTracker


PAIR_TYPE_TARGETS = {
    "same_season_multiyear": 0.40,
    "cross_season_sameyear": 0.20,
    "cross_season_multiyear": 0.40,
}

REQUIRED_COLUMNS = {
    "pair_id",
    "split",
    "district",
    "image_id_t1",
    "image_id_t2",
    "year_t1",
    "year_t2",
    "time_gap_years",
    "pair_type",
}


def assign_time_gap_group(time_gap_years: int) -> str:
    """Assign a project time-gap group from integer year gap."""
    gap = int(time_gap_years)
    if gap == 0:
        return "same_year"
    if gap == 1:
        return "short"
    if gap in {2, 3}:
        return "medium"
    if gap in {4, 5, 6}:
        return "long"
    if gap in {7, 8, 9}:
        return "very_long"
    return "other"


def prepare_candidate_pairs(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate candidate pairs before constrained sampling."""
    missing = sorted(REQUIRED_COLUMNS - set(candidate_df.columns))
    if missing:
        raise ValueError(f"Candidate pair table is missing required columns: {missing}")

    clean = candidate_df.copy()
    for column in ["year_t1", "year_t2", "time_gap_years"]:
        clean[column] = pd.to_numeric(clean[column], errors="raise").astype(int)

    before_count = len(clean)
    clean = clean.drop_duplicates(subset=["pair_id"]).copy()
    duplicate_count = before_count - len(clean)
    if duplicate_count:
        warnings.warn(f"Removed {duplicate_count} duplicate pair_id rows.", stacklevel=2)

    self_pair_mask = clean["image_id_t1"] == clean["image_id_t2"]
    self_pair_count = int(self_pair_mask.sum())
    if self_pair_count:
        warnings.warn(f"Removed {self_pair_count} self-pair rows.", stacklevel=2)
        clean = clean.loc[~self_pair_mask].copy()

    clean["time_gap_group"] = clean["time_gap_years"].map(assign_time_gap_group)
    return clean.reset_index(drop=True)


def sample_pairs_with_reuse_limit(
    df: pd.DataFrame,
    max_pairs: int,
    max_image_use: int,
    random_seed: int,
    pair_type_targets: dict[str, float],
) -> pd.DataFrame:
    """Sample pairs while respecting image reuse and approximate type balance."""
    if df.empty or max_pairs <= 0:
        return _empty_selected(df)

    rng = np.random.default_rng(random_seed)
    tracker = ImageReuseTracker(max_image_use)
    selected_indices: list[int] = []
    selected_reasons: dict[int, str] = {}
    selected_pair_ids: set[str] = set()

    target_counts = _target_counts(max_pairs, pair_type_targets)
    selected_type_counts = {pair_type: 0 for pair_type in target_counts}
    queues = _build_pair_type_gap_queues(df, pair_type_targets, rng)
    gap_positions = {pair_type: 0 for pair_type in target_counts}
    exhausted_pair_types: set[str] = set()

    while len(selected_indices) < max_pairs:
        eligible_pair_types = [
            pair_type
            for pair_type, target_count in target_counts.items()
            if selected_type_counts[pair_type] < target_count
            and pair_type not in exhausted_pair_types
        ]
        if not eligible_pair_types:
            break

        pair_type = min(
            eligible_pair_types,
            key=lambda item: selected_type_counts[item] / max(1, target_counts[item]),
        )
        selected = _pop_addable_from_pair_type_queues(
            df=df,
            queues=queues.get(pair_type, {}),
            tracker=tracker,
            selected_pair_ids=selected_pair_ids,
            gap_positions=gap_positions,
            pair_type=pair_type,
        )
        if selected is None:
            exhausted_pair_types.add(pair_type)
            continue

        index, gap_group = selected
        selected_indices.append(index)
        selected_type_counts[pair_type] += 1
        selected_reasons[index] = f"target_{pair_type}_{gap_group}"

    if len(selected_indices) < max_pairs:
        remaining = df.loc[~df["pair_id"].isin(selected_pair_ids)].copy()
        remaining = remaining.sample(
            frac=1.0,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        for index, row in remaining.iterrows():
            if len(selected_indices) >= max_pairs:
                break
            if not tracker.can_add(row["image_id_t1"], row["image_id_t2"]):
                continue
            tracker.add_pair(row["image_id_t1"], row["image_id_t2"])
            selected_indices.append(index)
            selected_pair_ids.add(row["pair_id"])
            selected_reasons[index] = "backfill_reuse_limited"

    selected = df.loc[selected_indices].copy()
    selected["selected_reason"] = [
        selected_reasons[index] for index in selected_indices
    ]
    selected["image_use_count_t1"] = selected["image_id_t1"].map(tracker.get_usage)
    selected["image_use_count_t2"] = selected["image_id_t2"].map(tracker.get_usage)
    selected["max_image_use_used"] = int(max_image_use)
    return selected.reset_index(drop=True)


def constrained_sample_all_splits(
    candidate_df: pd.DataFrame,
    split_targets: dict[str, int],
    max_image_use_global: int = 3,
    fallback_max_image_use: int = 5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run constrained pair sampling independently for each split."""
    prepared = prepare_candidate_pairs(candidate_df)
    selected_tables = []

    for split, max_pairs in split_targets.items():
        split_df = prepared[prepared["split"] == split].copy()
        if split_df.empty:
            warnings.warn(f"No candidate pairs found for split={split}.", stacklevel=2)
            continue

        first_pass = sample_pairs_with_reuse_limit(
            split_df,
            max_pairs=max_pairs,
            max_image_use=max_image_use_global,
            random_seed=random_seed,
            pair_type_targets=PAIR_TYPE_TARGETS,
        )
        final = first_pass
        fallback_used = False

        if len(first_pass) < min(max_pairs, len(split_df)):
            fallback = sample_pairs_with_reuse_limit(
                split_df,
                max_pairs=max_pairs,
                max_image_use=fallback_max_image_use,
                random_seed=random_seed,
                pair_type_targets=PAIR_TYPE_TARGETS,
            )
            if len(fallback) > len(first_pass):
                fallback_used = True
                final = fallback
                message = (
                    f"Warning: split={split} used fallback max_image_use="
                    f"{fallback_max_image_use}; selected {len(final)} pairs "
                    f"instead of {len(first_pass)} with max_image_use="
                    f"{max_image_use_global}."
                )
                print(message)
                warnings.warn(message, stacklevel=2)

        final["fallback_max_image_use_used"] = bool(fallback_used)
        selected_tables.append(final)

        _warn_underrepresented(split, final)

    if not selected_tables:
        return pd.DataFrame()

    return pd.concat(selected_tables, ignore_index=True)


def _build_pair_type_gap_queues(
    df: pd.DataFrame,
    pair_type_targets: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, dict[str, list[int]]]:
    """Build shuffled index queues by pair type and time-gap group."""
    queues: dict[str, dict[str, list[int]]] = {}
    for pair_type in pair_type_targets:
        pair_type_df = df[df["pair_type"] == pair_type]
        gap_queues: dict[str, list[int]] = {}
        for gap_group, group in pair_type_df.groupby("time_gap_group", sort=True):
            shuffled = group.sample(
                frac=1.0,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            gap_queues[str(gap_group)] = shuffled.index.tolist()
        queues[pair_type] = gap_queues
    return queues


def _pop_addable_from_pair_type_queues(
    df: pd.DataFrame,
    queues: dict[str, list[int]],
    tracker: ImageReuseTracker,
    selected_pair_ids: set[str],
    gap_positions: dict[str, int],
    pair_type: str,
) -> tuple[int, str] | None:
    """Pop the next addable pair from one pair type, rotating gap groups."""
    active_gap_groups = sorted(
        gap_group for gap_group, indices in queues.items() if indices
    )
    if not active_gap_groups:
        return None

    start_position = gap_positions[pair_type] % len(active_gap_groups)
    ordered_gap_groups = (
        active_gap_groups[start_position:] + active_gap_groups[:start_position]
    )

    for offset, gap_group in enumerate(ordered_gap_groups):
        indices = queues[gap_group]
        while indices:
            index = indices.pop(0)
            row = df.loc[index]
            if row["pair_id"] in selected_pair_ids:
                continue
            if not tracker.can_add(row["image_id_t1"], row["image_id_t2"]):
                continue

            tracker.add_pair(row["image_id_t1"], row["image_id_t2"])
            selected_pair_ids.add(row["pair_id"])
            next_position = (start_position + offset + 1) % len(active_gap_groups)
            gap_positions[pair_type] = next_position
            return index, gap_group

    return None


def _select_balanced_by_gap(
    pair_type_df: pd.DataFrame,
    quota: int,
    tracker: ImageReuseTracker,
    selected_pair_ids: set[str],
    rng: np.random.Generator,
    selected_reasons: dict[int, str],
    selected_reason: str,
) -> list[int]:
    """Select rows from one pair type while rotating across time-gap groups."""
    if pair_type_df.empty or quota <= 0:
        return []

    grouped_indices: dict[str, list[int]] = {}
    for gap_group, group in pair_type_df.groupby("time_gap_group", sort=True):
        shuffled = group.sample(
            frac=1.0,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        grouped_indices[str(gap_group)] = shuffled.index.tolist()

    selected: list[int] = []
    active_groups = sorted(grouped_indices)
    group_position = 0
    empty_rounds = 0

    while len(selected) < quota and active_groups:
        gap_group = active_groups[group_position % len(active_groups)]
        indices = grouped_indices[gap_group]
        added = False

        while indices:
            index = indices.pop(0)
            row = pair_type_df.loc[index]
            if row["pair_id"] in selected_pair_ids:
                continue
            if not tracker.can_add(row["image_id_t1"], row["image_id_t2"]):
                continue
            tracker.add_pair(row["image_id_t1"], row["image_id_t2"])
            selected_pair_ids.add(row["pair_id"])
            selected_reasons[index] = f"{selected_reason}_{gap_group}"
            selected.append(index)
            added = True
            break

        if not indices:
            active_groups = [group for group in active_groups if grouped_indices[group]]
            group_position = 0
        elif active_groups:
            group_position = (group_position + 1) % len(active_groups)

        empty_rounds = 0 if added else empty_rounds + 1
        if empty_rounds > max(1, len(active_groups)):
            break

    return selected


def _target_counts(max_pairs: int, pair_type_targets: dict[str, float]) -> dict[str, int]:
    """Convert pair-type proportions into integer target counts."""
    raw_counts = {
        pair_type: int(round(max_pairs * proportion))
        for pair_type, proportion in pair_type_targets.items()
    }
    difference = max_pairs - sum(raw_counts.values())
    if difference:
        largest_type = max(pair_type_targets, key=pair_type_targets.get)
        raw_counts[largest_type] += difference
    return raw_counts


def _warn_underrepresented(split: str, selected: pd.DataFrame) -> None:
    """Warn when a selected split is missing expected pair types or gap groups."""
    expected_pair_types = set(PAIR_TYPE_TARGETS)
    present_pair_types = set(selected["pair_type"].dropna().unique()) if not selected.empty else set()
    missing_pair_types = sorted(expected_pair_types - present_pair_types)
    if missing_pair_types:
        warnings.warn(
            f"Split {split} is missing pair types: {missing_pair_types}",
            stacklevel=2,
        )

    multiyear = selected[
        selected["pair_type"].isin(
            ["same_season_multiyear", "cross_season_multiyear"]
        )
    ]
    expected_gap_groups = {"short", "medium", "long", "very_long"}
    present_gap_groups = set(multiyear["time_gap_group"].dropna().unique())
    missing_gap_groups = sorted(expected_gap_groups - present_gap_groups)
    if missing_gap_groups:
        warnings.warn(
            f"Split {split} is missing multiyear time gap groups: {missing_gap_groups}",
            stacklevel=2,
        )


def _empty_selected(df: pd.DataFrame) -> pd.DataFrame:
    """Return an empty selected-pair frame with expected added columns."""
    selected = df.iloc[0:0].copy()
    selected["selected_reason"] = pd.Series(dtype="object")
    selected["image_use_count_t1"] = pd.Series(dtype="int64")
    selected["image_use_count_t2"] = pd.Series(dtype="int64")
    selected["max_image_use_used"] = pd.Series(dtype="int64")
    return selected
