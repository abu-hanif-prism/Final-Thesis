"""Balanced final patch metadata selection utilities."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.file_utils import ensure_dir


VALID_CHANGE_CLASSES = {"low", "medium", "high"}
CLASS_ORDER = ["low", "medium", "high"]
PAIR_CAPS_BY_CLASS = {
    "low": 50,
    "medium": 40,
    "high": 20,
}
HIGH_TO_LOW_MEDIUM_RATIO = 1.30


def load_valid_labeled_patches(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only successful, finite low/medium/high labeled patches."""
    required = {"label_status", "change_ratio", "change_class"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Labeled patch DataFrame is missing columns: {missing}")

    change_ratio = pd.to_numeric(df["change_ratio"], errors="coerce")
    valid = df[
        (df["label_status"] == "success")
        & change_ratio.notna()
        & (change_ratio.abs() != float("inf"))
        & df["change_class"].isin(VALID_CHANGE_CLASSES)
    ].copy()
    valid["change_ratio"] = change_ratio.loc[valid.index]
    return valid.reset_index(drop=True)


def summarize_success_label_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Report true successful label distribution before final selection."""
    change_ratio = pd.to_numeric(df["change_ratio"], errors="coerce")
    success_mask = df["label_status"] == "success"
    finite_ratio_mask = change_ratio.notna() & (change_ratio.abs() != float("inf"))
    valid_class_mask = df["change_class"].isin(VALID_CHANGE_CLASSES)
    valid_mask = success_mask & finite_ratio_mask & valid_class_mask

    rows: list[dict[str, Any]] = [
        {"metric": "total_input_rows", "value": int(len(df))},
        {"metric": "success_rows", "value": int(success_mask.sum())},
        {
            "metric": "invalid_low_valid_ratio_rows",
            "value": int((df["label_status"] == "invalid_low_valid_ratio").sum()),
        },
        {"metric": "failed_rows", "value": int((df["label_status"] == "failed").sum())},
        {"metric": "valid_successful_rows", "value": int(valid_mask.sum())},
    ]
    valid = df[valid_mask].copy()
    for change_class in CLASS_ORDER:
        rows.append(
            {
                "metric": f"true_successful_{change_class}_count",
                "value": int((valid["change_class"] == change_class).sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_label_distribution(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize labeled patches by key balancing dimensions."""
    return {
        "counts_by_split": _count_table(df, "split"),
        "counts_by_district": _count_table(df, "district"),
        "counts_by_pair_type": _count_table(df, "pair_type"),
        "counts_by_time_gap_group": _count_table(df, "time_gap_group"),
        "counts_by_change_class": _count_table(df, "change_class"),
        "change_ratio_statistics": _change_ratio_stats(df),
        "patches_per_pair_statistics": _patches_per_pair_stats(df),
    }


def compute_pair_patch_usage(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return patch usage by pair, district, and raster pair."""
    raster_columns = ["dw_path_t1", "dw_path_t2"]
    if set(raster_columns).issubset(df.columns):
        raster_pair_usage = (
            df.groupby(raster_columns)
            .size()
            .reset_index(name="patch_count")
            .sort_values("patch_count", ascending=False)
            .reset_index(drop=True)
        )
    else:
        raster_pair_usage = pd.DataFrame(columns=[*raster_columns, "patch_count"])

    return {
        "patches_per_pair_id": _usage_table(df, "pair_id"),
        "patches_per_district": _usage_table(df, "district"),
        "patches_per_raster_pair": raster_pair_usage,
    }


def balanced_class_sampling(
    df: pd.DataFrame,
    split_name: str,
    target_total: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Select one split with low/medium preserved and high capped aggressively."""
    split_df = df[df["split"] == split_name].copy()
    if split_df.empty:
        return split_df

    low = _select_class_with_pair_cap(
        split_df,
        "low",
        split_name,
        PAIR_CAPS_BY_CLASS["low"],
        random_seed,
    )
    medium = _select_class_with_pair_cap(
        split_df,
        "medium",
        split_name,
        PAIR_CAPS_BY_CLASS["medium"],
        random_seed,
    )
    low_medium_total = len(low) + len(medium)
    high_target = int(round(low_medium_total * HIGH_TO_LOW_MEDIUM_RATIO))
    remaining_target = max(0, int(target_total) - low_medium_total)
    if remaining_target:
        high_target = min(high_target, remaining_target)

    high = _select_class_with_pair_cap(
        split_df,
        "high",
        split_name,
        PAIR_CAPS_BY_CLASS["high"],
        random_seed,
    )
    high = _enforce_high_district_soft_cap(
        high,
        split_name=split_name,
        target_count=high_target,
        random_seed=random_seed,
    )
    if len(high) > high_target:
        high = _spatial_sample_group(
            high,
            max_patches=high_target,
            random_seed=_stable_seed(f"{split_name}_high_target", random_seed),
        )
        high["selection_stage"] = high["selection_stage"].astype(str) + "|high_target_cap"

    sampled_tables = [table for table in [low, medium, high] if not table.empty]
    if not sampled_tables:
        return split_df.iloc[0:0].copy()
    return pd.concat(sampled_tables, ignore_index=True)


def enforce_pair_diversity(
    df: pd.DataFrame,
    max_patches_per_pair: int = 30,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Limit excessive patches from the same pair_id."""
    if df.empty or "pair_id" not in df:
        return df.copy()

    selected = []
    for pair_id, group in df.groupby("pair_id", sort=False):
        if len(group) <= max_patches_per_pair:
            selected.append(group)
            continue
        selected.append(
            _spatial_sample_group(
                group,
                max_patches=max_patches_per_pair,
                random_seed=_stable_seed(pair_id, random_seed),
            )
        )
    output = pd.concat(selected, ignore_index=True) if selected else df.iloc[0:0].copy()
    output["selection_stage"] = output["selection_stage"].astype(str) + "|pair_diversity"
    return output


def enforce_district_diversity(
    df: pd.DataFrame,
    max_patches_per_district: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Optionally cap patch counts per district."""
    if df.empty or max_patches_per_district is None:
        return df.copy()

    selected = []
    for district, group in df.groupby("district", sort=False):
        if len(group) <= max_patches_per_district:
            selected.append(group)
            continue
        selected.append(
            group.sample(
                n=max_patches_per_district,
                random_state=_stable_seed(district, random_seed),
            )
        )
    output = pd.concat(selected, ignore_index=True) if selected else df.iloc[0:0].copy()
    output["selection_stage"] = output["selection_stage"].astype(str) + "|district_diversity"
    return output


def enforce_pair_type_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Return patches with all available pair types preserved."""
    if df.empty:
        return df.copy()
    output = df.copy()
    output["selection_stage"] = output["selection_stage"].astype(str) + "|pair_type_checked"
    return output


def create_final_patch_selection(
    labeled_patch_df: pd.DataFrame,
    split_targets: dict[str, int],
    random_seed: int = 42,
) -> pd.DataFrame:
    """Create the final balanced patch metadata dataset."""
    valid = load_valid_labeled_patches(labeled_patch_df)
    split_tables = []
    for split_name, target_total in split_targets.items():
        sampled = balanced_class_sampling(valid, split_name, target_total, random_seed)
        sampled = enforce_pair_type_balance(sampled)
        split_tables.append(sampled)

    if not split_tables:
        return valid.iloc[0:0].copy()

    final = pd.concat(split_tables, ignore_index=True)
    final = final.drop_duplicates(subset=["patch_id"]).copy()
    final["final_selected"] = True
    final = final.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return final


def create_selection_reports(
    original_df: pd.DataFrame,
    final_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Create final patch selection report CSV files."""
    report_dir = ensure_dir(output_dir)
    valid_original = load_valid_labeled_patches(original_df)
    removed = valid_original[~valid_original["patch_id"].isin(set(final_df["patch_id"]))].copy()

    paths = {
        "summary": report_dir / "final_patch_selection_summary.csv",
        "class_distribution": report_dir / "final_class_distribution.csv",
        "district_distribution": report_dir / "final_district_distribution.csv",
        "pair_distribution": report_dir / "final_pair_distribution.csv",
        "timegap_distribution": report_dir / "final_timegap_distribution.csv",
        "removed_summary": report_dir / "final_removed_patch_summary.csv",
        "true_success_label_distribution": report_dir / "true_success_label_distribution.csv",
    }
    summarize_success_label_filter(original_df).to_csv(
        paths["true_success_label_distribution"], index=False, encoding="utf-8"
    )
    _original_vs_final_summary(valid_original, final_df).to_csv(
        paths["summary"], index=False, encoding="utf-8"
    )
    _distribution_report(valid_original, final_df, "change_class").to_csv(
        paths["class_distribution"], index=False, encoding="utf-8"
    )
    _distribution_report(valid_original, final_df, "district").to_csv(
        paths["district_distribution"], index=False, encoding="utf-8"
    )
    _distribution_report(valid_original, final_df, "pair_type").to_csv(
        paths["pair_distribution"], index=False, encoding="utf-8"
    )
    _distribution_report(valid_original, final_df, "time_gap_group").to_csv(
        paths["timegap_distribution"], index=False, encoding="utf-8"
    )
    _removed_summary(valid_original, final_df, removed).to_csv(
        paths["removed_summary"], index=False, encoding="utf-8"
    )
    return paths


def _spatial_sample_group(
    group: pd.DataFrame,
    max_patches: int,
    random_seed: int,
) -> pd.DataFrame:
    """Sample one pair group with light spatial spread when coordinates exist."""
    if not {"x", "y"}.issubset(group.columns):
        return group.sample(n=max_patches, random_state=random_seed)

    sorted_group = group.sample(frac=1.0, random_state=random_seed).sort_values(["y", "x"])
    if max_patches <= 1:
        return sorted_group.head(max_patches)
    positions = pd.Series(
        [round(i) for i in pd.Series(range(max_patches)) * (len(sorted_group) - 1) / (max_patches - 1)]
    ).astype(int)
    return sorted_group.iloc[positions.tolist()].copy()


def _select_class_with_pair_cap(
    split_df: pd.DataFrame,
    change_class: str,
    split_name: str,
    max_patches_per_pair: int,
    random_seed: int,
) -> pd.DataFrame:
    """Select one change class using its class-specific pair cap."""
    class_df = split_df[split_df["change_class"] == change_class].copy()
    if class_df.empty:
        return class_df

    selected = []
    if "pair_id" not in class_df:
        selected = [class_df]
    else:
        for pair_id, group in class_df.groupby("pair_id", sort=False):
            if len(group) <= max_patches_per_pair:
                sampled = group.copy()
            else:
                sampled = _spatial_sample_group(
                    group,
                    max_patches=max_patches_per_pair,
                    random_seed=_stable_seed(f"{split_name}_{change_class}_{pair_id}", random_seed),
                )
            selected.append(sampled)

    output = pd.concat(selected, ignore_index=True) if selected else class_df.iloc[0:0].copy()
    output["selection_stage"] = f"{change_class}_pair_cap_{max_patches_per_pair}"
    output["class_sampling_weight"] = len(output) / max(1, len(class_df))
    return output


def _enforce_high_district_soft_cap(
    high_df: pd.DataFrame,
    split_name: str,
    target_count: int,
    random_seed: int,
) -> pd.DataFrame:
    """Softly prevent one district from dominating high-change samples."""
    if high_df.empty or "district" not in high_df or target_count <= 0:
        return high_df.iloc[0:0].copy() if target_count <= 0 else high_df.copy()

    district_cap = max(300, int(round(target_count * 0.08)))
    selected = []
    for district, group in high_df.groupby("district", sort=False):
        if len(group) <= district_cap:
            selected.append(group)
        else:
            sampled = _spatial_sample_group(
                group,
                max_patches=district_cap,
                random_seed=_stable_seed(f"{split_name}_high_{district}", random_seed),
            )
            selected.append(sampled)
    output = pd.concat(selected, ignore_index=True) if selected else high_df.iloc[0:0].copy()
    output["selection_stage"] = output["selection_stage"].astype(str) + "|high_district_soft_cap"
    return output


def _original_vs_final_summary(original: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """Create high-level original vs final summary rows."""
    pair_counts = final.groupby("pair_id").size() if not final.empty else pd.Series(dtype=int)
    rows = [
        {"metric": "original_valid_patch_count", "value": len(original)},
        {"metric": "final_selected_patch_count", "value": len(final)},
        {"metric": "reduction_ratio", "value": 0.0 if len(original) == 0 else len(final) / len(original)},
        {"metric": "max_patches_per_pair_final", "value": 0 if pair_counts.empty else int(pair_counts.max())},
        {"metric": "mean_patches_per_pair_final", "value": 0.0 if pair_counts.empty else float(pair_counts.mean())},
    ]
    for split, count in final["split"].value_counts().sort_index().items():
        rows.append({"metric": f"final_split_{split}", "value": int(count)})
    return pd.DataFrame(rows)


def _distribution_report(original: pd.DataFrame, final: pd.DataFrame, column: str) -> pd.DataFrame:
    """Create original/final count report for one categorical column."""
    original_counts = original[column].value_counts().rename("original_count")
    final_counts = final[column].value_counts().rename("final_count")
    report = pd.concat([original_counts, final_counts], axis=1).fillna(0).astype(int)
    report["removed_count"] = report["original_count"] - report["final_count"]
    report["final_ratio"] = report["final_count"] / max(1, int(report["final_count"].sum()))
    return report.rename_axis(column).reset_index().sort_values("final_count", ascending=False)


def _removed_summary(original: pd.DataFrame, final: pd.DataFrame, removed: pd.DataFrame) -> pd.DataFrame:
    """Create removed patch summary, including top districts and pair IDs."""
    rows: list[dict[str, Any]] = [
        {"summary_type": "metric", "value": "valid_original_count", "count": len(original)},
        {"summary_type": "metric", "value": "final_count", "count": len(final)},
        {"summary_type": "metric", "value": "removed_count", "count": len(removed)},
    ]
    rows.extend(_top_removed_rows(removed, "district", "top_removed_district"))
    rows.extend(_top_removed_rows(removed, "change_class", "removed_change_class"))
    rows.extend(_top_removed_rows(removed, "pair_type", "removed_pair_type"))
    rows.extend(_top_removed_rows(removed, "time_gap_group", "removed_time_gap_group"))
    rows.extend(_top_removed_rows(removed, "pair_id", "top_removed_pair_id"))
    return pd.DataFrame(rows)


def _top_removed_rows(df: pd.DataFrame, column: str, summary_type: str, top_n: int = 50) -> list[dict[str, Any]]:
    """Return top removed value-count rows."""
    if df.empty or column not in df:
        return []
    return [
        {"summary_type": summary_type, "value": str(value), "count": int(count)}
        for value, count in df[column].value_counts().head(top_n).items()
    ]


def _count_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return count table for one column."""
    if df.empty or column not in df:
        return pd.DataFrame(columns=[column, "count"])
    return df[column].value_counts().rename_axis(column).reset_index(name="count")


def _usage_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return usage count table for one identifier column."""
    return _count_table(df, column).rename(columns={"count": "patch_count"})


def _change_ratio_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return change-ratio descriptive statistics."""
    stats = df["change_ratio"].describe()
    return stats.rename_axis("statistic").reset_index(name="value")


def _patches_per_pair_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return patches-per-pair descriptive statistics."""
    counts = df.groupby("pair_id").size()
    if counts.empty:
        return pd.DataFrame(columns=["statistic", "value"])
    return counts.describe().rename_axis("statistic").reset_index(name="value")


def _stable_seed(value: object, random_seed: int) -> int:
    """Create deterministic seed from text and base seed."""
    total = int(random_seed)
    for character in str(value):
        total = (total * 131 + ord(character)) % (2**32 - 1)
    return total
