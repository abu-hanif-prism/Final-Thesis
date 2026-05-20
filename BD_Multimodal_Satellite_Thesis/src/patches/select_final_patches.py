"""Balanced final patch metadata selection utilities."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.file_utils import ensure_dir


VALID_CHANGE_CLASSES = {"low", "medium", "high"}
CLASS_TARGETS = {
    "low": 0.20,
    "medium": 0.35,
    "high": 0.45,
}


def load_valid_labeled_patches(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only successful, finite low/medium/high labeled patches."""
    required = {"label_status", "change_ratio", "change_class"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Labeled patch DataFrame is missing columns: {missing}")

    valid = df[
        (df["label_status"] == "success")
        & df["change_ratio"].notna()
        & df["change_class"].isin(VALID_CHANGE_CLASSES)
    ].copy()
    return valid.reset_index(drop=True)


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
    """Sample one split toward low/medium/high targets without oversampling."""
    split_df = df[df["split"] == split_name].copy()
    if split_df.empty:
        return split_df

    effective_total = min(int(target_total), len(split_df))
    sampled_tables = []
    for change_class in ["low", "medium", "high"]:
        class_df = split_df[split_df["change_class"] == change_class].copy()
        if class_df.empty:
            continue
        class_target = int(round(effective_total * CLASS_TARGETS[change_class]))
        class_target = max(1, class_target)
        sample_count = min(len(class_df), class_target)
        sampled = class_df.sample(
            n=sample_count,
            random_state=_stable_seed(f"{split_name}_{change_class}", random_seed),
        )
        sampled["selection_stage"] = f"class_balanced_{change_class}"
        sampled["class_sampling_weight"] = (
            1.0 if sample_count == len(class_df) else sample_count / len(class_df)
        )
        sampled_tables.append(sampled)

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
        sampled = enforce_pair_diversity(sampled, max_patches_per_pair=30, random_seed=random_seed)
        sampled = enforce_district_diversity(sampled, max_patches_per_district=None, random_seed=random_seed)
        sampled = enforce_pair_type_balance(sampled)
        split_tables.append(sampled)

    if not split_tables:
        return valid.iloc[0:0].copy()

    final = pd.concat(split_tables, ignore_index=True)
    final = final.drop_duplicates(subset=["patch_id"]).copy()
    final["final_selected"] = True
    return final.reset_index(drop=True)


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
    }
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
