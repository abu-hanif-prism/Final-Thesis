"""Pair distribution and image reuse statistics."""

from pathlib import Path

import pandas as pd

from src.utils.file_utils import ensure_dir


def summarize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected pairs by split, district, type, and time gap."""
    rows = [
        {
            "summary_type": "metric",
            "value": "total_pairs",
            "count": int(len(df)),
        },
        {
            "summary_type": "metric",
            "value": "same_season_count",
            "count": _true_count(df, "same_season"),
        },
        {
            "summary_type": "metric",
            "value": "cross_season_count",
            "count": _true_count(df, "cross_season"),
        },
    ]
    rows.extend(_count_rows(df, "split", "pairs_by_split"))
    rows.extend(_count_rows(df, "district", "pairs_by_district"))
    rows.extend(_count_rows(df, "pair_type", "pairs_by_pair_type"))
    rows.extend(_count_rows(df, "time_gap_group", "pairs_by_time_gap_group"))
    rows.extend(_count_rows(df, "time_gap_years", "pairs_by_time_gap_years"))
    return pd.DataFrame(rows, columns=["summary_type", "value", "count"])


def summarize_image_reuse(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize how often images are reused across selected pairs."""
    usage = _image_usage_counts(df)
    if usage.empty:
        return pd.DataFrame(
            [
                {"summary_type": "metric", "value": "unique_images", "count": 0},
                {"summary_type": "metric", "value": "max_image_reuse", "count": 0},
                {"summary_type": "metric", "value": "mean_image_reuse", "count": 0.0},
            ],
            columns=["summary_type", "value", "count"],
        )

    rows = [
        {
            "summary_type": "metric",
            "value": "unique_images",
            "count": int(len(usage)),
        },
        {
            "summary_type": "metric",
            "value": "max_image_reuse",
            "count": int(usage.max()),
        },
        {
            "summary_type": "metric",
            "value": "mean_image_reuse",
            "count": float(usage.mean()),
        },
    ]

    distribution = usage.value_counts().sort_index()
    rows.extend(
        {
            "summary_type": "image_reuse_distribution",
            "value": str(int(usage_count)),
            "count": int(image_count),
        }
        for usage_count, image_count in distribution.items()
    )
    return pd.DataFrame(rows, columns=["summary_type", "value", "count"])


def save_pair_statistics(df: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Save pair and image reuse statistics CSV files."""
    report_dir = ensure_dir(output_dir)
    pair_statistics_path = report_dir / "constrained_pair_statistics.csv"
    image_reuse_path = report_dir / "image_reuse_statistics.csv"

    summarize_pairs(df).to_csv(pair_statistics_path, index=False, encoding="utf-8")
    summarize_image_reuse(df).to_csv(image_reuse_path, index=False, encoding="utf-8")

    return {
        "pair_statistics": pair_statistics_path,
        "image_reuse_statistics": image_reuse_path,
    }


def _image_usage_counts(df: pd.DataFrame) -> pd.Series:
    """Return usage counts over both image_id_t1 and image_id_t2."""
    required_columns = {"image_id_t1", "image_id_t2"}
    if df.empty or not required_columns.issubset(df.columns):
        return pd.Series(dtype="int64")

    image_ids = pd.concat(
        [df["image_id_t1"], df["image_id_t2"]],
        ignore_index=True,
    )
    return image_ids.value_counts().sort_index()


def _count_rows(
    df: pd.DataFrame,
    column: str,
    summary_type: str,
) -> list[dict[str, object]]:
    """Return value-count rows for a column."""
    if df.empty or column not in df:
        return []

    counts = df[column].dropna().value_counts().sort_index()
    return [
        {
            "summary_type": summary_type,
            "value": str(value),
            "count": int(count),
        }
        for value, count in counts.items()
    ]


def _true_count(df: pd.DataFrame, column: str) -> int:
    """Count True values in a boolean-like column."""
    if df.empty or column not in df:
        return 0
    return int((df[column] == True).sum())  # noqa: E712
