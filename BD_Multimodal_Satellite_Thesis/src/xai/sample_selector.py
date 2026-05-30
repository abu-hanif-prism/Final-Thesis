"""Representative sample selection for XAI workflows."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from statistics import median
from typing import Any


OUTPUT_COLUMNS = [
    "patch_id",
    "pair_id",
    "district",
    "split",
    "change_class",
    "pair_type",
    "time_gap_group",
    "npz_path",
    "y_true_change_ratio",
    "y_pred_change_ratio",
    "abs_error",
    "selection_reason",
]


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Load CSV rows with the Python standard library."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_csv_rows(
    rows: list[dict[str, Any]],
    path: str | Path,
    fieldnames: list[str] | None = None,
) -> Path:
    """Save rows to CSV with parent directory creation."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = _fieldnames(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return csv_path


def safe_float(value: Any, default: float | None = None) -> float | None:
    """Convert value to float, returning default when invalid."""
    if value is None:
        return default
    try:
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def compute_prediction_error(row: dict[str, Any]) -> float | None:
    """Return absolute prediction error from available columns."""
    abs_error = safe_float(row.get("abs_error"))
    if abs_error is not None:
        return abs(abs_error)

    y_true = safe_float(row.get("y_true_change_ratio"))
    y_pred = safe_float(row.get("y_pred_change_ratio"))
    if y_true is not None and y_pred is not None:
        return abs(y_true - y_pred)

    change_ratio = safe_float(row.get("change_ratio"))
    prediction = safe_float(row.get("prediction"))
    if change_ratio is not None and prediction is not None:
        return abs(change_ratio - prediction)
    return None


def merge_predictions_with_index(
    prediction_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge prediction rows with NPZ index rows by patch_id."""
    index_by_patch_id = {row.get("patch_id"): row for row in index_rows if row.get("patch_id")}
    merged: list[dict[str, Any]] = []
    for prediction in prediction_rows:
        patch_id = prediction.get("patch_id")
        if not patch_id or patch_id not in index_by_patch_id:
            continue
        index_row = index_by_patch_id[patch_id]
        abs_error = compute_prediction_error(prediction)
        row = {
            "patch_id": patch_id,
            "pair_id": prediction.get("pair_id") or index_row.get("pair_id"),
            "district": prediction.get("district") or index_row.get("district"),
            "split": prediction.get("split") or index_row.get("split"),
            "change_class": prediction.get("change_class") or index_row.get("change_class"),
            "pair_type": prediction.get("pair_type") or index_row.get("pair_type"),
            "time_gap_group": prediction.get("time_gap_group") or index_row.get("time_gap_group"),
            "npz_path": index_row.get("npz_path"),
            "y_true_change_ratio": prediction.get("y_true_change_ratio") or index_row.get("change_ratio"),
            "y_pred_change_ratio": prediction.get("y_pred_change_ratio") or prediction.get("prediction"),
            "abs_error": abs_error if abs_error is not None else "",
        }
        merged.append(row)
    return merged


def select_representative_samples(
    rows: list[dict[str, Any]],
    num_samples: int = 100,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """Select diverse, deterministic XAI samples."""
    rng = random.Random(random_seed)
    candidates = [row for row in rows if row.get("split") == "test"]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    target_total = max(0, int(num_samples))
    if target_total == 0:
        return []

    def add(row: dict[str, Any], reason: str) -> bool:
        patch_id = row.get("patch_id")
        if not patch_id or patch_id in selected_ids or len(selected) >= target_total:
            return False
        output_row = {key: row.get(key, "") for key in OUTPUT_COLUMNS}
        output_row["selection_reason"] = reason
        selected.append(output_row)
        selected_ids.add(patch_id)
        return True

    per_bucket = max(2, target_total // 20)
    for change_class in ["low", "medium", "high"]:
        bucket = [row for row in candidates if row.get("change_class") == change_class]
        rng.shuffle(bucket)
        for row in bucket[:per_bucket]:
            add(row, f"change_class_{change_class}")

    rows_with_error = [row for row in candidates if compute_prediction_error(row) is not None]
    low_error = sorted(rows_with_error, key=lambda row: compute_prediction_error(row) or 0.0)
    high_error = sorted(rows_with_error, key=lambda row: compute_prediction_error(row) or 0.0, reverse=True)
    for row in low_error[:per_bucket]:
        add(row, "low_prediction_error")
    for row in high_error[:per_bucket]:
        add(row, "high_prediction_error")

    for pair_type in ["same_season_multiyear", "cross_season_sameyear", "cross_season_multiyear"]:
        bucket = [row for row in candidates if row.get("pair_type") == pair_type]
        rng.shuffle(bucket)
        for row in bucket[:per_bucket]:
            add(row, f"pair_type_{pair_type}")

    for time_gap_group in ["short", "medium", "long", "very_long", "same_year"]:
        bucket = [row for row in candidates if row.get("time_gap_group") == time_gap_group]
        rng.shuffle(bucket)
        for row in bucket[:per_bucket]:
            add(row, f"time_gap_group_{time_gap_group}")

    district_rows: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        district_rows.setdefault(str(row.get("district", "")), []).append(row)
    for district in sorted(district_rows):
        bucket = district_rows[district]
        rng.shuffle(bucket)
        if bucket:
            add(bucket[0], f"district_{district}")

    remaining = [row for row in candidates if row.get("patch_id") not in selected_ids]
    fill_buckets = _stratified_buckets(remaining)
    while len(selected) < target_total and fill_buckets:
        progressed = False
        for key in sorted(fill_buckets):
            bucket = fill_buckets[key]
            if not bucket:
                continue
            row = bucket.pop()
            if add(row, "stratified_random_fill"):
                progressed = True
            if len(selected) >= target_total:
                break
        fill_buckets = {key: bucket for key, bucket in fill_buckets.items() if bucket}
        if not progressed:
            break
    return selected


def summarize_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return summary rows suitable for CSV."""
    errors = [value for value in (safe_float(row.get("abs_error")) for row in rows) if value is not None]
    summary: list[dict[str, Any]] = [{"metric": "total_selected", "group": "all", "value": len(rows)}]
    summary.extend(_count_rows(rows, "change_class"))
    summary.extend(_count_rows(rows, "pair_type"))
    summary.extend(_count_rows(rows, "time_gap_group"))
    district_counts = _counts(rows, "district")
    for district, count in sorted(district_counts.items(), key=lambda item: item[1], reverse=True)[:20]:
        summary.append({"metric": "district_count", "group": district, "value": count})
    if errors:
        summary.extend(
            [
                {"metric": "mean_abs_error", "group": "error_stats", "value": sum(errors) / len(errors)},
                {"metric": "median_abs_error", "group": "error_stats", "value": median(errors)},
                {"metric": "max_abs_error", "group": "error_stats", "value": max(errors)},
            ]
        )
    return summary


def _stratified_buckets(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("change_class", "")),
            str(row.get("pair_type", "")),
            str(row.get("time_gap_group", "")),
        )
        buckets.setdefault(key, []).append(row)
    rng = random.Random(42)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    return buckets


def _counts(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(column, ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_rows(rows: list[dict[str, Any]], column: str) -> list[dict[str, Any]]:
    return [
        {"metric": f"{column}_count", "group": key, "value": count}
        for key, count in sorted(_counts(rows, column).items())
    ]


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames
