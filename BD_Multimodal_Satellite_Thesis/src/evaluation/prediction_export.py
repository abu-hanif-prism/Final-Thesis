"""Prediction and metric export helpers for model evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def save_predictions(
    predictions: list[dict[str, Any]],
    output_path: str | Path,
    save_parquet: bool = False,
) -> dict[str, Path | None]:
    """Save predictions to CSV and optionally parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    parquet_path = output_path.with_suffix(".parquet")
    _write_csv_rows(predictions, csv_path)
    parquet_saved: Path | None = None
    if save_parquet:
        try:
            import pandas as pd

            prediction_df = pd.DataFrame(predictions)
            prediction_df.to_parquet(parquet_path, index=False)
            parquet_saved = parquet_path
        except Exception as exc:
            print(f"Warning: parquet prediction export failed and was skipped: {exc}", flush=True)
    return {"csv": csv_path, "parquet": parquet_saved}


def save_metrics(metrics_dict: dict[str, Any], output_path: str | Path) -> dict[str, Path]:
    """Save metrics as JSON and CSV-friendly key/value table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_suffix(".json")
    csv_path = output_path.with_suffix(".csv")
    clean_metrics = {key: _json_safe(value) for key, value in metrics_dict.items()}
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(clean_metrics, file, indent=2, allow_nan=True)
    _write_csv_rows([{"metric": key, "value": value} for key, value in clean_metrics.items()], csv_path)
    return {"json": json_path, "csv": csv_path}


def save_group_summaries(
    summary_dict: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Path]:
    """Save group summary DataFrames as CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for group_name, rows in summary_dict.items():
        path = output_dir / f"{prefix}_by_{group_name}.csv"
        _write_csv_rows(rows, path)
        paths[group_name] = path
    return paths


def make_evaluation_report_text(
    metrics_dict: dict[str, Any],
    group_summary_dict: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Create a readable text evaluation report."""
    lines = ["Evaluation Report", "=================", "", "Metrics:"]
    for key in sorted(metrics_dict):
        value = metrics_dict[key]
        if isinstance(value, float):
            value_text = f"{value:.6f}" if np.isfinite(value) else "nan"
        else:
            value_text = str(value)
        lines.append(f"- {key}: {value_text}")

    if group_summary_dict:
        lines.extend(["", "Group summaries:"])
        for group_name, rows in group_summary_dict.items():
            lines.append(f"- {group_name}: {len(rows)} groups")
            if rows:
                lines.append(_format_rows_preview(rows[:5]))
    return "\n".join(lines) + "\n"


def save_evaluation_report(report_text: str, output_path: str | Path) -> Path:
    """Save text evaluation report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    return output_path


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_csv_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Write dictionaries to CSV with the standard library."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_safe(row.get(key)) for key in fieldnames})


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    """Return stable fieldnames preserving first-seen order."""
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames


def _csv_safe(value: Any) -> Any:
    """Convert values to CSV-safe scalars."""
    value = _json_safe(value)
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return "nan"
    return value


def _format_rows_preview(rows: list[dict[str, Any]]) -> str:
    """Format a small list of rows for the text report."""
    if not rows:
        return ""
    fieldnames = _fieldnames(rows)
    lines = [", ".join(fieldnames)]
    for row in rows:
        lines.append(", ".join(str(row.get(key, "")) for key in fieldnames))
    return "\n".join(lines)
