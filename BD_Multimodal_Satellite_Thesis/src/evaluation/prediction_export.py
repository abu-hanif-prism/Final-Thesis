"""Prediction and metric export helpers for model evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_predictions(prediction_df: pd.DataFrame, output_path: str | Path) -> dict[str, Path | None]:
    """Save predictions to CSV and, when available, parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    parquet_path = output_path.with_suffix(".parquet")
    prediction_df.to_csv(csv_path, index=False, encoding="utf-8")
    parquet_saved: Path | None = None
    try:
        prediction_df.to_parquet(parquet_path, index=False)
        parquet_saved = parquet_path
    except Exception:
        parquet_saved = None
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
    pd.DataFrame([{"metric": key, "value": value} for key, value in clean_metrics.items()]).to_csv(
        csv_path,
        index=False,
        encoding="utf-8",
    )
    return {"json": json_path, "csv": csv_path}


def save_group_summaries(
    summary_dict: dict[str, pd.DataFrame],
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Path]:
    """Save group summary DataFrames as CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for group_name, dataframe in summary_dict.items():
        path = output_dir / f"{prefix}_by_{group_name}.csv"
        dataframe.to_csv(path, index=False, encoding="utf-8")
        paths[group_name] = path
    return paths


def make_evaluation_report_text(
    metrics_dict: dict[str, Any],
    group_summary_dict: dict[str, pd.DataFrame] | None = None,
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
        for group_name, dataframe in group_summary_dict.items():
            lines.append(f"- {group_name}: {len(dataframe)} groups")
            if not dataframe.empty:
                lines.append(dataframe.head(5).to_string(index=False))
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
