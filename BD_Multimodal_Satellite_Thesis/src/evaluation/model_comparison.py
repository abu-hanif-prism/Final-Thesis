"""Utilities for comparing saved model evaluation outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REGRESSION_METRICS = ["mae", "mse", "rmse", "r2", "pearson_corr"]
CLASSIFICATION_METRICS = ["accuracy", "macro_f1", "weighted_f1"]


def load_metrics_file(path: str | Path) -> dict[str, Any]:
    """Load metrics from JSON or CSV, returning an empty dict when missing."""
    path = Path(path)
    if not path.exists():
        print(f"Warning: metrics file missing: {path}")
        return {}
    try:
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            if {"metric", "value"}.issubset(df.columns):
                return dict(zip(df["metric"], df["value"]))
            if len(df) == 1:
                return df.iloc[0].to_dict()
            return {column: df[column].tolist() for column in df.columns}
    except Exception as exc:
        print(f"Warning: could not load metrics file {path}: {exc}")
        return {}
    print(f"Warning: unsupported metrics file type: {path}")
    return {}


def load_predictions_file(path: str | Path) -> pd.DataFrame:
    """Load predictions CSV/parquet, returning an empty DataFrame when missing."""
    path = Path(path)
    if not path.exists():
        print(f"Warning: predictions file missing: {path}")
        return pd.DataFrame()
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
    except Exception as exc:
        print(f"Warning: could not load predictions file {path}: {exc}")
        return pd.DataFrame()
    print(f"Warning: unsupported predictions file type: {path}")
    return pd.DataFrame()


def collect_model_results(
    experiment_names: list[str] | tuple[str, ...],
    evaluation_root: str | Path = "outputs/evaluation",
    split: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect metric rows and prediction file paths for experiments."""
    evaluation_root = Path(evaluation_root)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for experiment_name in experiment_names:
        experiment_dir = evaluation_root / experiment_name / split
        metrics_path = experiment_dir / "metrics.json"
        predictions_path = _resolve_predictions_path(experiment_dir)
        metrics = load_metrics_file(metrics_path)
        predictions_exists = predictions_path is not None and predictions_path.exists()
        prediction_count = None
        if predictions_exists:
            predictions_df = load_predictions_file(predictions_path)
            prediction_count = len(predictions_df)
        status = "ok" if metrics and predictions_exists else "missing"
        metric_rows.append(
            {
                "experiment_name": experiment_name,
                "model_name": _infer_model_name(experiment_name),
                "split": split,
                "mae": _metric_value(metrics, "mae", "reg_mae"),
                "mse": _metric_value(metrics, "mse", "reg_mse"),
                "rmse": _metric_value(metrics, "rmse", "reg_rmse"),
                "r2": _metric_value(metrics, "r2", "reg_r2"),
                "pearson_corr": _metric_value(metrics, "pearson_corr", "reg_pearson_corr"),
                "accuracy": _metric_value(metrics, "accuracy", "cls_accuracy"),
                "macro_f1": _metric_value(metrics, "macro_f1", "cls_macro_f1"),
                "weighted_f1": _metric_value(metrics, "weighted_f1", "cls_weighted_f1"),
                "average_loss": _metric_value(metrics, "average_loss", "loss"),
                "prediction_count": prediction_count,
                "metrics_path": str(metrics_path),
                "predictions_path": str(predictions_path) if predictions_path else "",
                "status": status,
            }
        )
        prediction_rows.append(
            {
                "experiment_name": experiment_name,
                "split": split,
                "predictions_path": str(predictions_path) if predictions_path else "",
                "exists": bool(predictions_exists),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def rank_models(
    metrics_df: pd.DataFrame,
    primary_metric: str = "rmse",
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """Rank models by primary metric, placing missing metrics last."""
    ranked = metrics_df.copy()
    if primary_metric not in ranked.columns:
        ranked[primary_metric] = np.nan
    ranked["_metric_missing"] = ranked[primary_metric].isna()
    ranked = ranked.sort_values(
        by=["_metric_missing", primary_metric],
        ascending=[True, lower_is_better],
        na_position="last",
    ).reset_index(drop=True)
    available = ~ranked["_metric_missing"]
    ranked["rank"] = np.nan
    ranked.loc[available, "rank"] = range(1, int(available.sum()) + 1)
    return ranked.drop(columns=["_metric_missing"])


def compare_prediction_errors(
    prediction_dfs: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Compare regression prediction errors overall and by common groups."""
    overall_rows = []
    group_rows = {column: [] for column in ["change_class", "pair_type", "time_gap_group", "district"]}
    for experiment_name, df in prediction_dfs.items():
        if df.empty or not {"abs_error", "squared_error"}.issubset(df.columns):
            continue
        overall_rows.append({"experiment_name": experiment_name, **_regression_error_summary(df)})
        for column in group_rows:
            if column not in df.columns:
                continue
            for value, group in df.groupby(column, dropna=False):
                group_rows[column].append(
                    {
                        "experiment_name": experiment_name,
                        column: value,
                        **_regression_error_summary(group),
                    }
                )
    return {
        "overall_error_comparison": pd.DataFrame(overall_rows),
        "by_change_class": pd.DataFrame(group_rows["change_class"]),
        "by_pair_type": pd.DataFrame(group_rows["pair_type"]),
        "by_time_gap_group": pd.DataFrame(group_rows["time_gap_group"]),
        "by_district": pd.DataFrame(group_rows["district"]),
    }


def create_model_comparison_markdown(
    metrics_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    error_summary_dict: dict[str, pd.DataFrame],
) -> str:
    """Create a readable Markdown model comparison report."""
    lines = ["# Model Comparison Report", ""]
    lines.extend(["## Ranking", ""])
    if ranked_df.empty:
        lines.append("No model metrics were found.")
    else:
        lines.append(_dataframe_to_markdown(ranked_df))
    lines.extend(["", "## Metric Comparison", ""])
    lines.append(_dataframe_to_markdown(metrics_df) if not metrics_df.empty else "No metrics available.")

    available_ranked = ranked_df.dropna(subset=["rank"]) if "rank" in ranked_df else pd.DataFrame()
    if not available_ranked.empty:
        best = available_ranked.sort_values("rank").iloc[0]
        lines.extend(
            [
                "",
                "## Best Model Summary",
                "",
                f"- Best model: `{best['experiment_name']}`",
                f"- Primary rank: {int(best['rank'])}",
            ]
        )

    missing = metrics_df[metrics_df.get("status", "") != "ok"] if not metrics_df.empty else pd.DataFrame()
    lines.extend(["", "## Missing Models / Files", ""])
    if missing.empty:
        lines.append("No missing model outputs detected.")
    else:
        for _, row in missing.iterrows():
            lines.append(f"- `{row['experiment_name']}` status: {row['status']}")

    lines.extend(["", "## Group-Level Error Highlights", ""])
    for name, df in error_summary_dict.items():
        lines.append(f"### {name}")
        if df.empty:
            lines.append("No regression prediction errors available.")
        else:
            sort_col = "rmse" if "rmse" in df.columns else df.columns[-1]
            lines.append(_dataframe_to_markdown(df.sort_values(sort_col).head(10)))
        lines.append("")
    return "\n".join(lines)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small Markdown table without optional tabulate dependency."""
    if df.empty:
        return ""
    table = df.copy()
    table = table.where(pd.notna(table), "")
    headers = [str(column) for column in table.columns]
    rows = [
        [str(value) for value in row]
        for row in table.astype(object).itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _resolve_predictions_path(experiment_dir: Path) -> Path | None:
    csv_path = experiment_dir / "predictions.csv"
    parquet_path = experiment_dir / "predictions.parquet"
    if csv_path.exists():
        return csv_path
    if parquet_path.exists():
        return parquet_path
    return csv_path


def _infer_model_name(experiment_name: str) -> str:
    return experiment_name.split("_")[0]


def _metric_value(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in metrics:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                return None
    return None


def _regression_error_summary(df: pd.DataFrame) -> dict[str, float | int]:
    abs_error = df["abs_error"].astype(float)
    squared_error = df["squared_error"].astype(float)
    row = {
        "count": int(len(df)),
        "mean_abs_error": float(abs_error.mean()),
        "median_abs_error": float(abs_error.median()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "max_abs_error": float(abs_error.max()),
    }
    if "y_true_change_ratio" in df.columns:
        row["mean_true_change_ratio"] = float(df["y_true_change_ratio"].astype(float).mean())
    if "y_pred_change_ratio" in df.columns:
        row["mean_pred_change_ratio"] = float(df["y_pred_change_ratio"].astype(float).mean())
    return row
