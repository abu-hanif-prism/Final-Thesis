"""Smoke-test model comparison using fake evaluation outputs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.model_comparison import (  # noqa: E402
    collect_model_results,
    compare_prediction_errors,
    create_model_comparison_markdown,
    load_predictions_file,
    rank_models,
)


def main() -> None:
    """Create fake outputs and run model comparison."""
    evaluation_root = Path("outputs/evaluation/smoke_compare")
    output_dir = Path("outputs/reports/smoke_model_comparison")
    experiments = ["cnn_regression", "swin_regression"]
    create_fake_evaluation_outputs(evaluation_root, experiments)

    metrics_df, prediction_paths_df = collect_model_results(experiments, evaluation_root, split="test")
    ranked_df = rank_models(metrics_df, primary_metric="rmse", lower_is_better=True)
    prediction_dfs = {
        row["experiment_name"]: load_predictions_file(row["predictions_path"])
        for _, row in prediction_paths_df.iterrows()
        if row["exists"]
    }
    error_summary = compare_prediction_errors(prediction_dfs)
    report = create_model_comparison_markdown(metrics_df, ranked_df, error_summary)
    save_outputs(output_dir, metrics_df, ranked_df, error_summary, report)

    print("Smoke model comparison completed.")
    print(f"  experiments: {experiments}")
    print("  ranking:")
    print(ranked_df[["rank", "experiment_name", "rmse", "status"]].to_string(index=False))
    print(f"  output_dir: {output_dir}")


def create_fake_evaluation_outputs(evaluation_root: Path, experiments: list[str]) -> None:
    """Create fake metrics/predictions for smoke comparison."""
    rng = np.random.default_rng(42)
    for index, experiment in enumerate(experiments):
        model_dir = evaluation_root / experiment / "test"
        model_dir.mkdir(parents=True, exist_ok=True)
        y_true = rng.uniform(0, 1, size=20)
        noise_scale = 0.08 + index * 0.04
        y_pred = np.clip(y_true + rng.normal(0, noise_scale, size=20), 0, 1)
        abs_error = np.abs(y_pred - y_true)
        squared_error = (y_pred - y_true) ** 2
        metrics = {
            "mae": float(abs_error.mean()),
            "mse": float(squared_error.mean()),
            "rmse": float(np.sqrt(squared_error.mean())),
            "r2": 0.5 - index * 0.1,
            "pearson_corr": 0.8 - index * 0.1,
        }
        with (model_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
        pd.DataFrame(
            {
                "patch_id": [f"patch_{i}" for i in range(20)],
                "district": rng.choice(["A", "B", "C"], size=20),
                "change_class": rng.choice(["low", "medium", "high"], size=20),
                "pair_type": rng.choice(["same_season_multiyear", "cross_season_sameyear"], size=20),
                "time_gap_group": rng.choice(["short", "medium", "long"], size=20),
                "y_true_change_ratio": y_true,
                "y_pred_change_ratio": y_pred,
                "abs_error": abs_error,
                "squared_error": squared_error,
            }
        ).to_csv(model_dir / "predictions.csv", index=False)


def save_outputs(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    error_summary: dict[str, pd.DataFrame],
    report: str,
) -> None:
    """Save smoke comparison outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "model_comparison_metrics.csv", index=False)
    ranked_df.to_csv(output_dir / "model_ranking.csv", index=False)
    for name, df in error_summary.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "model_comparison_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
