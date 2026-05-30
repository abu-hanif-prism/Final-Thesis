"""Compare saved evaluation outputs across model experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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


DEFAULT_EXPERIMENTS = [
    "cnn_regression",
    "swin_regression",
    "convnext_regression",
    "maxvit_regression_stable",
]


def parse_args() -> argparse.Namespace:
    """Parse model comparison arguments."""
    parser = argparse.ArgumentParser(description="Compare evaluated model outputs.")
    parser.add_argument("--evaluation_root", default="outputs/evaluation")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help=(
            "Experiments to compare. Accepts either comma-separated or "
            "space-separated values."
        ),
    )
    parser.add_argument("--primary_metric", default="rmse")
    parser.add_argument(
        "--lower_is_better",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether lower primary metric is better.",
    )
    parser.add_argument("--output_dir", default="outputs/reports/model_comparison")
    return parser.parse_args()


def main() -> None:
    """Run comparison and save reports."""
    args = parse_args()
    args.experiments = normalize_experiments(args.experiments)
    metrics_df, prediction_paths_df = collect_model_results(
        args.experiments,
        evaluation_root=args.evaluation_root,
        split=args.split,
    )
    ranked_df = rank_models(
        metrics_df,
        primary_metric=args.primary_metric,
        lower_is_better=args.lower_is_better,
    )
    prediction_dfs = {
        row["experiment_name"]: load_predictions_file(row["predictions_path"])
        for _, row in prediction_paths_df.iterrows()
        if row["exists"]
    }
    error_summary = compare_prediction_errors(prediction_dfs)
    report = create_model_comparison_markdown(metrics_df, ranked_df, error_summary)
    save_outputs(Path(args.output_dir), metrics_df, ranked_df, error_summary, report)
    print_console_summary(args, metrics_df, prediction_paths_df, ranked_df)


def save_outputs(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    error_summary: dict[str, pd.DataFrame],
    report: str,
) -> None:
    """Save all model comparison outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "model_comparison_metrics.csv", index=False)
    ranked_df.to_csv(output_dir / "model_ranking.csv", index=False)
    ranked_df.to_csv(output_dir / "model_comparison_ranked.csv", index=False)
    output_names = {
        "overall_error_comparison": "overall_error_comparison.csv",
        "by_change_class": "error_by_change_class.csv",
        "by_pair_type": "error_by_pair_type.csv",
        "by_time_gap_group": "error_by_time_gap_group.csv",
        "by_district": "error_by_district.csv",
    }
    for key, filename in output_names.items():
        error_summary.get(key, pd.DataFrame()).to_csv(output_dir / filename, index=False)
    (output_dir / "model_comparison_report.md").write_text(report, encoding="utf-8")


def print_console_summary(
    args: argparse.Namespace,
    metrics_df: pd.DataFrame,
    prediction_paths_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
) -> None:
    """Print concise comparison summary."""
    print("Model comparison completed.")
    print(f"  experiments checked: {args.experiments}")
    print(f"  split: {args.split}")
    print("  metrics files:")
    for _, row in metrics_df.iterrows():
        print(f"    {row['experiment_name']}: {'found' if row['status'] == 'ok' else 'missing/partial'} {row['metrics_path']}")
    print("  prediction files:")
    for _, row in prediction_paths_df.iterrows():
        print(f"    {row['experiment_name']}: {'found' if row['exists'] else 'missing'} {row['predictions_path']}")
    print(f"  ranking by {args.primary_metric}:")
    columns = ["rank", "experiment_name", args.primary_metric, "status"]
    existing_columns = [column for column in columns if column in ranked_df.columns]
    print(ranked_df[existing_columns].to_string(index=False))
    available = ranked_df.dropna(subset=["rank"]) if "rank" in ranked_df else pd.DataFrame()
    if not available.empty:
        best = available.sort_values("rank").iloc[0]
        print(f"  best model: {best['experiment_name']}")
    else:
        print("  best model: unavailable")
    print(f"  output folder: {args.output_dir}")


def normalize_experiments(experiments: list[str] | str) -> list[str]:
    """Normalize comma-separated or space-separated experiment arguments."""
    if isinstance(experiments, str):
        raw_values = [experiments]
    else:
        raw_values = experiments
    normalized: list[str] = []
    for value in raw_values:
        for part in str(value).split(","):
            experiment = part.strip()
            if experiment:
                normalized.append(experiment)
    return normalized


if __name__ == "__main__":
    main()
