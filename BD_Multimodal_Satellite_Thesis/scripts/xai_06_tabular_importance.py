"""Run XAI-06 tabular feature ablation importance."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.xai.handoff_loader import load_trained_model  # noqa: E402
from src.xai.sample_selector import load_csv_rows, save_csv_rows  # noqa: E402
from src.xai.tabular_importance import (  # noqa: E402
    GROUP_SUMMARY_COLUMNS,
    TABULAR_COLUMNS,
    TABULAR_SUMMARY_COLUMNS,
    build_tabular_report,
    compute_tabular_importance,
    load_feature_names,
    save_top_feature_plot,
    summarize_tabular_groups,
    summarize_tabular_importance,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="XAI-06 tabular feature importance.")
    parser.add_argument("--model_name", required=True, choices=["cnn", "convnext", "swin", "maxvit"])
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--selected_samples", default=None)
    parser.add_argument("--feature_columns_path", default="data/tabular/processed/pair_tabular_feature_columns.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--output_dir", default="outputs/xai")
    return parser.parse_args()


def main() -> None:
    """Run tabular feature importance and save outputs."""
    args = parse_args()
    device = resolve_device(args.device)
    selected_path = Path(args.selected_samples) if args.selected_samples else (
        Path(args.output_dir) / "selected_samples" / f"xai_selected_samples_{args.experiment_name}.csv"
    )
    print(f"model_name: {args.model_name}", flush=True)
    print(f"experiment_name: {args.experiment_name}", flush=True)
    print(f"selected samples: {selected_path}", flush=True)
    print(f"device: {device}", flush=True)

    model, _, _ = load_trained_model(args.model_name, args.experiment_name, device=device)
    selected_rows = load_csv_rows(selected_path)
    sample_count = min(len(selected_rows), max(0, int(args.num_samples)))
    feature_names = load_feature_names(args.feature_columns_path)
    per_sample_rows = compute_tabular_importance(
        model=model,
        selected_rows=selected_rows,
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        device=device,
        feature_names=feature_names,
        num_samples=sample_count,
    )
    summary_rows = summarize_tabular_importance(per_sample_rows)
    group_rows = summarize_tabular_groups(summary_rows)

    output_root = Path(args.output_dir)
    tabular_dir = output_root / "tabular_importance"
    report_dir = output_root / "reports"
    per_sample_path = tabular_dir / f"tabular_importance_per_sample_{args.experiment_name}.csv"
    summary_path = tabular_dir / f"tabular_importance_summary_{args.experiment_name}.csv"
    group_path = tabular_dir / f"tabular_group_importance_{args.experiment_name}.csv"
    plot_path = tabular_dir / f"top_tabular_features_{args.experiment_name}.png"
    report_path = report_dir / f"tabular_importance_report_{args.experiment_name}.md"

    save_csv_rows(per_sample_rows, per_sample_path, fieldnames=TABULAR_COLUMNS)
    save_csv_rows(summary_rows, summary_path, fieldnames=TABULAR_SUMMARY_COLUMNS)
    save_csv_rows(group_rows, group_path, fieldnames=GROUP_SUMMARY_COLUMNS)
    save_top_feature_plot(summary_rows, plot_path, title=f"Top tabular features: {args.experiment_name}", top_k=args.top_k)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_tabular_report(
            args.model_name,
            args.experiment_name,
            sample_count,
            per_sample_path,
            summary_path,
            group_path,
            plot_path,
            summary_rows,
            group_rows,
        ),
        encoding="utf-8",
    )

    print(f"samples explained: {sample_count}", flush=True)
    print(f"per-sample CSV: {per_sample_path}", flush=True)
    print(f"feature summary CSV: {summary_path}", flush=True)
    print(f"group summary CSV: {group_path}", flush=True)
    print(f"bar plot: {plot_path}", flush=True)
    print(f"report: {report_path}", flush=True)


def resolve_device(requested: str) -> torch.device:
    """Resolve device with safe CUDA fallback."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


if __name__ == "__main__":
    main()
