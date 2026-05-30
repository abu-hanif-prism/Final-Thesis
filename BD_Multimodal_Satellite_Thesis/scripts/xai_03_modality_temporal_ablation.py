"""Run XAI-03 modality and temporal ablation on selected samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.xai.explanation_report import build_modality_temporal_report, save_markdown_report  # noqa: E402
from src.xai.handoff_loader import load_trained_model  # noqa: E402
from src.xai.modality_temporal_ablation import (  # noqa: E402
    ABLATION_COLUMNS,
    default_selected_samples_path,
    run_modality_temporal_ablation,
    summarize_ablation_results,
)
from src.xai.sample_selector import load_csv_rows, save_csv_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse modality/temporal ablation args."""
    parser = argparse.ArgumentParser(description="XAI-03 modality and temporal ablation.")
    parser.add_argument("--model_name", required=True, choices=["cnn", "convnext", "swin", "maxvit"])
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--index_path", default="data/npz/final_npz_index.csv")
    parser.add_argument("--selected_samples", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_dir", default="outputs/xai")
    return parser.parse_args()


def main() -> None:
    """Load model and selected samples, run ablations, and save outputs."""
    args = parse_args()
    device = resolve_device(args.device)
    selected_path = Path(args.selected_samples) if args.selected_samples else default_selected_samples_path(
        args.output_dir,
        args.experiment_name,
    )
    checkpoint_path = Path("checkpoints") / f"{args.experiment_name}_best.pt"
    print(f"model_name: {args.model_name}", flush=True)
    print(f"experiment_name: {args.experiment_name}", flush=True)
    print(f"checkpoint: {checkpoint_path}", flush=True)
    print(f"selected samples: {selected_path}", flush=True)
    print(f"device: {device}", flush=True)

    model, _, _ = load_trained_model(
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        checkpoint_dir="checkpoints",
        device=device,
    )
    selected_rows = load_csv_rows(selected_path)
    if args.num_samples is not None:
        selected_rows = selected_rows[: max(0, int(args.num_samples))]
    print(f"samples to ablate: {len(selected_rows)}", flush=True)

    result_rows = run_modality_temporal_ablation(
        model=model,
        selected_rows=selected_rows,
        device=device,
    )
    summary_rows = summarize_ablation_results(result_rows)

    output_root = Path(args.output_dir)
    result_path = output_root / "modality_ablation" / f"modality_temporal_ablation_{args.experiment_name}.csv"
    summary_path = output_root / "reports" / f"modality_temporal_ablation_summary_{args.experiment_name}.csv"
    report_path = output_root / "reports" / f"modality_temporal_ablation_report_{args.experiment_name}.md"

    save_csv_rows(result_rows, result_path, fieldnames=ABLATION_COLUMNS)
    save_csv_rows(summary_rows, summary_path, fieldnames=["metric", "group", "value"])
    report_text = build_modality_temporal_report(
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        checkpoint_path=checkpoint_path,
        num_samples=len(result_rows),
        summary_rows=summary_rows,
        output_csv_path=result_path,
    )
    save_markdown_report(report_text, report_path)

    print_summary(summary_rows)
    print(f"ablation csv: {result_path}", flush=True)
    print(f"summary csv: {summary_path}", flush=True)
    print(f"report md: {report_path}", flush=True)


def resolve_device(requested: str) -> torch.device:
    """Resolve requested device, falling back cleanly when CUDA is unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def print_summary(summary_rows: list[dict[str, object]]) -> None:
    """Print key summary metrics."""
    wanted = {
        "num_samples",
        "average_image_contribution",
        "average_tabular_contribution",
        "average_t1_contribution",
        "average_t2_contribution",
        "average_temporal_difference_contribution",
        "average_temporal_order_sensitivity",
    }
    for row in summary_rows:
        if row.get("metric") in wanted:
            print(f"{row['metric']}: {row['value']}", flush=True)


if __name__ == "__main__":
    main()
