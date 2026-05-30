"""Select representative test samples for XAI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.xai.sample_selector import (  # noqa: E402
    load_csv_rows,
    merge_predictions_with_index,
    safe_float,
    save_csv_rows,
    select_representative_samples,
    summarize_selection,
)
from src.xai.xai_config import DEFAULT_NPZ_INDEX_PATH, DEFAULT_XAI_OUTPUT_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse representative sample selection args."""
    parser = argparse.ArgumentParser(description="XAI Step 2: select representative samples.")
    parser.add_argument("--model_name", required=True, choices=["cnn", "convnext", "swin", "maxvit"])
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--index_path", default=DEFAULT_NPZ_INDEX_PATH)
    parser.add_argument("--predictions_path", default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Load predictions/index, select representative samples, and save reports."""
    args = parse_args()
    experiment_name = args.experiment_name or f"{args.model_name}_regression"
    predictions_path = Path(args.predictions_path) if args.predictions_path else (
        Path("outputs/evaluation") / experiment_name / "test" / "predictions.csv"
    )

    index_rows = load_csv_rows(args.index_path)
    prediction_rows = load_csv_rows(predictions_path)
    merged_rows = merge_predictions_with_index(prediction_rows, index_rows)
    selected_rows = select_representative_samples(
        merged_rows,
        num_samples=args.num_samples,
        random_seed=args.random_seed,
    )
    summary_rows = summarize_selection(selected_rows)

    selected_path = (
        Path(DEFAULT_XAI_OUTPUT_DIR)
        / "selected_samples"
        / f"xai_selected_samples_{experiment_name}.csv"
    )
    summary_path = (
        Path(DEFAULT_XAI_OUTPUT_DIR)
        / "reports"
        / f"xai_sample_selection_summary_{experiment_name}.csv"
    )
    report_path = (
        Path(DEFAULT_XAI_OUTPUT_DIR)
        / "reports"
        / f"xai_sample_selection_report_{experiment_name}.md"
    )

    save_csv_rows(selected_rows, selected_path)
    save_csv_rows(summary_rows, summary_path, fieldnames=["metric", "group", "value"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_markdown_report(
            model_name=args.model_name,
            experiment_name=experiment_name,
            predictions_path=predictions_path,
            index_path=Path(args.index_path),
            prediction_count=len(prediction_rows),
            index_count=len(index_rows),
            merged_count=len(merged_rows),
            selected_rows=selected_rows,
            summary_rows=summary_rows,
            selected_path=selected_path,
            summary_path=summary_path,
        ),
        encoding="utf-8",
    )

    print(f"model_name: {args.model_name}")
    print(f"experiment_name: {experiment_name}")
    print(f"prediction rows loaded: {len(prediction_rows)}")
    print(f"index rows loaded: {len(index_rows)}")
    print(f"merged rows: {len(merged_rows)}")
    print(f"selected rows: {len(selected_rows)}")
    print_counts("counts by change_class", selected_rows, "change_class")
    print_counts("counts by pair_type", selected_rows, "pair_type")
    print_counts("counts by time_gap_group", selected_rows, "time_gap_group")
    print_error_stats(selected_rows)
    print(f"selected samples path: {selected_path}")
    print(f"summary path: {summary_path}")
    print(f"report path: {report_path}")


def build_markdown_report(
    model_name: str,
    experiment_name: str,
    predictions_path: Path,
    index_path: Path,
    prediction_count: int,
    index_count: int,
    merged_count: int,
    selected_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    selected_path: Path,
    summary_path: Path,
) -> str:
    """Build a compact Markdown selection report."""
    lines = [
        f"# XAI Sample Selection: {experiment_name}",
        "",
        f"- model_name: {model_name}",
        f"- experiment_name: {experiment_name}",
        f"- predictions_path: {predictions_path}",
        f"- index_path: {index_path}",
        f"- prediction rows loaded: {prediction_count}",
        f"- index rows loaded: {index_count}",
        f"- merged rows: {merged_count}",
        f"- selected rows: {len(selected_rows)}",
        f"- selected_samples_csv: {selected_path}",
        f"- summary_csv: {summary_path}",
        "",
        "## Summary",
        "",
    ]
    for row in summary_rows:
        lines.append(f"- {row['metric']} / {row['group']}: {row['value']}")
    return "\n".join(lines) + "\n"


def print_counts(title: str, rows: list[dict[str, object]], column: str) -> None:
    """Print value counts for one selected-sample column."""
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(column, ""))
        counts[key] = counts.get(key, 0) + 1
    print(f"{title}: {dict(sorted(counts.items()))}")


def print_error_stats(rows: list[dict[str, object]]) -> None:
    """Print mean/median/max absolute error."""
    errors = [value for value in (safe_float(row.get("abs_error")) for row in rows) if value is not None]
    if not errors:
        print("abs_error stats: unavailable")
        return
    sorted_errors = sorted(errors)
    mid = len(sorted_errors) // 2
    if len(sorted_errors) % 2:
        median = sorted_errors[mid]
    else:
        median = (sorted_errors[mid - 1] + sorted_errors[mid]) / 2
    print(f"mean abs_error: {sum(errors) / len(errors)}")
    print(f"median abs_error: {median}")
    print(f"max abs_error: {max(errors)}")


if __name__ == "__main__":
    main()
