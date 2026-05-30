"""Generate XAI case-study report from existing XAI outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.xai.case_study_report import build_case_studies, save_case_studies  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="XAI-08 case study report.")
    parser.add_argument("--model_name", required=True, choices=["cnn", "convnext", "swin", "maxvit"])
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--max_cases", type=int, default=10)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--output_dir", default="outputs/xai")
    return parser.parse_args()


def main() -> None:
    """Build and save case-study artifacts."""
    args = parse_args()
    checkpoint_path = args.checkpoint_path or str(Path("checkpoints") / f"{args.experiment_name}_best.pt")
    print(f"model_name: {args.model_name}", flush=True)
    print(f"experiment_name: {args.experiment_name}", flush=True)
    print(f"checkpoint: {checkpoint_path}", flush=True)
    print(f"max_cases: {args.max_cases}", flush=True)

    rows = build_case_studies(
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        output_root=args.output_dir,
        max_cases=args.max_cases,
        checkpoint_path=checkpoint_path,
    )
    csv_path, md_path = save_case_studies(rows, args.model_name, args.experiment_name, output_root=args.output_dir)
    print(f"cases generated: {len(rows)}", flush=True)
    print(f"case-study CSV: {csv_path}", flush=True)
    print(f"case-study report: {md_path}", flush=True)


if __name__ == "__main__":
    main()
