"""Check that an XAI handoff/model package can be loaded and run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.xai.handoff_loader import (  # noqa: E402
    get_test_samples,
    load_npz_index,
    load_npz_sample,
    load_trained_model,
    prepare_xai_batch,
    resolve_experiment_name,
)
from src.xai.xai_config import DEFAULT_NPZ_INDEX_PATH, DEFAULT_XAI_OUTPUT_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse XAI handoff check arguments."""
    parser = argparse.ArgumentParser(description="XAI Step 1: check handoff/model loading.")
    parser.add_argument("--model_name", required=True, choices=["cnn", "convnext", "swin", "maxvit"])
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--index_path", default=DEFAULT_NPZ_INDEX_PATH)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_samples", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    """Load trained model, sample data, run forward pass, and save report."""
    args = parse_args()
    device = torch.device(args.device)
    experiment_name = resolve_experiment_name(args.model_name, args.experiment_name)
    checkpoint_path = Path(args.checkpoint_dir) / f"{experiment_name}_best.pt"

    model, config, checkpoint = load_trained_model(
        model_name=args.model_name,
        experiment_name=experiment_name,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
    )
    index_rows = load_npz_index(args.index_path)
    test_rows = get_test_samples(index_rows, num_samples=args.num_samples)
    if not test_rows:
        raise ValueError(f"No test samples found in index: {args.index_path}")

    report_lines = [
        f"# XAI Handoff Check: {experiment_name}",
        "",
        f"- model_name: {args.model_name}",
        f"- experiment_name: {experiment_name}",
        f"- checkpoint_path: {checkpoint_path}",
        f"- checkpoint_epoch: {checkpoint.get('epoch')}",
        f"- index_path: {args.index_path}",
        f"- device: {device}",
        f"- samples_checked: {len(test_rows)}",
        "",
    ]

    print(f"model_name: {args.model_name}", flush=True)
    print(f"experiment_name: {experiment_name}", flush=True)
    print(f"checkpoint path: {checkpoint_path}", flush=True)

    for sample_index, row in enumerate(test_rows, start=1):
        sample = load_npz_sample(row["npz_path"])
        batch = prepare_xai_batch(sample, device=device)
        with torch.no_grad():
            output = model(
                batch["image_t1"],
                batch["image_t2"],
                batch["tabular"],
                return_features=True,
            )

        prediction = extract_prediction(output)
        metadata_keys = sorted(key for key in sample if key not in {"image_t1", "image_t2", "tabular"})

        print(f"\nSample {sample_index}", flush=True)
        print(f"  image_t1 shape: {tuple(batch['image_t1'].shape)}", flush=True)
        print(f"  image_t2 shape: {tuple(batch['image_t2'].shape)}", flush=True)
        print(f"  tabular shape: {tuple(batch['tabular'].shape)}", flush=True)
        print(f"  target change_ratio: {sample.get('change_ratio')}", flush=True)
        print(f"  metadata keys: {metadata_keys}", flush=True)
        print(f"  output type / keys: {describe_output(output)}", flush=True)
        print(f"  prediction value: {prediction}", flush=True)

        report_lines.extend(
            [
                f"## Sample {sample_index}",
                "",
                f"- patch_id: {sample.get('patch_id')}",
                f"- npz_path: {row.get('npz_path')}",
                f"- image_t1 shape: {tuple(batch['image_t1'].shape)}",
                f"- image_t2 shape: {tuple(batch['image_t2'].shape)}",
                f"- tabular shape: {tuple(batch['tabular'].shape)}",
                f"- target change_ratio: {sample.get('change_ratio')}",
                f"- metadata keys: {metadata_keys}",
                f"- output type / keys: {describe_output(output)}",
                f"- prediction value: {prediction}",
                "",
            ]
        )

    report_path = save_report(report_lines, experiment_name)
    print(f"\nReport saved: {report_path}", flush=True)


def extract_prediction(output: Any) -> float | None:
    """Extract scalar regression prediction from model output."""
    outputs = output.get("outputs") if isinstance(output, dict) and "outputs" in output else output
    if isinstance(outputs, torch.Tensor):
        return float(outputs.detach().cpu().view(-1)[0].item())
    if isinstance(outputs, dict) and "change_ratio_pred" in outputs:
        return float(outputs["change_ratio_pred"].detach().cpu().view(-1)[0].item())
    return None


def describe_output(output: Any) -> str:
    """Describe model output type and keys for handoff sanity check."""
    if isinstance(output, dict):
        return f"dict keys={sorted(output.keys())}"
    return type(output).__name__


def save_report(lines: list[str], experiment_name: str) -> Path:
    """Save markdown handoff check report."""
    report_dir = Path(DEFAULT_XAI_OUTPUT_DIR) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"handoff_check_report_{experiment_name}.md"
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


if __name__ == "__main__":
    main()
