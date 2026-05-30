"""Run XAI-07 Grad-CAM for CNN and ConvNeXt models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.xai.gradcam import compute_siamese_gradcam, find_last_conv2d, is_gradcam_supported  # noqa: E402
from src.xai.handoff_loader import load_npz_sample, load_trained_model, prepare_xai_batch  # noqa: E402
from src.xai.sample_selector import load_csv_rows, safe_float, save_csv_rows  # noqa: E402
from src.xai.visualization import make_rgb_preview, save_overlay_png  # noqa: E402


SUMMARY_COLUMNS = [
    "model_name",
    "experiment_name",
    "patch_id",
    "pair_id",
    "district",
    "split",
    "change_class",
    "pair_type",
    "time_gap_group",
    "target_image",
    "prediction",
    "true_change_ratio",
    "absolute_error",
    "target_layer",
    "max_cam",
    "mean_cam",
    "overlay_path",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="XAI-07 Grad-CAM.")
    parser.add_argument("--model_name", required=True, choices=["cnn", "convnext", "swin", "maxvit"])
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--selected_samples", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--output_dir", default="outputs/xai")
    return parser.parse_args()


def main() -> None:
    """Run Grad-CAM for selected samples."""
    args = parse_args()
    output_root = Path(args.output_dir)
    report_path = output_root / "reports" / f"gradcam_report_{args.experiment_name}.md"
    if not is_gradcam_supported(args.model_name):
        message = (
            f"Grad-CAM is supported only for cnn and convnext. "
            f"Model `{args.model_name}` should use occlusion sensitivity instead."
        )
        print(message, flush=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(f"# Grad-CAM Report: {args.experiment_name}\n\n{message}\n", encoding="utf-8")
        return

    device = resolve_device(args.device)
    selected_path = Path(args.selected_samples) if args.selected_samples else (
        output_root / "selected_samples" / f"xai_selected_samples_{args.experiment_name}.csv"
    )
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else (
        Path("checkpoints") / f"{args.experiment_name}_best.pt"
    )
    print(f"model_name: {args.model_name}", flush=True)
    print(f"experiment_name: {args.experiment_name}", flush=True)
    print(f"checkpoint: {checkpoint_path}", flush=True)
    print(f"selected samples: {selected_path}", flush=True)
    print(f"device: {device}", flush=True)

    model, _, _ = load_trained_model(args.model_name, args.experiment_name, device=device)
    target_layer_name, target_layer = find_last_conv2d(model)
    print(f"Grad-CAM target layer: {target_layer_name}", flush=True)
    selected_rows = load_csv_rows(selected_path)[: max(0, int(args.num_samples))]
    summary_rows: list[dict[str, object]] = []

    for index, row in enumerate(selected_rows, start=1):
        print(f"Grad-CAM sample {index}/{len(selected_rows)}: {row.get('patch_id')}", flush=True)
        sample = load_npz_sample(row["npz_path"])
        batch = prepare_xai_batch(sample, device=device)
        result = compute_siamese_gradcam(
            model,
            batch["image_t1"],
            batch["image_t2"],
            batch["tabular"],
            target_layer=target_layer,
            target_layer_name=target_layer_name,
        )
        sample_dir = output_root / "gradcam" / args.experiment_name / safe_path_name(str(row.get("patch_id", "sample")))
        t1_overlay = sample_dir / "t1_gradcam_overlay.png"
        t2_overlay = sample_dir / "t2_gradcam_overlay.png"
        save_overlay_png(make_rgb_preview(batch["image_t1"]), result.t1_cam, t1_overlay)
        save_overlay_png(make_rgb_preview(batch["image_t2"]), result.t2_cam, t2_overlay)
        true_change_ratio = safe_float(row.get("y_true_change_ratio"), safe_float(sample.get("change_ratio"), None))
        absolute_error = abs(result.prediction - true_change_ratio) if true_change_ratio is not None else ""
        summary_rows.extend(
            [
                build_summary_row(args, row, result.prediction, true_change_ratio, absolute_error, "t1", result.t1_cam, target_layer_name, t1_overlay),
                build_summary_row(args, row, result.prediction, true_change_ratio, absolute_error, "t2", result.t2_cam, target_layer_name, t2_overlay),
            ]
        )

    summary_path = output_root / "gradcam" / f"gradcam_summary_{args.experiment_name}.csv"
    save_csv_rows(summary_rows, summary_path, fieldnames=SUMMARY_COLUMNS)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(args, checkpoint_path, target_layer_name, len(selected_rows), summary_path, summary_rows),
        encoding="utf-8",
    )
    print(f"summary CSV: {summary_path}", flush=True)
    print(f"report: {report_path}", flush=True)


def build_summary_row(
    args: argparse.Namespace,
    row: dict[str, object],
    prediction: float,
    true_change_ratio: float | None,
    absolute_error: float | str,
    target_image: str,
    cam: torch.Tensor,
    target_layer_name: str,
    overlay_path: Path,
) -> dict[str, object]:
    """Build one Grad-CAM summary row."""
    return {
        "model_name": args.model_name,
        "experiment_name": args.experiment_name,
        "patch_id": row.get("patch_id", ""),
        "pair_id": row.get("pair_id", ""),
        "district": row.get("district", ""),
        "split": row.get("split", ""),
        "change_class": row.get("change_class", ""),
        "pair_type": row.get("pair_type", ""),
        "time_gap_group": row.get("time_gap_group", ""),
        "target_image": target_image,
        "prediction": prediction,
        "true_change_ratio": true_change_ratio if true_change_ratio is not None else "",
        "absolute_error": absolute_error,
        "target_layer": target_layer_name,
        "max_cam": float(cam.max().item()),
        "mean_cam": float(cam.mean().item()),
        "overlay_path": str(overlay_path),
    }


def build_report(
    args: argparse.Namespace,
    checkpoint_path: Path,
    target_layer_name: str,
    sample_count: int,
    summary_path: Path,
    summary_rows: list[dict[str, object]],
) -> str:
    """Build Markdown Grad-CAM report."""
    lines = [
        f"# Grad-CAM Report: {args.experiment_name}",
        "",
        f"- model_name: {args.model_name}",
        f"- experiment_name: {args.experiment_name}",
        f"- checkpoint used: {checkpoint_path}",
        f"- samples explained: {sample_count}",
        f"- target convolution layer: {target_layer_name}",
        f"- summary CSV: {summary_path}",
        "",
        "## Method",
        "",
        "Grad-CAM uses gradients at the last convolutional layer to highlight spatial regions that influence the scalar change-ratio prediction. The shared Siamese image encoder is hooked separately for image_t1 and image_t2.",
        "",
        "## Generated Overlays",
        "",
    ]
    for row in summary_rows[:20]:
        lines.append(f"- {row.get('patch_id')} {row.get('target_image')}: {row.get('overlay_path')}")
    return "\n".join(lines) + "\n"


def safe_path_name(value: str) -> str:
    """Create a Windows-friendly folder name."""
    invalid = '<>:"/\\|?*'
    cleaned = "".join("_" if char in invalid else char for char in value)
    return cleaned.strip() or "sample"


def resolve_device(requested: str) -> torch.device:
    """Resolve device with safe CUDA fallback."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


if __name__ == "__main__":
    main()
