"""Run XAI-04 occlusion sensitivity heatmaps for selected samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.xai.handoff_loader import (  # noqa: E402
    build_model_from_config,
    load_model_config,
    load_npz_sample,
    load_trained_model,
    prepare_xai_batch,
)
from src.xai.occlusion_sensitivity import compute_occlusion_map  # noqa: E402
from src.xai.sample_selector import load_csv_rows, safe_float, save_csv_rows  # noqa: E402
from src.xai.visualization import (  # noqa: E402
    make_rgb_preview,
    save_heatmap_png,
    save_overlay_png,
    save_rgb_png,
    save_side_by_side_xai,
)


SCORE_COLUMNS = [
    "model_name",
    "experiment_name",
    "checkpoint_path",
    "patch_id",
    "pair_id",
    "district",
    "split",
    "change_class",
    "pair_type",
    "time_gap_group",
    "true_change_ratio",
    "full_prediction",
    "absolute_error",
    "target_image",
    "max_importance",
    "mean_importance",
    "heatmap_path",
    "overlay_path",
    "side_by_side_path",
]


def parse_args() -> argparse.Namespace:
    """Parse occlusion sensitivity arguments."""
    parser = argparse.ArgumentParser(description="XAI-04 occlusion sensitivity heatmaps.")
    parser.add_argument("--model_name", required=True, choices=["cnn", "convnext", "swin", "maxvit"])
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--index_path", default="data/npz/final_npz_index.csv")
    parser.add_argument("--selected_samples", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--mask_value", default="zero", choices=["zero", "channel_mean"])
    parser.add_argument("--output_dir", default="outputs/xai")
    return parser.parse_args()


def main() -> None:
    """Run occlusion sensitivity for selected samples and save outputs."""
    args = parse_args()
    device = resolve_device(args.device)
    selected_path = Path(args.selected_samples) if args.selected_samples else (
        Path(args.output_dir) / "selected_samples" / f"xai_selected_samples_{args.experiment_name}.csv"
    )
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else (
        Path("checkpoints") / f"{args.experiment_name}_best.pt"
    )

    print(f"model_name: {args.model_name}", flush=True)
    print(f"experiment_name: {args.experiment_name}", flush=True)
    print(f"checkpoint: {checkpoint_path}", flush=True)
    print(f"selected samples: {selected_path}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"patch_size: {args.patch_size}, stride: {args.stride}, mask_value: {args.mask_value}", flush=True)

    model = load_model(args.model_name, args.experiment_name, checkpoint_path, device)
    selected_rows = load_csv_rows(selected_path)[: max(0, int(args.num_samples))]
    score_rows: list[dict[str, object]] = []

    for index, row in enumerate(selected_rows, start=1):
        print(f"Occlusion sample {index}/{len(selected_rows)}: {row.get('patch_id')}", flush=True)
        sample = load_npz_sample(row["npz_path"])
        batch = prepare_xai_batch(sample, device=device)
        t1_heatmap, full_prediction = compute_occlusion_map(
            model,
            batch["image_t1"],
            batch["image_t2"],
            batch["tabular"],
            target_image="t1",
            patch_size=args.patch_size,
            stride=args.stride,
            mask_value=args.mask_value,
        )
        t2_heatmap, _ = compute_occlusion_map(
            model,
            batch["image_t1"],
            batch["image_t2"],
            batch["tabular"],
            target_image="t2",
            patch_size=args.patch_size,
            stride=args.stride,
            mask_value=args.mask_value,
        )
        paths = save_sample_images(args.output_dir, args.experiment_name, row, batch, t1_heatmap, t2_heatmap)
        true_change_ratio = safe_float(row.get("y_true_change_ratio"), safe_float(sample.get("change_ratio"), 0.0))
        absolute_error = abs(float(full_prediction) - float(true_change_ratio)) if true_change_ratio is not None else ""
        score_rows.extend(
            build_score_rows(
                args=args,
                row=row,
                checkpoint_path=checkpoint_path,
                full_prediction=full_prediction,
                true_change_ratio=true_change_ratio,
                absolute_error=absolute_error,
                t1_heatmap=t1_heatmap,
                t2_heatmap=t2_heatmap,
                paths=paths,
            )
        )

    score_path = Path(args.output_dir) / "occlusion" / f"occlusion_scores_{args.experiment_name}.csv"
    summary_path = Path(args.output_dir) / "reports" / f"occlusion_summary_{args.experiment_name}.csv"
    report_path = Path(args.output_dir) / "reports" / f"occlusion_report_{args.experiment_name}.md"
    summary_rows = summarize_scores(score_rows)
    save_csv_rows(score_rows, score_path, fieldnames=SCORE_COLUMNS)
    save_csv_rows(summary_rows, summary_path, fieldnames=["metric", "group", "value"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(args, checkpoint_path, len(selected_rows), score_path, summary_rows, score_rows),
        encoding="utf-8",
    )

    print(f"occlusion scores: {score_path}", flush=True)
    print(f"summary: {summary_path}", flush=True)
    print(f"report: {report_path}", flush=True)


def load_model(model_name: str, experiment_name: str, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load default or custom checkpoint."""
    if checkpoint_path == Path("checkpoints") / f"{experiment_name}_best.pt":
        model, _, _ = load_trained_model(model_name, experiment_name, checkpoint_dir="checkpoints", device=device)
        return model
    config = load_model_config(experiment_name, checkpoint_dir="checkpoints")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model_from_config(model_name, config, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def save_sample_images(
    output_dir: str | Path,
    experiment_name: str,
    row: dict[str, object],
    batch: dict[str, torch.Tensor],
    t1_heatmap: torch.Tensor,
    t2_heatmap: torch.Tensor,
) -> dict[str, Path]:
    """Save RGB previews, heatmaps, overlays, and side-by-side figure."""
    patch_id = safe_path_name(str(row.get("patch_id", "sample")))
    sample_dir = Path(output_dir) / "occlusion" / experiment_name / patch_id
    t1_rgb = make_rgb_preview(batch["image_t1"])
    t2_rgb = make_rgb_preview(batch["image_t2"])
    paths = {
        "t1_rgb": sample_dir / "t1_rgb.png",
        "t2_rgb": sample_dir / "t2_rgb.png",
        "t1_heatmap": sample_dir / "t1_occlusion_heatmap.png",
        "t2_heatmap": sample_dir / "t2_occlusion_heatmap.png",
        "t1_overlay": sample_dir / "t1_occlusion_overlay.png",
        "t2_overlay": sample_dir / "t2_occlusion_overlay.png",
        "side_by_side": sample_dir / "side_by_side_xai.png",
    }
    save_rgb_png(t1_rgb, paths["t1_rgb"])
    save_rgb_png(t2_rgb, paths["t2_rgb"])
    save_heatmap_png(t1_heatmap, paths["t1_heatmap"])
    save_heatmap_png(t2_heatmap, paths["t2_heatmap"])
    save_overlay_png(t1_rgb, t1_heatmap, paths["t1_overlay"])
    save_overlay_png(t2_rgb, t2_heatmap, paths["t2_overlay"])
    save_side_by_side_xai(t1_rgb, t2_rgb, t1_heatmap, t2_heatmap, paths["side_by_side"], title=str(row.get("patch_id", "")))
    return paths


def build_score_rows(
    args: argparse.Namespace,
    row: dict[str, object],
    checkpoint_path: Path,
    full_prediction: float,
    true_change_ratio: float | None,
    absolute_error: float | str,
    t1_heatmap: torch.Tensor,
    t2_heatmap: torch.Tensor,
    paths: dict[str, Path],
) -> list[dict[str, object]]:
    """Build two score rows, one for t1 and one for t2."""
    base = {
        "model_name": args.model_name,
        "experiment_name": args.experiment_name,
        "checkpoint_path": str(checkpoint_path),
        "patch_id": row.get("patch_id", ""),
        "pair_id": row.get("pair_id", ""),
        "district": row.get("district", ""),
        "split": row.get("split", ""),
        "change_class": row.get("change_class", ""),
        "pair_type": row.get("pair_type", ""),
        "time_gap_group": row.get("time_gap_group", ""),
        "true_change_ratio": true_change_ratio if true_change_ratio is not None else "",
        "full_prediction": full_prediction,
        "absolute_error": absolute_error,
    }
    return [
        {
            **base,
            "target_image": "t1",
            "max_importance": float(t1_heatmap.max().item()),
            "mean_importance": float(t1_heatmap.mean().item()),
            "heatmap_path": str(paths["t1_heatmap"]),
            "overlay_path": str(paths["t1_overlay"]),
            "side_by_side_path": str(paths["side_by_side"]),
        },
        {
            **base,
            "target_image": "t2",
            "max_importance": float(t2_heatmap.max().item()),
            "mean_importance": float(t2_heatmap.mean().item()),
            "heatmap_path": str(paths["t2_heatmap"]),
            "overlay_path": str(paths["t2_overlay"]),
            "side_by_side_path": str(paths["side_by_side"]),
        },
    ]


def summarize_scores(score_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize occlusion importance scores."""
    t1_rows = [row for row in score_rows if row.get("target_image") == "t1"]
    t2_rows = [row for row in score_rows if row.get("target_image") == "t2"]
    t1_mean = average([safe_float(row.get("mean_importance")) for row in t1_rows])
    t2_mean = average([safe_float(row.get("mean_importance")) for row in t2_rows])
    t1_max = average([safe_float(row.get("max_importance")) for row in t1_rows])
    t2_max = average([safe_float(row.get("max_importance")) for row in t2_rows])
    more_influential = "t1" if (t1_mean or 0.0) >= (t2_mean or 0.0) else "t2"
    return [
        {"metric": "number_of_samples", "group": "all", "value": len(t1_rows)},
        {"metric": "average_t1_mean_importance", "group": "all", "value": t1_mean},
        {"metric": "average_t2_mean_importance", "group": "all", "value": t2_mean},
        {"metric": "average_t1_max_importance", "group": "all", "value": t1_max},
        {"metric": "average_t2_max_importance", "group": "all", "value": t2_max},
        {"metric": "more_spatially_influential", "group": "all", "value": more_influential},
    ]


def build_report(
    args: argparse.Namespace,
    checkpoint_path: Path,
    sample_count: int,
    score_path: Path,
    summary_rows: list[dict[str, object]],
    score_rows: list[dict[str, object]],
) -> str:
    """Build Markdown occlusion report."""
    summary = {(row["metric"], row["group"]): row["value"] for row in summary_rows}
    influential = summary.get(("more_spatially_influential", "all"), "")
    lines = [
        f"# Occlusion Sensitivity Report: {args.experiment_name}",
        "",
        f"- model_name: {args.model_name}",
        f"- experiment_name: {args.experiment_name}",
        f"- checkpoint used: {checkpoint_path}",
        f"- number of samples: {sample_count}",
        f"- patch_size: {args.patch_size}",
        f"- stride: {args.stride}",
        f"- mask_value: {args.mask_value}",
        f"- scores_csv: {score_path}",
        "",
        "## Method",
        "",
        "A square mask is moved over image_t1 or image_t2. For each location, the selected image region is replaced by the mask value and the model prediction is recomputed. Importance is the absolute change from the full prediction.",
        "",
        "RGB previews use the first three image channels with robust percentile normalization because exact display band order may be uncertain.",
        "",
        "## Summary Interpretation",
        "",
        f"On average, `{influential}` was more spatially influential for the selected samples.",
        "",
        "## Summary Metrics",
        "",
    ]
    for row in summary_rows:
        lines.append(f"- {row['metric']} / {row['group']}: {row['value']}")
    lines.extend(["", "## Generated Images", ""])
    for row in score_rows[:20]:
        lines.append(f"- {row['patch_id']} {row['target_image']}: {row['overlay_path']}")
    return "\n".join(lines) + "\n"


def average(values: list[float | None]) -> float:
    """Average non-None values."""
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else 0.0


def safe_path_name(value: str) -> str:
    """Create a Windows-friendly folder name."""
    invalid = '<>:"/\\|?*'
    cleaned = "".join("_" if char in invalid else char for char in value)
    return cleaned.strip() or "sample"


def resolve_device(requested: str) -> torch.device:
    """Resolve requested torch device."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


if __name__ == "__main__":
    main()
