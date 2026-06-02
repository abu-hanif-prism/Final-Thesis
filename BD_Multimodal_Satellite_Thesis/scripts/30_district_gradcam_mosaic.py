"""Create district-level Grad-CAM mosaics from selected patch predictions."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.xai.gradcam import compute_siamese_gradcam, find_last_conv2d, is_gradcam_supported
from src.xai.handoff_loader import load_npz_sample, load_trained_model, prepare_xai_batch


PATCH_PIXEL_SIZE = 128
UNSUPPORTED_MESSAGE = (
    "District Grad-CAM is supported only for CNN/ConvNeXt. "
    "Use district occlusion mosaic for Swin/MaxViT."
)
COORDINATE_PATTERN = re.compile(r"_x(?P<x>\d+)_y(?P<y>\d+)(?:$|_)")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build district-level Grad-CAM XAI mosaics.")
    parser.add_argument("--model_name", default="cnn", choices=["cnn", "convnext", "swin", "maxvit"])
    parser.add_argument("--experiment_name", default="cnn_regression")
    parser.add_argument("--index_path", default="data/npz/final_npz_index.csv")
    parser.add_argument("--prediction_csv", default=None)
    parser.add_argument("--district", required=True)
    parser.add_argument("--pair_id", required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--sort_by",
        default="predicted_change_ratio",
        choices=["predicted_change_ratio", "prediction", "absolute_error"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target_image", default="t2", choices=["t1", "t2"])
    parser.add_argument("--output_dir", default="outputs/xai/district_gradcam")
    return parser.parse_args()


def default_prediction_csv(experiment_name: str) -> Path:
    """Return default evaluation prediction CSV path."""
    return Path("outputs/evaluation") / experiment_name / "test" / "predictions.csv"


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read CSV rows with the standard library."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def save_csv_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Save dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_patch_coordinates(patch_id: str) -> tuple[int, int]:
    """Parse x/y patch coordinates from patch_id."""
    match = COORDINATE_PATTERN.search(str(patch_id))
    if not match:
        raise ValueError(f"Could not parse x/y coordinates from patch_id: {patch_id}")
    return int(match.group("x")), int(match.group("y"))


def safe_float(value: Any) -> float | None:
    """Convert a value to a finite float if possible."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def prediction_value(row: dict[str, Any]) -> float | None:
    """Return prediction value from known prediction column names."""
    for column in ("y_pred_change_ratio", "predicted_change_ratio", "prediction", "y_pred", "pred"):
        value = safe_float(row.get(column))
        if value is not None:
            return value
    return None


def true_value(row: dict[str, Any]) -> float | None:
    """Return true change ratio from known target column names."""
    for column in ("y_true_change_ratio", "true_change_ratio", "change_ratio", "target"):
        value = safe_float(row.get(column))
        if value is not None:
            return value
    return None


def absolute_error_value(row: dict[str, Any]) -> float | None:
    """Return absolute error from CSV or compute it from prediction and truth."""
    for column in ("abs_error", "absolute_error"):
        value = safe_float(row.get(column))
        if value is not None:
            return value
    pred = prediction_value(row)
    true = true_value(row)
    if pred is None or true is None:
        return None
    return abs(pred - true)


def sort_value(row: dict[str, Any], sort_by: str) -> float:
    """Return sortable score, placing missing values last."""
    if sort_by == "absolute_error":
        value = absolute_error_value(row)
    else:
        value = prediction_value(row)
    return value if value is not None else float("-inf")


def filter_and_select_predictions(
    prediction_rows: list[dict[str, str]],
    district: str,
    pair_id: str,
    top_k: int,
    sort_by: str,
) -> list[dict[str, Any]]:
    """Filter prediction CSV rows and select top-k patches."""
    filtered: list[dict[str, Any]] = []
    for row in prediction_rows:
        if row.get("district") != district or row.get("pair_id") != pair_id:
            continue
        x_coord, y_coord = parse_patch_coordinates(row.get("patch_id", ""))
        parsed = dict(row)
        parsed["x"] = x_coord
        parsed["y"] = y_coord
        parsed["predicted_change_ratio"] = prediction_value(row)
        parsed["true_change_ratio"] = true_value(row)
        parsed["absolute_error"] = absolute_error_value(row)
        filtered.append(parsed)

    selected = sorted(filtered, key=lambda row: sort_value(row, sort_by), reverse=True)
    return selected[: max(0, int(top_k))]


def index_by_patch_id(index_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Map NPZ index rows by patch_id."""
    return {row["patch_id"]: row for row in index_rows if row.get("patch_id")}


def resolve_npz_path(path_value: str, index_path: Path) -> Path:
    """Resolve absolute or relative NPZ paths from the index CSV."""
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    project_candidate = Path.cwd() / path
    if project_candidate.exists():
        return project_candidate
    return index_path.parent / path


def resolve_device(requested: str) -> torch.device:
    """Resolve requested torch device, falling back when CUDA is unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def safe_name(value: str) -> str:
    """Create a file-system-safe name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "value"


def resolve_run_dir(base_dir: Path) -> Path:
    """Avoid overwriting previous district Grad-CAM outputs."""
    expected_names = {
        "selected_patches.csv",
        "district_gradcam_mosaic_t1.png",
        "district_gradcam_mosaic_t2.png",
        "district_gradcam_selected_prediction_map.png",
        "district_gradcam_report.md",
    }
    if not base_dir.exists() or not any((base_dir / name).exists() for name in expected_names):
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    index = 2
    while True:
        candidate = base_dir.with_name(f"{base_dir.name}_run{index}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            print(f"Existing outputs found. Writing to non-overwriting run directory: {candidate}", flush=True)
            return candidate
        index += 1


def normalize_image_for_display(image: np.ndarray) -> np.ndarray:
    """Normalize first three channels of a CHW image to RGB display range."""
    rgb = np.asarray(image[:3], dtype=np.float32)
    normalized = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(rgb.shape[0]):
        values = rgb[channel]
        low, high = np.percentile(values, [2, 98])
        if high > low:
            normalized[channel] = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.moveaxis(normalized, 0, -1)


def normalized_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normalize heatmap to 0-1 for display."""
    heat = np.asarray(heatmap, dtype=np.float32)
    min_value = float(np.nanmin(heat))
    max_value = float(np.nanmax(heat))
    if max_value <= min_value:
        return np.zeros_like(heat, dtype=np.float32)
    return (heat - min_value) / (max_value - min_value)


def save_patch_heatmap(heatmap: np.ndarray, path: Path, title: str) -> None:
    """Save one patch-level Grad-CAM heatmap with matplotlib."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(heatmap, cmap="inferno")
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_patch_overlay(rgb: np.ndarray, heatmap: np.ndarray, path: Path, title: str) -> None:
    """Save one patch-level RGB/Grad-CAM overlay with matplotlib."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(rgb)
    ax.imshow(normalized_heatmap(heatmap), cmap="inferno", alpha=0.45)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def stitch_heatmaps(selected_rows: list[dict[str, Any]], target_image: str) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Stitch selected 128x128 Grad-CAM maps into a district coordinate mosaic."""
    key = f"{target_image}_gradcam_array"
    x_values = [int(row["x"]) for row in selected_rows if key in row]
    y_values = [int(row["y"]) for row in selected_rows if key in row]
    if not x_values or not y_values:
        raise ValueError(f"No Grad-CAM heatmaps available for {target_image}.")

    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    width = (max_x - min_x) + PATCH_PIXEL_SIZE
    height = (max_y - min_y) + PATCH_PIXEL_SIZE
    mosaic = np.full((height, width), np.nan, dtype=np.float32)

    for row in selected_rows:
        if key not in row:
            continue
        x_start = int(row["x"]) - min_x
        y_start = int(row["y"]) - min_y
        mosaic[y_start : y_start + PATCH_PIXEL_SIZE, x_start : x_start + PATCH_PIXEL_SIZE] = row[key]
    return mosaic, (min_x, max_x + PATCH_PIXEL_SIZE, max_y + PATCH_PIXEL_SIZE, min_y)


def save_mosaic_png(mosaic: np.ndarray, extent: tuple[int, int, int, int], path: Path, title: str) -> None:
    """Save a district-level Grad-CAM mosaic."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    masked = np.ma.masked_invalid(mosaic)
    fig_width = max(8.0, min(18.0, mosaic.shape[1] / 180.0))
    fig_height = max(6.0, min(16.0, mosaic.shape[0] / 180.0))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(masked, cmap="inferno", extent=extent, interpolation="nearest", aspect="equal")
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Grad-CAM intensity")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_selected_prediction_map(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    """Save selected patch prediction values as a coordinate grid."""
    import matplotlib.pyplot as plt

    x_values = sorted({int(row["x"]) for row in rows})
    y_values = sorted({int(row["y"]) for row in rows})
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float32)
    for row in rows:
        value = safe_float(row.get("predicted_change_ratio"))
        if value is not None:
            grid[y_index[int(row["y"])], x_index[int(row["x"])]] = value

    x_edges = x_values + [x_values[-1] + PATCH_PIXEL_SIZE]
    y_edges = y_values + [y_values[-1] + PATCH_PIXEL_SIZE]
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(x_edges, y_edges, grid, shading="flat", cmap="viridis")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="predicted change ratio")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def mean(values: list[float | None]) -> float | None:
    """Average non-missing values."""
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def fmt(value: float | None) -> str:
    """Format optional float for reports."""
    return "not available" if value is None else f"{value:.6f}"


def write_report(
    path: Path,
    args: argparse.Namespace,
    run_dir: Path,
    selected_rows: list[dict[str, Any]],
    target_layer_name: str,
    mosaic_path: Path,
    prediction_map_path: Path,
) -> None:
    """Write district Grad-CAM Markdown report."""
    predicted_values = [safe_float(row.get("predicted_change_ratio")) for row in selected_rows]
    true_values = [safe_float(row.get("true_change_ratio")) for row in selected_rows]
    error_values = [safe_float(row.get("absolute_error")) for row in selected_rows]
    lines = [
        "# District Grad-CAM Mosaic Report",
        "",
        f"- model_name: `{args.model_name}`",
        f"- experiment_name: `{args.experiment_name}`",
        f"- district: `{args.district}`",
        f"- pair_id: `{args.pair_id}`",
        f"- selected patches: {len(selected_rows)}",
        f"- target image: `{args.target_image}`",
        f"- target convolution layer: `{target_layer_name}`",
        f"- mean selected predicted change_ratio: {fmt(mean(predicted_values))}",
        f"- mean selected true change_ratio: {fmt(mean(true_values))}",
        f"- mean selected absolute error: {fmt(mean(error_values))}",
        f"- output folder: `{run_dir}`",
        "",
        "## Generated Files",
        "",
        f"- selected patches: `{run_dir / 'selected_patches.csv'}`",
        f"- selected prediction map: `{prediction_map_path}`",
        f"- district Grad-CAM mosaic: `{mosaic_path}`",
        "",
        "## Interpretation Notes",
        "",
        "- This is not full-image training and not direct full-scene inference.",
        "- The trained model is patch-based and receives 13x128x128 image_t1/image_t2 tensors plus tabular features.",
        "- Grad-CAM was computed on selected 128x128 patches.",
        "- The patch-level heatmaps were stitched into a district-level coordinate mosaic using x/y values parsed from patch_id.",
        "- Brighter regions indicate areas that contributed more strongly to the CNN/ConvNeXt output.",
        "- Grad-CAM is not used for MaxViT/Swin in this script.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run selected patch Grad-CAM and stitch results into a district mosaic."""
    args = parse_args()
    if not is_gradcam_supported(args.model_name):
        print(UNSUPPORTED_MESSAGE, flush=True)
        return

    prediction_csv = Path(args.prediction_csv) if args.prediction_csv else default_prediction_csv(args.experiment_name)
    index_path = Path(args.index_path)
    device = resolve_device(args.device)

    print(f"model_name: {args.model_name}", flush=True)
    print(f"experiment_name: {args.experiment_name}", flush=True)
    print(f"prediction_csv: {prediction_csv}", flush=True)
    print(f"index_path: {index_path}", flush=True)
    print(f"district: {args.district}", flush=True)
    print(f"pair_id: {args.pair_id}", flush=True)
    print(f"target_image: {args.target_image}", flush=True)
    print(f"device: {device}", flush=True)

    prediction_rows = read_csv_rows(prediction_csv)
    selected_rows = filter_and_select_predictions(
        prediction_rows,
        args.district,
        args.pair_id,
        args.top_k,
        args.sort_by,
    )
    if not selected_rows:
        raise ValueError(
            f"No prediction rows found for district={args.district!r}, pair_id={args.pair_id!r}."
        )

    index_rows = read_csv_rows(index_path)
    npz_index = index_by_patch_id(index_rows)
    for row in selected_rows:
        index_row = npz_index.get(str(row["patch_id"]))
        if index_row is None:
            raise KeyError(f"Patch {row['patch_id']} was not found in NPZ index: {index_path}")
        row["npz_path"] = str(resolve_npz_path(index_row["npz_path"], index_path))
        row["index_split"] = index_row.get("split", "")
        row["index_change_class"] = index_row.get("change_class", "")

    base_run_dir = (
        Path(args.output_dir)
        / safe_name(args.experiment_name)
        / f"{safe_name(args.district)}_{safe_name(args.pair_id)}"
    )
    run_dir = resolve_run_dir(base_run_dir)
    patches_dir = run_dir / "patch_gradcam"

    print(f"Loading trained model from checkpoints/{args.experiment_name}_best.pt", flush=True)
    model, _, _ = load_trained_model(args.model_name, args.experiment_name, checkpoint_dir="checkpoints", device=device)
    target_layer_name, target_layer = find_last_conv2d(model)
    print(f"Grad-CAM target layer: {target_layer_name}", flush=True)

    for index, row in enumerate(selected_rows, start=1):
        patch_id = str(row["patch_id"])
        print(f"Grad-CAM {index}/{len(selected_rows)}: {patch_id}", flush=True)
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
        cam_tensor = result.t1_cam if args.target_image == "t1" else result.t2_cam
        cam_np = cam_tensor.detach().cpu().numpy().astype(np.float32)
        row[f"{args.target_image}_gradcam_array"] = cam_np
        row["gradcam_prediction"] = result.prediction
        row["gradcam_mean"] = float(np.mean(cam_np))
        row["gradcam_max"] = float(np.max(cam_np))

        image_np = np.asarray(sample["image_t1"] if args.target_image == "t1" else sample["image_t2"])
        rgb = normalize_image_for_display(image_np)
        patch_dir = patches_dir / safe_name(patch_id)
        heatmap_path = patch_dir / f"{args.target_image}_gradcam_heatmap.png"
        overlay_path = patch_dir / f"{args.target_image}_gradcam_overlay.png"
        save_patch_heatmap(cam_np, heatmap_path, f"{args.target_image} Grad-CAM\n{patch_id}")
        save_patch_overlay(rgb, cam_np, overlay_path, f"{args.target_image} Grad-CAM overlay\n{patch_id}")
        row["gradcam_heatmap_path"] = str(heatmap_path)
        row["gradcam_overlay_path"] = str(overlay_path)

    selected_csv_rows = [
        {key: value for key, value in row.items() if not key.endswith("_gradcam_array")}
        for row in selected_rows
    ]
    selected_patches_path = run_dir / "selected_patches.csv"
    save_csv_rows(selected_csv_rows, selected_patches_path)

    title_base = f"{args.experiment_name} | {args.district} | {args.pair_id} | patches={len(selected_rows)}"
    prediction_map_path = run_dir / "district_gradcam_selected_prediction_map.png"
    save_selected_prediction_map(selected_rows, prediction_map_path, f"Selected predicted change ratio\n{title_base}")

    mosaic, extent = stitch_heatmaps(selected_rows, args.target_image)
    mosaic_path = run_dir / f"district_gradcam_mosaic_{args.target_image}.png"
    save_mosaic_png(mosaic, extent, mosaic_path, f"{args.target_image} district Grad-CAM mosaic\n{title_base}")

    report_path = run_dir / "district_gradcam_report.md"
    write_report(report_path, args, run_dir, selected_rows, target_layer_name, mosaic_path, prediction_map_path)

    print("District Grad-CAM mosaic created.", flush=True)
    print(f"  selected_patches: {selected_patches_path}", flush=True)
    print(f"  prediction_map: {prediction_map_path}", flush=True)
    print(f"  mosaic: {mosaic_path}", flush=True)
    print(f"  report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
