"""Create district-level patch-grid prediction maps from prediction CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re
from typing import Any


DEFAULT_EXPERIMENT_NAME = "maxvit_regression_stable"
DEFAULT_OUTPUT_DIR = Path("outputs/reports/district_prediction_maps")
PATCH_SIZE = 128
COORDINATE_PATTERN = re.compile(r"_x(?P<x>\d+)_y(?P<y>\d+)(?:$|_)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot patch-grid prediction maps for one district and temporal pair."
    )
    parser.add_argument("--experiment_name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--prediction_csv", default=None)
    parser.add_argument("--district", default=None)
    parser.add_argument("--pair_id", default=None)
    parser.add_argument("--top_pairs", type=int, default=20)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--list_pairs", action="store_true")
    return parser.parse_args()


def default_prediction_csv(experiment_name: str) -> Path:
    """Return default prediction CSV path for an experiment."""
    return Path("outputs/evaluation") / experiment_name / "test" / "predictions.csv"


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    """Load prediction rows from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def parse_patch_coordinates(patch_id: str) -> tuple[int, int]:
    """Parse x and y coordinates from a patch_id string."""
    match = COORDINATE_PATTERN.search(str(patch_id))
    if not match:
        raise ValueError(f"Could not parse x/y coordinates from patch_id: {patch_id}")
    return int(match.group("x")), int(match.group("y"))


def to_float(value: Any) -> float | None:
    """Convert CSV value to float, returning None for missing/non-finite values."""
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


def available_pair_counts(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Count available district and pair_id combinations."""
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        district = row.get("district", "")
        pair_id = row.get("pair_id", "")
        counts[(district, pair_id)] = counts.get((district, pair_id), 0) + 1
    pair_rows = [
        {"district": district, "pair_id": pair_id, "patch_count": count}
        for (district, pair_id), count in counts.items()
    ]
    return sorted(pair_rows, key=lambda item: (-int(item["patch_count"]), item["district"], item["pair_id"]))


def save_available_pairs(rows: list[dict[str, Any]], path: Path) -> None:
    """Save available pair counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["district", "pair_id", "patch_count"])
        writer.writeheader()
        writer.writerows(rows)


def print_available_pairs(rows: list[dict[str, Any]], top_pairs: int) -> None:
    """Print available pair counts to the console."""
    limit = max(0, int(top_pairs))
    print(f"Available district + pair_id combinations, top {limit}:")
    print("district,patch_count,pair_id")
    for row in rows[:limit]:
        print(f"{row['district']},{row['patch_count']},{row['pair_id']}")


def filter_rows(rows: list[dict[str, str]], district: str, pair_id: str) -> list[dict[str, str]]:
    """Filter prediction rows to one district and pair_id."""
    return [
        row
        for row in rows
        if row.get("district") == district and row.get("pair_id") == pair_id
    ]


def add_coordinates(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Return rows with parsed x/y coordinates."""
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        x_coord, y_coord = parse_patch_coordinates(row.get("patch_id", ""))
        parsed_row: dict[str, Any] = dict(row)
        parsed_row["x"] = x_coord
        parsed_row["y"] = y_coord
        parsed_rows.append(parsed_row)
    return parsed_rows


def value_column_exists(rows: list[dict[str, Any]], column: str) -> bool:
    """Return whether at least one row has a usable numeric value for a column."""
    return any(to_float(row.get(column)) is not None for row in rows)


def build_grid(rows: list[dict[str, Any]], value_column: str) -> tuple[list[int], list[int], list[list[float]]]:
    """Build a y-by-x grid of values from parsed patch rows."""
    x_values = sorted({int(row["x"]) for row in rows})
    y_values = sorted({int(row["y"]) for row in rows})
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    grid = [[math.nan for _ in x_values] for _ in y_values]

    for row in rows:
        value = to_float(row.get(value_column))
        if value is None:
            continue
        grid[y_index[int(row["y"])]][x_index[int(row["x"])]] = value
    return x_values, y_values, grid


def coordinate_edges(values: list[int]) -> list[int]:
    """Create patch-cell edges from top-left patch coordinates."""
    if not values:
        return []
    return values + [values[-1] + PATCH_SIZE]


def plot_grid_map(
    rows: list[dict[str, Any]],
    value_column: str,
    output_path: Path,
    title: str,
    colorbar_label: str,
    cmap: str,
) -> None:
    """Save a high-resolution patch-grid map."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to create PNG maps. Install it in the active "
            "environment with: python -m pip install matplotlib"
        ) from exc

    x_values, y_values, grid = build_grid(rows, value_column)
    x_edges = coordinate_edges(x_values)
    y_edges = coordinate_edges(y_values)

    fig_width = max(8.0, min(16.0, len(x_values) * 0.35))
    fig_height = max(6.0, min(14.0, len(y_values) * 0.35))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    mesh = ax.pcolormesh(x_edges, y_edges, grid, shading="flat", cmap=cmap)
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    fig.colorbar(mesh, ax=ax, label=colorbar_label)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def safe_filename_part(value: str) -> str:
    """Make a safe filename component while preserving readable pair IDs."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def save_filtered_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Save filtered predictions with parsed coordinates."""
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


def mean(values: list[float]) -> float | None:
    """Return mean value or None."""
    if not values:
        return None
    return sum(values) / len(values)


def format_optional(value: float | None) -> str:
    """Format optional float for Markdown."""
    return "not available" if value is None else f"{value:.6f}"


def write_report(
    path: Path,
    experiment_name: str,
    district: str,
    pair_id: str,
    rows: list[dict[str, Any]],
    figure_paths: list[Path],
) -> None:
    """Write a Markdown report for one district pair map."""
    pred_values = [value for value in (to_float(row.get("y_pred_change_ratio")) for row in rows) if value is not None]
    true_values = [value for value in (to_float(row.get("y_true_change_ratio")) for row in rows) if value is not None]
    error_values = [value for value in (to_float(row.get("abs_error")) for row in rows) if value is not None]

    lines = [
        "# District Prediction Patch-Grid Report",
        "",
        f"- Experiment name: `{experiment_name}`",
        f"- District: `{district}`",
        f"- Pair ID: `{pair_id}`",
        f"- Number of patches: {len(rows)}",
        f"- Mean predicted change_ratio: {format_optional(mean(pred_values))}",
        f"- Max predicted change_ratio: {format_optional(max(pred_values) if pred_values else None)}",
        f"- Mean true change_ratio: {format_optional(mean(true_values))}",
        f"- Mean absolute error: {format_optional(mean(error_values))}",
        "",
        "## Generated Figures",
        "",
    ]
    for figure_path in figure_paths:
        lines.append(f"- `{figure_path}`")
    lines.extend(
        [
            "",
            "This visualization is a patch-grid map reconstructed from existing 128x128 patch predictions. It is not direct full-image model inference and does not require raw raster files.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def print_missing_selection_message(experiment_name: str, prediction_csv: Path) -> None:
    """Print guidance when district/pair_id are missing."""
    print("No district and pair_id were provided, so no map was created.")
    print("Run this first to see available combinations:")
    print(
        "  python -u scripts/28_plot_district_prediction_map.py "
        f"--experiment_name {experiment_name} --prediction_csv \"{prediction_csv}\" --list_pairs"
    )
    print("Then run again with --district and --pair_id.")


def main() -> None:
    """Create list-pair outputs or one selected district prediction map."""
    args = parse_args()
    experiment_name = args.experiment_name
    prediction_csv = Path(args.prediction_csv) if args.prediction_csv else default_prediction_csv(experiment_name)
    output_dir = Path(args.output_dir)
    rows = read_prediction_rows(prediction_csv)

    if args.list_pairs:
        pair_rows = available_pair_counts(rows)
        available_pairs_path = output_dir / f"available_pairs_{experiment_name}.csv"
        save_available_pairs(pair_rows, available_pairs_path)
        print_available_pairs(pair_rows, args.top_pairs)
        print(f"Saved available pairs CSV: {available_pairs_path}")
        return

    if not args.district or not args.pair_id:
        print_missing_selection_message(experiment_name, prediction_csv)
        return

    filtered_rows = add_coordinates(filter_rows(rows, args.district, args.pair_id))
    if not filtered_rows:
        print(f"No rows found for district={args.district!r}, pair_id={args.pair_id!r}.")
        print("Run with --list_pairs to inspect available combinations.")
        return

    prefix = (
        f"{safe_filename_part(experiment_name)}_"
        f"{safe_filename_part(args.district)}_"
        f"{safe_filename_part(args.pair_id)}"
    )
    filtered_csv_path = output_dir / f"{prefix}_patch_predictions.csv"
    report_path = output_dir / f"{prefix}_district_prediction_report.md"
    figure_paths: list[Path] = []

    save_filtered_csv(filtered_rows, filtered_csv_path)

    title_base = (
        f"{experiment_name} | {args.district} | {args.pair_id} | "
        f"patches={len(filtered_rows)}"
    )
    pred_path = output_dir / f"{prefix}_predicted_change_ratio_map.png"
    plot_grid_map(
        filtered_rows,
        "y_pred_change_ratio",
        pred_path,
        f"Predicted change ratio\n{title_base}",
        "predicted change ratio",
        "viridis",
    )
    figure_paths.append(pred_path)

    if value_column_exists(filtered_rows, "y_true_change_ratio"):
        true_path = output_dir / f"{prefix}_true_change_ratio_map.png"
        plot_grid_map(
            filtered_rows,
            "y_true_change_ratio",
            true_path,
            f"True change ratio\n{title_base}",
            "true change ratio",
            "viridis",
        )
        figure_paths.append(true_path)

    if value_column_exists(filtered_rows, "abs_error"):
        error_path = output_dir / f"{prefix}_absolute_error_map.png"
        plot_grid_map(
            filtered_rows,
            "abs_error",
            error_path,
            f"Absolute error\n{title_base}",
            "absolute error",
            "magma",
        )
        figure_paths.append(error_path)

    write_report(
        report_path,
        experiment_name,
        args.district,
        args.pair_id,
        filtered_rows,
        figure_paths,
    )

    print("District prediction map created.")
    print(f"  filtered CSV: {filtered_csv_path}")
    for figure_path in figure_paths:
        print(f"  figure: {figure_path}")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
