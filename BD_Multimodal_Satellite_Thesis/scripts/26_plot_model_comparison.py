"""Create thesis-ready model comparison bar charts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


METRICS_PATH = Path("outputs/reports/model_comparison/model_comparison_metrics.csv")
OUTPUT_DIR = Path("outputs/reports/model_comparison/figures")
REPORT_PATH = Path("outputs/reports/model_comparison/model_comparison_figure_report.md")

LABELS = {
    "cnn_regression": "CNN",
    "swin_regression": "Swin",
    "convnext_regression": "ConvNeXt",
    "maxvit_regression_stable": "MaxViT Stable",
    "maxvit_regression": "MaxViT",
}

COLORS = {
    "CNN": "#4C78A8",
    "Swin": "#F58518",
    "ConvNeXt": "#54A24B",
    "MaxViT Stable": "#B279A2",
    "MaxViT": "#B279A2",
}


def main() -> None:
    """Read model comparison metrics and create PNG figures plus report."""
    rows = load_metrics(METRICS_PATH)
    if not rows:
        raise ValueError(f"No rows found in {METRICS_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rmse_order = sorted(rows, key=lambda row: row["rmse"])
    mae_order = sorted(rows, key=lambda row: row["rmse"])
    r2_order = sorted(rows, key=lambda row: row["r2"], reverse=True)
    pearson_order = sorted(rows, key=lambda row: row["pearson_corr"], reverse=True)

    figure_paths = {
        "RMSE": OUTPUT_DIR / "model_comparison_rmse.png",
        "MAE": OUTPUT_DIR / "model_comparison_mae.png",
        "R2": OUTPUT_DIR / "model_comparison_r2.png",
        "Pearson": OUTPUT_DIR / "model_comparison_pearson.png",
        "All Metrics": OUTPUT_DIR / "model_comparison_all_metrics.png",
    }

    plot_metric(rmse_order, "rmse", "RMSE", "Lower is better", figure_paths["RMSE"])
    plot_metric(mae_order, "mae", "MAE", "Lower is better", figure_paths["MAE"])
    plot_metric(r2_order, "r2", "R²", "Higher is better", figure_paths["R2"])
    plot_metric(pearson_order, "pearson_corr", "Pearson Correlation", "Higher is better", figure_paths["Pearson"])
    plot_combined(rows, figure_paths["All Metrics"])

    REPORT_PATH.write_text(build_report(rows, figure_paths), encoding="utf-8")

    print("Model comparison figures created.", flush=True)
    for name, path in figure_paths.items():
        print(f"  {name}: {path}", flush=True)
    print(f"  report: {REPORT_PATH}", flush=True)


def load_metrics(path: Path) -> list[dict[str, Any]]:
    """Load metric rows from CSV using the standard library."""
    if not path.exists():
        raise FileNotFoundError(f"Model comparison metrics file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("status") and row.get("status") != "ok":
                continue
            parsed = dict(row)
            parsed["label"] = LABELS.get(row.get("experiment_name", ""), row.get("experiment_name", "unknown"))
            for metric in ["rmse", "mae", "r2", "pearson_corr"]:
                parsed[metric] = parse_float(row.get(metric))
            if all(parsed[metric] is not None for metric in ["rmse", "mae", "r2", "pearson_corr"]):
                rows.append(parsed)
    return rows


def plot_metric(rows: list[dict[str, Any]], metric: str, title: str, subtitle: str, output_path: Path) -> None:
    """Save one bar chart for a single metric."""
    labels = [row["label"] for row in rows]
    values = [float(row[metric]) for row in rows]
    colors = [COLORS.get(label, "#777777") for label in labels]

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    bars = ax.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.7)
    ax.set_title(f"{title} by Model", fontsize=15, weight="bold", pad=14)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(title)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_value_labels(ax, bars, values)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_combined(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Save a 2x2 combined figure for all four metrics."""
    specs = [
        ("rmse", "RMSE", "Lower is better", sorted(rows, key=lambda row: row["rmse"])),
        ("mae", "MAE", "Lower is better", sorted(rows, key=lambda row: row["rmse"])),
        ("r2", "R²", "Higher is better", sorted(rows, key=lambda row: row["r2"], reverse=True)),
        (
            "pearson_corr",
            "Pearson",
            "Higher is better",
            sorted(rows, key=lambda row: row["pearson_corr"], reverse=True),
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=160)
    for ax, (metric, title, subtitle, metric_rows) in zip(axes.flatten(), specs):
        labels = [row["label"] for row in metric_rows]
        values = [float(row[metric]) for row in metric_rows]
        colors = [COLORS.get(label, "#777777") for label in labels]
        bars = ax.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.6)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        add_value_labels(ax, bars, values, fontsize=8)
        ax.tick_params(axis="x", labelrotation=12)
    fig.suptitle("Model Comparison Metrics", fontsize=17, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def add_value_labels(ax: Any, bars: Any, values: list[float], fontsize: int = 9) -> None:
    """Add numeric labels above bars."""
    max_value = max(values) if values else 1.0
    offset = max_value * 0.015 if max_value else 0.005
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def build_report(rows: list[dict[str, Any]], figure_paths: dict[str, Path]) -> str:
    """Build Markdown report for the generated comparison figures."""
    best_rmse = min(rows, key=lambda row: row["rmse"])
    best_mae = min(rows, key=lambda row: row["mae"])
    best_r2 = max(rows, key=lambda row: row["r2"])
    best_pearson = max(rows, key=lambda row: row["pearson_corr"])
    lines = [
        "# Model Comparison Figure Report",
        "",
        f"- best model by RMSE: {best_rmse['label']} ({best_rmse['rmse']:.6f})",
        f"- best model by MAE: {best_mae['label']} ({best_mae['mae']:.6f})",
        f"- best model by R²: {best_r2['label']} ({best_r2['r2']:.6f})",
        f"- best model by Pearson: {best_pearson['label']} ({best_pearson['pearson_corr']:.6f})",
        "",
        "## Figures",
        "",
    ]
    for name, path in figure_paths.items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def parse_float(value: Any) -> float | None:
    """Parse a float from CSV text."""
    if value is None or str(value).strip() == "":
        return None
    return float(value)


if __name__ == "__main__":
    main()
