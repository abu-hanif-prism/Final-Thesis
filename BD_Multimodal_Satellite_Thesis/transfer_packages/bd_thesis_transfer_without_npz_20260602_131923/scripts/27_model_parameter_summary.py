"""Summarize layer-wise parameter counts for final thesis models."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_factory import create_model_from_config, load_model_config


MODELS = [
    {
        "experiment_name": "cnn_regression",
        "label": "CNN regression",
        "config_path": Path("checkpoints/cnn_regression_model_config.json"),
        "summary_path": Path("outputs/reports/model_architecture/cnn_parameter_summary.csv"),
    },
    {
        "experiment_name": "swin_regression",
        "label": "Swin regression",
        "config_path": Path("checkpoints/swin_regression_model_config.json"),
        "summary_path": Path("outputs/reports/model_architecture/swin_parameter_summary.csv"),
    },
    {
        "experiment_name": "convnext_regression",
        "label": "ConvNeXt regression",
        "config_path": Path("checkpoints/convnext_regression_model_config.json"),
        "summary_path": Path("outputs/reports/model_architecture/convnext_parameter_summary.csv"),
    },
    {
        "experiment_name": "maxvit_regression_stable",
        "label": "MaxViT regression stable",
        "config_path": Path("checkpoints/maxvit_regression_stable_model_config.json"),
        "summary_path": Path(
            "outputs/reports/model_architecture/maxvit_regression_stable_parameter_summary.csv"
        ),
    },
]

OUTPUT_DIR = Path("outputs/reports/model_architecture")
TOTALS_PATH = OUTPUT_DIR / "model_parameter_totals.csv"
REPORT_PATH = OUTPUT_DIR / "model_architecture_summary.md"


def direct_parameter_count(module) -> tuple[int, int]:
    """Return total and trainable parameters owned directly by one module."""
    parameters = list(module.parameters(recurse=False))
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return int(total), int(trainable)


def summarize_model_modules(model) -> tuple[list[dict[str, Any]], int, int]:
    """Build a non-duplicated module-level parameter summary."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    rows: list[dict[str, Any]] = []

    for module_name, module in model.named_modules():
        module_parameters, trainable_module_parameters = direct_parameter_count(module)
        if module_parameters == 0:
            continue
        parameter_percentage = (
            (module_parameters / total_parameters) * 100.0 if total_parameters else 0.0
        )
        rows.append(
            {
                "module_name": module_name or "<root>",
                "module_type": module.__class__.__name__,
                "parameters": module_parameters,
                "trainable_parameters": trainable_module_parameters,
                "parameter_percentage": round(parameter_percentage, 6),
            }
        )

    return rows, int(total_parameters), int(trainable_parameters)


def save_module_summary(rows: list[dict[str, Any]], path: Path) -> None:
    """Save one model's layer-wise parameter summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "module_name",
        "module_type",
        "parameters",
        "trainable_parameters",
        "parameter_percentage",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_totals(totals: list[dict[str, Any]], path: Path) -> None:
    """Save model-level parameter totals."""
    fieldnames = [
        "experiment_name",
        "model_name",
        "output_mode",
        "tabular_dim",
        "total_parameters",
        "trainable_parameters",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(totals)


def build_markdown_report(totals: list[dict[str, Any]]) -> str:
    """Create a thesis-friendly markdown summary."""
    highest = max(totals, key=lambda row: int(row["total_parameters"]))
    lines = [
        "# Model Architecture Parameter Summary",
        "",
        "This report summarizes the layer-wise and total parameter counts for the final regression models.",
        "",
        "## Architectural Notes",
        "",
        "- The models use a shared Siamese image encoder: the same image encoder processes t1 and t2 images, so t1/t2 branches do not double the image-encoder parameters.",
        "- The tabular encoder processes 146 pair-level tabular features.",
        "- The final prediction head outputs one regression value for the predicted change ratio.",
        f"- MaxViT has the highest parameter count among the final models: {highest['experiment_name']} with {highest['total_parameters']:,} parameters.",
        "",
        "## Total Parameters",
        "",
        "| Model | Architecture | Output | Tabular features | Total parameters | Trainable parameters |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in totals:
        lines.append(
            "| {experiment_name} | {model_name} | {output_mode} | {tabular_dim} | "
            "{total_parameters:,} | {trainable_parameters:,} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Layer-wise CSV Files",
            "",
            "- `cnn_parameter_summary.csv`",
            "- `swin_parameter_summary.csv`",
            "- `convnext_parameter_summary.csv`",
            "- `maxvit_regression_stable_parameter_summary.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def print_model_summary(experiment_name: str, rows: list[dict[str, Any]], total: int, trainable: int) -> None:
    """Print a concise layer-wise summary to the console."""
    print(f"\n{experiment_name}")
    print(f"total_parameters: {total:,}")
    print(f"trainable_parameters: {trainable:,}")
    print("module_name,module_type,parameters,trainable_parameters,parameter_percentage")
    for row in rows:
        print(
            "{module_name},{module_type},{parameters},{trainable_parameters},{parameter_percentage}".format(
                **row
            )
        )


def main() -> None:
    """Load final model configs, summarize parameters, and write reports."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    totals: list[dict[str, Any]] = []

    for model_info in MODELS:
        config_path = model_info["config_path"]
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")

        config = load_model_config(config_path)
        model = create_model_from_config(config)
        rows, total_parameters, trainable_parameters = summarize_model_modules(model)
        save_module_summary(rows, model_info["summary_path"])

        total_row = {
            "experiment_name": model_info["experiment_name"],
            "model_name": config.get("model_name", ""),
            "output_mode": config.get("output_mode", ""),
            "tabular_dim": config.get("tabular_dim", 146),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        }
        totals.append(total_row)
        print_model_summary(model_info["experiment_name"], rows, total_parameters, trainable_parameters)

    save_totals(totals, TOTALS_PATH)
    REPORT_PATH.write_text(build_markdown_report(totals), encoding="utf-8")

    print("\nSaved model architecture reports:")
    for model_info in MODELS:
        print(f"  {model_info['summary_path']}")
    print(f"  {TOTALS_PATH}")
    print(f"  {REPORT_PATH}")


if __name__ == "__main__":
    main()
