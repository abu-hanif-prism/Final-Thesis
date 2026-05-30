"""Markdown report helpers for XAI explanation steps."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_modality_temporal_report(
    model_name: str,
    experiment_name: str,
    checkpoint_path: str | Path,
    num_samples: int,
    summary_rows: list[dict[str, Any]],
    output_csv_path: str | Path,
) -> str:
    """Build a Markdown report for modality and temporal ablation."""
    summary = {(row["metric"], row["group"]): row["value"] for row in summary_rows}
    avg_image = summary.get(("average_image_contribution", "all"), "")
    avg_tabular = summary.get(("average_tabular_contribution", "all"), "")
    avg_t1 = summary.get(("average_t1_contribution", "all"), "")
    avg_t2 = summary.get(("average_t2_contribution", "all"), "")
    lines = [
        f"# Modality and Temporal Ablation: {experiment_name}",
        "",
        f"- model_name: {model_name}",
        f"- experiment_name: {experiment_name}",
        f"- checkpoint used: {checkpoint_path}",
        f"- number of samples: {num_samples}",
        f"- output_csv: {output_csv_path}",
        "",
        "## Average Contributions",
        "",
        f"- average image contribution: {avg_image}",
        f"- average tabular contribution: {avg_tabular}",
        f"- average t1 contribution: {avg_t1}",
        f"- average t2 contribution: {avg_t2}",
        "",
        "## Interpretation",
        "",
        _interpret(avg_image, avg_tabular, avg_t1, avg_t2),
        "",
        "## Summary Table",
        "",
    ]
    for row in summary_rows:
        lines.append(f"- {row['metric']} / {row['group']}: {row['value']}")
    return "\n".join(lines) + "\n"


def save_markdown_report(text: str, path: str | Path) -> Path:
    """Save Markdown text to disk."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    return report_path


def _interpret(avg_image: Any, avg_tabular: Any, avg_t1: Any, avg_t2: Any) -> str:
    """Create a short human-readable interpretation."""
    image = _to_float(avg_image)
    tabular = _to_float(avg_tabular)
    t1 = _to_float(avg_t1)
    t2 = _to_float(avg_t2)
    parts: list[str] = []
    if image is not None and tabular is not None:
        if image >= tabular:
            parts.append("On average, image evidence changes the prediction more than tabular evidence.")
        else:
            parts.append("On average, tabular evidence changes the prediction more than image evidence.")
    if t1 is not None and t2 is not None:
        if t1 >= t2:
            parts.append("The model is more sensitive to removing the first timestamp than the second timestamp.")
        else:
            parts.append("The model is more sensitive to removing the second timestamp than the first timestamp.")
    if not parts:
        return "Contribution statistics were unavailable for interpretation."
    return " ".join(parts)


def _to_float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
