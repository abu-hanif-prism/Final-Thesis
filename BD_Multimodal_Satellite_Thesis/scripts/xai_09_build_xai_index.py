"""Build a CSV and Markdown index of XAI output artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


INDEX_COLUMNS = ["experiment_name", "artifact_type", "path", "parent", "size_bytes"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build final XAI output index.")
    parser.add_argument("--xai_root", default="outputs/xai")
    return parser.parse_args()


def main() -> None:
    """Scan outputs/xai and save CSV plus Markdown index."""
    args = parse_args()
    root = Path(args.xai_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = build_index_rows(root)
    csv_path = root / "xai_output_index.csv"
    md_path = root / "xai_output_index.md"
    save_csv(rows, csv_path)
    md_path.write_text(build_markdown(rows, root), encoding="utf-8")
    print(f"indexed artifacts: {len(rows)}", flush=True)
    print(f"CSV index: {csv_path}", flush=True)
    print(f"Markdown index: {md_path}", flush=True)


def build_index_rows(root: Path) -> list[dict[str, object]]:
    """Return one row per CSV, JSON, Markdown, PNG, and PNG folder."""
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".png"}:
            rows.append(
                {
                    "experiment_name": infer_experiment_name(path),
                    "artifact_type": path.suffix.lower().lstrip("."),
                    "path": str(path),
                    "parent": str(path.parent),
                    "size_bytes": path.stat().st_size,
                }
            )
    png_folders = sorted({path.parent for path in root.rglob("*.png")})
    for folder in png_folders:
        rows.append(
            {
                "experiment_name": infer_experiment_name(folder),
                "artifact_type": "png_folder",
                "path": str(folder),
                "parent": str(folder.parent),
                "size_bytes": "",
            }
        )
    return rows


def infer_experiment_name(path: Path) -> str:
    """Infer experiment name from file or folder path."""
    text = str(path).replace("\\", "/")
    known = ["cnn_regression", "convnext_regression", "swin_regression", "maxvit_regression"]
    for name in known:
        if name in text:
            return name
    parts = path.stem.split("_")
    for index in range(len(parts) - 1):
        candidate = "_".join(parts[index : index + 2])
        if candidate.endswith("_regression"):
            return candidate
    return "unknown"


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Save index rows as CSV."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in INDEX_COLUMNS})


def build_markdown(rows: list[dict[str, object]], root: Path) -> str:
    """Build Markdown index."""
    by_experiment: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_experiment.setdefault(str(row.get("experiment_name", "unknown")), []).append(row)
    lines = ["# XAI Output Index", "", f"Root: `{root}`", ""]
    for experiment, experiment_rows in sorted(by_experiment.items()):
        lines.extend([f"## {experiment}", ""])
        for row in experiment_rows:
            lines.append(f"- {row.get('artifact_type')}: `{row.get('path')}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    main()
