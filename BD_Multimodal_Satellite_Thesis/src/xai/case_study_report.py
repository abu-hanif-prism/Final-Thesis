"""Combine XAI outputs into compact case-study summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.xai.sample_selector import load_csv_rows, safe_float, save_csv_rows


CASE_COLUMNS = [
    "model_name",
    "experiment_name",
    "checkpoint",
    "patch_id",
    "district",
    "pair_id",
    "true_change_ratio",
    "predicted_change_ratio",
    "absolute_error",
    "dominant_modality",
    "dominant_time_step",
    "top_bands",
    "top_tabular_features",
    "occlusion_paths",
    "gradcam_paths",
    "plain_english_explanation",
]


def build_case_studies(
    model_name: str,
    experiment_name: str,
    output_root: str | Path = "outputs/xai",
    max_cases: int = 10,
    checkpoint_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build case-study rows by joining existing XAI CSV outputs."""
    root = Path(output_root)
    selected = _load_optional(root / "selected_samples" / f"xai_selected_samples_{experiment_name}.csv")
    modality = _by_patch(_load_optional(root / "modality_ablation" / f"modality_temporal_ablation_{experiment_name}.csv"))
    band_rows = _load_optional(root / "band_importance" / f"band_importance_per_sample_{experiment_name}.csv")
    tabular_rows = _load_optional(root / "tabular_importance" / f"tabular_importance_per_sample_{experiment_name}.csv")
    occlusion_rows = _load_optional(root / "occlusion" / f"occlusion_scores_{experiment_name}.csv")
    gradcam_rows = _load_optional(root / "gradcam" / f"gradcam_summary_{experiment_name}.csv")
    checkpoint = Path(checkpoint_path) if checkpoint_path else Path("checkpoints") / f"{experiment_name}_best.pt"

    output: list[dict[str, Any]] = []
    for row in selected[: max(0, int(max_cases))]:
        patch_id = str(row.get("patch_id", ""))
        modality_row = modality.get(patch_id, {})
        top_bands = _top_items(band_rows, patch_id, "band_name", "importance", top_k=3)
        top_features = _top_items(tabular_rows, patch_id, "feature_name", "importance", top_k=5)
        occlusion_paths = _paths_for_patch(occlusion_rows, patch_id, ["overlay_path", "side_by_side_path"])
        gradcam_paths = _paths_for_patch(gradcam_rows, patch_id, ["overlay_path"])
        true_ratio = row.get("y_true_change_ratio") or modality_row.get("true_change_ratio")
        pred_ratio = row.get("y_pred_change_ratio") or modality_row.get("full_prediction")
        error = row.get("abs_error") or modality_row.get("absolute_error")
        dominant_modality = modality_row.get("dominant_modality", "")
        dominant_time_step = modality_row.get("dominant_time_step", "")
        output.append(
            {
                "model_name": model_name,
                "experiment_name": experiment_name,
                "checkpoint": str(checkpoint),
                "patch_id": patch_id,
                "district": row.get("district", ""),
                "pair_id": row.get("pair_id", ""),
                "true_change_ratio": true_ratio,
                "predicted_change_ratio": pred_ratio,
                "absolute_error": error,
                "dominant_modality": dominant_modality,
                "dominant_time_step": dominant_time_step,
                "top_bands": "; ".join(top_bands),
                "top_tabular_features": "; ".join(top_features),
                "occlusion_paths": "; ".join(occlusion_paths),
                "gradcam_paths": "; ".join(gradcam_paths),
                "plain_english_explanation": _plain_english(
                    row,
                    true_ratio,
                    pred_ratio,
                    error,
                    dominant_modality,
                    dominant_time_step,
                    top_bands,
                    top_features,
                ),
            }
        )
    return output


def save_case_studies(
    rows: list[dict[str, Any]],
    model_name: str,
    experiment_name: str,
    output_root: str | Path = "outputs/xai",
) -> tuple[Path, Path]:
    """Save case-study CSV and Markdown report."""
    root = Path(output_root)
    case_dir = root / "case_studies"
    csv_path = case_dir / f"case_study_summary_{experiment_name}.csv"
    md_path = case_dir / f"case_study_report_{experiment_name}.md"
    save_csv_rows(rows, csv_path, fieldnames=CASE_COLUMNS)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(build_case_study_markdown(rows, model_name, experiment_name), encoding="utf-8")
    return csv_path, md_path


def build_case_study_markdown(rows: list[dict[str, Any]], model_name: str, experiment_name: str) -> str:
    """Create Markdown case-study report text."""
    lines = [
        f"# XAI Case Study Report: {experiment_name}",
        "",
        f"- model_name: {model_name}",
        f"- experiment_name: {experiment_name}",
        f"- cases included: {len(rows)}",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## Case {index}: {row.get('patch_id')}",
                "",
                f"- district: {row.get('district')}",
                f"- pair_id: {row.get('pair_id')}",
                f"- true change_ratio: {row.get('true_change_ratio')}",
                f"- predicted change_ratio: {row.get('predicted_change_ratio')}",
                f"- absolute error: {row.get('absolute_error')}",
                f"- dominant modality: {row.get('dominant_modality')}",
                f"- dominant time step: {row.get('dominant_time_step')}",
                f"- top bands: {row.get('top_bands')}",
                f"- top tabular features: {row.get('top_tabular_features')}",
                f"- occlusion paths: {row.get('occlusion_paths')}",
                f"- Grad-CAM paths: {row.get('gradcam_paths')}",
                "",
                row.get("plain_english_explanation", ""),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _load_optional(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"Warning: optional XAI file missing: {path}", flush=True)
        return []
    return load_csv_rows(path)


def _by_patch(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("patch_id")): row for row in rows if row.get("patch_id")}


def _top_items(
    rows: list[dict[str, Any]],
    patch_id: str,
    name_column: str,
    score_column: str,
    top_k: int,
) -> list[str]:
    matches = [row for row in rows if str(row.get("patch_id", "")) == patch_id]
    matches = sorted(matches, key=lambda row: safe_float(row.get(score_column), 0.0) or 0.0, reverse=True)
    output = []
    for row in matches[:top_k]:
        value = safe_float(row.get(score_column), 0.0) or 0.0
        output.append(f"{row.get(name_column, '')} ({value:.6f})")
    return output


def _paths_for_patch(rows: list[dict[str, Any]], patch_id: str, columns: list[str]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if str(row.get("patch_id", "")) != patch_id:
            continue
        for column in columns:
            value = str(row.get(column, "")).strip()
            if value and value not in seen:
                paths.append(value)
                seen.add(value)
    return paths


def _plain_english(
    row: dict[str, Any],
    true_ratio: Any,
    pred_ratio: Any,
    error: Any,
    dominant_modality: Any,
    dominant_time_step: Any,
    top_bands: list[str],
    top_features: list[str],
) -> str:
    true_value = safe_float(true_ratio)
    pred_value = safe_float(pred_ratio)
    error_value = safe_float(error)
    change_class = row.get("change_class", "unknown")
    sentence = (
        f"For this {change_class} change sample in {row.get('district', 'unknown district')}, "
        f"the model predicted {pred_value:.3f} against a true value of {true_value:.3f} "
        if true_value is not None and pred_value is not None
        else f"For this {change_class} change sample, the model prediction metadata is incomplete "
    )
    if error_value is not None:
        sentence += f"with absolute error {error_value:.3f}. "
    else:
        sentence += ". "
    if dominant_modality:
        sentence += f"The ablation results suggest the {dominant_modality} modality was more influential. "
    if dominant_time_step:
        sentence += f"The {dominant_time_step} image contributed more strongly in the temporal comparison. "
    if top_bands:
        sentence += f"Top image bands were {', '.join(top_bands[:3])}. "
    if top_features:
        sentence += f"Top tabular drivers were {', '.join(top_features[:3])}."
    return sentence
