"""Export an XAI handoff package for trained multimodal Siamese models."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SENTINEL_BAND_NAMES = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
    "NDVI",
    "NDWI",
    "MNDWI",
    "NDBI",
    "NDMI",
    "BSI",
    "EVI",
]


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if missing and return it as Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_predictions(prediction_path: str | Path) -> pd.DataFrame:
    """Load predictions from parquet or CSV, preferring parquet when available."""
    path = Path(prediction_path)
    candidates = []
    if path.suffix.lower() in {".parquet", ".csv"}:
        candidates.append(path)
        alternate_suffix = ".csv" if path.suffix.lower() == ".parquet" else ".parquet"
        candidates.append(path.with_suffix(alternate_suffix))
    else:
        candidates.extend([path.with_suffix(".parquet"), path.with_suffix(".csv")])

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix.lower() == ".parquet":
            return pd.read_parquet(candidate)
        if candidate.suffix.lower() == ".csv":
            return pd.read_csv(candidate)
    raise FileNotFoundError(f"Prediction file not found. Tried: {[str(c) for c in candidates]}")


def load_npz_index(index_path: str | Path) -> pd.DataFrame:
    """Load final NPZ index parquet."""
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ index not found: {path}")
    return pd.read_parquet(path)


def select_xai_handoff_samples(
    predictions_df: pd.DataFrame,
    npz_index_df: pd.DataFrame,
    num_samples: int = 200,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Select representative samples for XAI handoff without duplicate patch IDs."""
    if "patch_id" not in predictions_df.columns or "patch_id" not in npz_index_df.columns:
        raise KeyError("Both predictions_df and npz_index_df must contain patch_id")

    merged = predictions_df.merge(
        npz_index_df,
        on="patch_id",
        how="left",
        suffixes=("", "_index"),
    )
    merged = _coalesce_duplicate_columns(merged)
    if "abs_error" not in merged.columns and {"y_true_change_ratio", "y_pred_change_ratio"}.issubset(merged.columns):
        merged["abs_error"] = (merged["y_pred_change_ratio"] - merged["y_true_change_ratio"]).abs()
    if "change_ratio" not in merged.columns and "y_true_change_ratio" in merged.columns:
        merged["change_ratio"] = merged["y_true_change_ratio"]

    selected_parts: list[pd.DataFrame] = []
    per_bucket = max(1, int(num_samples) // 12)
    for change_class in ["low", "medium", "high"]:
        selected_parts.append(
            _sample_group(
                merged[merged.get("change_class", "") == change_class],
                min(per_bucket, num_samples),
                random_seed,
                f"class_{change_class}",
            )
        )

    if "abs_error" in merged.columns:
        error_sorted = merged.sort_values("abs_error", ascending=True)
        selected_parts.append(_sample_group(error_sorted.head(max(per_bucket * 2, 1)), per_bucket, random_seed, "low_prediction_error"))
        selected_parts.append(_sample_group(error_sorted.tail(max(per_bucket * 2, 1)), per_bucket, random_seed, "high_prediction_error"))

    for column in ["district", "pair_type", "time_gap_group"]:
        if column not in merged.columns:
            continue
        selected_parts.append(_sample_across_values(merged, column, per_bucket, random_seed))

    selected = _combine_selected(selected_parts)
    if len(selected) < min(num_samples, len(merged)):
        remaining = merged[~merged["patch_id"].isin(set(selected["patch_id"]))]
        fill_count = min(num_samples - len(selected), len(remaining))
        if fill_count > 0:
            filler = remaining.sample(n=fill_count, random_state=random_seed).copy()
            filler["selection_reason"] = "random_fill"
            selected = pd.concat([selected, filler], ignore_index=True)

    selected = selected.drop_duplicates("patch_id").head(num_samples).reset_index(drop=True)
    output_columns = [
        "patch_id",
        "pair_id",
        "district",
        "split",
        "change_class",
        "change_ratio",
        "y_true_change_ratio",
        "y_pred_change_ratio",
        "abs_error",
        "pair_type",
        "time_gap_group",
        "npz_path",
        "selection_reason",
    ]
    available_columns = [column for column in output_columns if column in selected.columns]
    return selected[available_columns].copy()


def create_feature_info(
    feature_columns_path: str | Path = "data/tabular/processed/pair_tabular_feature_columns.json",
) -> dict[str, Any]:
    """Create feature and input metadata for XAI handoff."""
    path = Path(feature_columns_path)
    feature_columns: list[str] = []
    raw_payload: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            raw_payload = json.load(file)
        feature_columns = _extract_feature_columns(raw_payload)
    return {
        "sentinel_band_names": SENTINEL_BAND_NAMES,
        "image_shape": [13, 128, 128],
        "tabular_feature_count": len(feature_columns),
        "tabular_feature_columns": feature_columns,
        "class_mapping": {"low": 0, "medium": 1, "high": 2},
        "target_name": "change_ratio",
        "model_inputs": ["image_t1", "image_t2", "tabular"],
        "feature_columns_source": str(path),
        "feature_columns_payload_keys": sorted(raw_payload.keys()),
    }


def copy_checkpoint(source_checkpoint_path: str | Path, output_dir: str | Path) -> Path:
    """Copy checkpoint to best.pt inside output_dir."""
    source = Path(source_checkpoint_path)
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    output_path = ensure_dir(output_dir) / "best.pt"
    shutil.copy2(source, output_path)
    return output_path


def copy_model_config(
    source_config_path: str | Path,
    output_dir: str | Path,
    minimal_config: dict[str, Any] | None = None,
) -> Path:
    """Copy model config or create a minimal fallback config."""
    source = Path(source_config_path)
    output_path = ensure_dir(output_dir) / "model_config.json"
    if source.exists():
        shutil.copy2(source, output_path)
    else:
        config = minimal_config or {}
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)
            file.write("\n")
    return output_path


def save_handoff_package(
    experiment_name: str,
    model_name: str,
    checkpoint_path: str | Path,
    model_config_path: str | Path,
    predictions_path: str | Path,
    npz_index_path: str | Path,
    feature_columns_path: str | Path,
    output_root: str | Path,
    num_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    """Create a complete XAI handoff package and return manifest data."""
    output_dir = ensure_dir(Path(output_root) / experiment_name)
    predictions_df = load_predictions(predictions_path)
    npz_index_df = load_npz_index(npz_index_path)
    selected = select_xai_handoff_samples(predictions_df, npz_index_df, num_samples, random_seed)
    selected_patch_ids = set(selected["patch_id"])
    sample_predictions = predictions_df[predictions_df["patch_id"].isin(selected_patch_ids)].copy()
    feature_info = create_feature_info(feature_columns_path)

    checkpoint_out = copy_checkpoint(checkpoint_path, output_dir)
    config_out = copy_model_config(
        model_config_path,
        output_dir,
        minimal_config=_minimal_config(model_name, experiment_name),
    )
    feature_info_path = output_dir / "feature_info.json"
    _write_json(feature_info, feature_info_path)
    sample_predictions_csv = output_dir / "sample_predictions.csv"
    sample_metadata_csv = output_dir / "sample_metadata.csv"
    sample_predictions.to_csv(sample_predictions_csv, index=False, encoding="utf-8")
    selected.to_csv(sample_metadata_csv, index=False, encoding="utf-8")
    sample_predictions_parquet = _try_write_parquet(sample_predictions, output_dir / "sample_predictions.parquet")
    sample_metadata_parquet = _try_write_parquet(selected, output_dir / "sample_metadata.parquet")

    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_out),
        "model_config_path": str(config_out),
        "feature_info_path": str(feature_info_path),
        "sample_predictions_path": str(sample_predictions_parquet or sample_predictions_csv),
        "sample_metadata_path": str(sample_metadata_parquet or sample_metadata_csv),
        "created_at": created_at,
        "num_selected_samples": int(len(selected)),
        "required_input_keys": ["image_t1", "image_t2", "tabular"],
        "expected_model_call": "model(image_t1, image_t2, tabular, return_features=True)",
    }
    manifest_path = output_dir / "handoff_manifest.json"
    _write_json(manifest, manifest_path)
    readme_path = output_dir / "README_XAI_HANDOFF.md"
    readme_path.write_text(_handoff_readme(experiment_name, model_name, feature_info), encoding="utf-8")
    manifest["output_dir"] = str(output_dir)
    manifest["manifest_path"] = str(manifest_path)
    manifest["readme_path"] = str(readme_path)
    manifest["saved_files"] = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    return manifest


def _sample_group(df: pd.DataFrame, n: int, random_seed: int, reason: str) -> pd.DataFrame:
    if df.empty or n <= 0:
        return pd.DataFrame()
    sample = df.sample(n=min(n, len(df)), random_state=random_seed).copy()
    sample["selection_reason"] = reason
    return sample


def _sample_across_values(df: pd.DataFrame, column: str, per_bucket: int, random_seed: int) -> pd.DataFrame:
    parts = []
    values = [value for value in df[column].dropna().unique()]
    if not values:
        return pd.DataFrame()
    n_per_value = max(1, per_bucket // max(1, len(values)))
    for value in values:
        parts.append(_sample_group(df[df[column] == value], n_per_value, random_seed, f"diverse_{column}"))
    return _combine_selected(parts)


def _combine_selected(parts: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [part for part in parts if part is not None and not part.empty]
    if not non_empty:
        return pd.DataFrame()
    combined = pd.concat(non_empty, ignore_index=True)
    if "patch_id" in combined.columns:
        combined = combined.drop_duplicates("patch_id")
    return combined


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in list(output.columns):
        if not column.endswith("_index"):
            continue
        base = column[: -len("_index")]
        if base in output.columns:
            output[base] = output[base].combine_first(output[column])
            output = output.drop(columns=[column])
        else:
            output = output.rename(columns={column: base})
    return output


def _extract_feature_columns(payload: dict[str, Any]) -> list[str]:
    for key in ["tabular_feature_columns", "feature_columns", "processed_feature_columns", "scaled_feature_columns"]:
        if isinstance(payload.get(key), list):
            return list(payload[key])
    raw = payload.get("raw_feature_columns", [])
    encoded = payload.get("encoded_feature_columns", [])
    columns = []
    if isinstance(raw, list):
        columns.extend(raw)
    if isinstance(encoded, list):
        columns.extend(encoded)
    return columns


def _minimal_config(model_name: str, experiment_name: str) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "experiment_name": experiment_name,
        "output_mode": "regression",
        "image_channels": 13,
        "tabular_dim": 146,
        "image_embedding_dim": 256,
        "tabular_embedding_dim": 128,
        "fusion_dim": 256,
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def _try_write_parquet(df: pd.DataFrame, path: Path) -> Path | None:
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        return None


def _handoff_readme(experiment_name: str, model_name: str, feature_info: dict[str, Any]) -> str:
    bands = "\n".join(f"{index + 1}. {name}" for index, name in enumerate(feature_info["sentinel_band_names"]))
    return f"""# XAI Handoff Package

Experiment: `{experiment_name}`
Model: `{model_name}`

## Files

- `best.pt`: trained model checkpoint for explanation only.
- `model_config.json`: model construction/config metadata.
- `feature_info.json`: Sentinel band order, tabular feature columns, class mapping, and input schema.
- `sample_predictions.csv` / `sample_predictions.parquet`: selected prediction rows for XAI analysis.
- `sample_metadata.csv` / `sample_metadata.parquet`: selected sample metadata joined with NPZ paths.
- `handoff_manifest.json`: machine-readable package manifest.
- `README_XAI_HANDOFF.md`: this handoff guide.

## Loading Samples

Use `sample_metadata.csv` to locate each `npz_path`. Each `.npz` sample contains:

- `image_t1`: `[13, 128, 128]`
- `image_t2`: `[13, 128, 128]`
- `tabular`: `[146]`
- `change_ratio`
- `patch_id`, `pair_id`, `district`, `split`, `change_class`, `pair_type`, `time_gap_group`

Expected model call:

```python
outputs = model(image_t1, image_t2, tabular, return_features=True)
```

## Sentinel Band Order

{bands}

## Target Meaning

`change_ratio` is the patch-level land-cover change ratio between t1 and t2. Change classes map to:

- low: 0
- medium: 1
- high: 2

## Important Warning

Do not retrain the model for XAI. This package is for explaining the existing trained checkpoint only.
"""
