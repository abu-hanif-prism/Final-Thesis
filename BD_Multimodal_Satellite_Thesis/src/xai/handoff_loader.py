"""Utilities for loading trained model packages for XAI checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.model_factory import create_model_from_config
from src.training.npz_dataset import load_npz_index


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file safely."""
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_experiment_name(model_name: str, experiment_name: str | None = None) -> str:
    """Return explicit experiment name or the default regression experiment name."""
    if experiment_name:
        return experiment_name
    return f"{model_name}_regression"


def load_model_config(experiment_name: str, checkpoint_dir: str | Path = "checkpoints") -> dict[str, Any]:
    """Load checkpoints/{experiment_name}_model_config.json."""
    config_path = Path(checkpoint_dir) / f"{experiment_name}_model_config.json"
    return load_json(config_path)


def load_model_checkpoint(
    experiment_name: str,
    checkpoint_dir: str | Path = "checkpoints",
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load checkpoints/{experiment_name}_best.pt."""
    checkpoint_path = Path(checkpoint_dir) / f"{experiment_name}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=torch.device(device))


def build_model_from_config(
    model_name: str,
    model_config: dict[str, Any],
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Build model from saved config and place it in eval mode."""
    config = dict(model_config)
    config.setdefault("model_name", model_name)
    model = create_model_from_config(config)
    model.to(torch.device(device))
    model.eval()
    return model


def load_trained_model(
    model_name: str,
    experiment_name: str | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """Load config, best checkpoint, and trained model state."""
    resolved_experiment = resolve_experiment_name(model_name, experiment_name)
    config = load_model_config(resolved_experiment, checkpoint_dir=checkpoint_dir)
    checkpoint = load_model_checkpoint(resolved_experiment, checkpoint_dir=checkpoint_dir, device=device)
    model = build_model_from_config(model_name, config, device=device)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(f"Checkpoint for {resolved_experiment} does not contain model_state_dict.")
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, checkpoint


def load_npz_sample(npz_path: str | Path) -> dict[str, Any]:
    """Load one NPZ sample and return model inputs plus metadata."""
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ sample not found: {path}")

    keys = [
        "image_t1",
        "image_t2",
        "tabular",
        "change_ratio",
        "change_class",
        "patch_id",
        "pair_id",
        "district",
        "split",
        "pair_type",
        "time_gap_group",
    ]
    sample: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as npz:
        for key in keys:
            if key not in npz.files:
                continue
            value = npz[key]
            if key in {"image_t1", "image_t2", "tabular"}:
                sample[key] = value
            else:
                sample[key] = _scalar_to_python(value)
    return sample


def prepare_xai_batch(sample: dict[str, Any], device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Convert one NPZ sample into a batched tensor input dictionary."""
    torch_device = torch.device(device)
    return {
        "image_t1": torch.as_tensor(sample["image_t1"], dtype=torch.float32).unsqueeze(0).to(torch_device),
        "image_t2": torch.as_tensor(sample["image_t2"], dtype=torch.float32).unsqueeze(0).to(torch_device),
        "tabular": torch.as_tensor(sample["tabular"], dtype=torch.float32).unsqueeze(0).to(torch_device),
    }


def get_test_samples(index_rows: Any, num_samples: int = 5) -> list[dict[str, Any]]:
    """Return first num_samples rows from the test split."""
    rows = _rows_from_index(index_rows)
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("split") == "test":
            selected.append(row)
        if len(selected) >= int(num_samples):
            break
    return selected


def _rows_from_index(index_rows: Any) -> list[dict[str, Any]]:
    if isinstance(index_rows, list):
        return [dict(row) for row in index_rows]
    if hasattr(index_rows, "to_dict"):
        return index_rows.to_dict(orient="records")
    raise TypeError(f"Unsupported index rows type: {type(index_rows).__name__}")


def _scalar_to_python(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return array.tolist()
