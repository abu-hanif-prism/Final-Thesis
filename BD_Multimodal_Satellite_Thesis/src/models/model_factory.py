"""Unified factory for multimodal Siamese model variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch

from src.models.siamese_cnn import create_siamese_cnn_model
from src.models.siamese_convnext import create_siamese_convnext_model
from src.models.siamese_maxvit import create_siamese_maxvit_model
from src.models.siamese_swin import create_siamese_swin_model


SUPPORTED_MODELS = ("cnn", "swin", "convnext", "maxvit")
MODEL_ALIASES = {
    "cnn": "cnn",
    "siamese_cnn": "cnn",
    "multimodal_cnn": "cnn",
    "swin": "swin",
    "siamese_swin": "swin",
    "multimodal_swin": "swin",
    "convnext": "convnext",
    "siamese_convnext": "convnext",
    "multimodal_convnext": "convnext",
    "maxvit": "maxvit",
    "siamese_maxvit": "maxvit",
    "multimodal_maxvit": "maxvit",
}

SHARED_DEFAULTS = {
    "image_channels": 13,
    "tabular_dim": 146,
    "image_embedding_dim": 256,
    "tabular_embedding_dim": 128,
    "fusion_dim": 256,
    "num_classes": 3,
    "dropout": 0.2,
}

MODEL_DEFAULTS = {
    "cnn": {"base_channels": 32, "activation": "relu"},
    "swin": {"embed_dim": 96, "patch_size": 4, "depth": 2, "num_heads": 4},
    "convnext": {"depths": [2, 2, 3, 2], "dims": [32, 64, 128, 256]},
    "maxvit": {"dims": [64, 128, 256], "depths": [1, 1, 1], "num_heads": [4, 4, 8], "window_size": 8},
}

CONFIG_MODEL_KWARGS = {
    "cnn": {"base_channels"},
    "swin": {"embed_dim", "patch_size", "depth", "num_heads"},
    "convnext": {"depths", "dims"},
    "maxvit": {"dims", "depths", "num_heads", "window_size"},
}

MODEL_CREATORS: dict[str, Callable[..., torch.nn.Module]] = {
    "cnn": create_siamese_cnn_model,
    "swin": create_siamese_swin_model,
    "convnext": create_siamese_convnext_model,
    "maxvit": create_siamese_maxvit_model,
}


def normalize_model_name(model_name: str) -> str:
    """Normalize a model name or alias to cnn, swin, convnext, or maxvit."""
    normalized = str(model_name).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in MODEL_ALIASES:
        return MODEL_ALIASES[normalized]
    raise ValueError(
        f"Unsupported model_name {model_name!r}. "
        f"Supported names and aliases: {sorted(MODEL_ALIASES)}"
    )


def create_model(
    model_name: str,
    output_mode: str = "regression",
    image_channels: int = 13,
    tabular_dim: int = 146,
    image_embedding_dim: int = 256,
    tabular_embedding_dim: int = 128,
    fusion_dim: int = 256,
    num_classes: int = 3,
    dropout: float = 0.2,
    **model_kwargs: Any,
) -> torch.nn.Module:
    """Create a supported multimodal Siamese model by name."""
    canonical_name = normalize_model_name(model_name)
    defaults = MODEL_DEFAULTS[canonical_name].copy()
    allowed_model_kwargs = CONFIG_MODEL_KWARGS[canonical_name]
    constructor_defaults = {key: value for key, value in defaults.items() if key in allowed_model_kwargs}
    unsupported = sorted(set(model_kwargs) - allowed_model_kwargs)
    if unsupported:
        print(f"Ignoring unsupported kwargs for {canonical_name}: {unsupported}")
    constructor_defaults.update({key: value for key, value in model_kwargs.items() if key in allowed_model_kwargs})

    creator = MODEL_CREATORS[canonical_name]
    return creator(
        output_mode=output_mode,
        image_channels=image_channels,
        tabular_dim=tabular_dim,
        image_embedding_dim=image_embedding_dim,
        tabular_embedding_dim=tabular_embedding_dim,
        fusion_dim=fusion_dim,
        num_classes=num_classes,
        dropout=dropout,
        **constructor_defaults,
    )


def get_model_default_config(model_name: str) -> dict[str, Any]:
    """Return default config dictionary for one model family."""
    canonical_name = normalize_model_name(model_name)
    return {
        "model_name": model_name,
        "canonical_model_name": canonical_name,
        "output_mode": "regression",
        **SHARED_DEFAULTS,
        "model_kwargs": MODEL_DEFAULTS[canonical_name].copy(),
    }


def list_supported_models() -> dict[str, list[str]]:
    """Return supported canonical names and aliases."""
    return {
        "canonical": list(SUPPORTED_MODELS),
        "aliases": sorted(MODEL_ALIASES),
    }


def create_model_from_config(config: dict[str, Any]) -> torch.nn.Module:
    """Create a model from a saved or in-memory config dictionary."""
    model_kwargs = config.get("model_kwargs", {}) or {}
    canonical_name = normalize_model_name(config["model_name"])
    model_kwargs = {
        key: value
        for key, value in model_kwargs.items()
        if key in CONFIG_MODEL_KWARGS[canonical_name]
    }
    return create_model(
        model_name=config["model_name"],
        output_mode=config.get("output_mode", "regression"),
        image_channels=config.get("image_channels", SHARED_DEFAULTS["image_channels"]),
        tabular_dim=config.get("tabular_dim", SHARED_DEFAULTS["tabular_dim"]),
        image_embedding_dim=config.get("image_embedding_dim", SHARED_DEFAULTS["image_embedding_dim"]),
        tabular_embedding_dim=config.get("tabular_embedding_dim", SHARED_DEFAULTS["tabular_embedding_dim"]),
        fusion_dim=config.get("fusion_dim", SHARED_DEFAULTS["fusion_dim"]),
        num_classes=config.get("num_classes", SHARED_DEFAULTS["num_classes"]),
        dropout=config.get("dropout", SHARED_DEFAULTS["dropout"]),
        **model_kwargs,
    )


def save_model_config(config: dict[str, Any], path: str | Path) -> Path:
    """Save model config as UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
        file.write("\n")
    return path


def load_model_config(path: str | Path) -> dict[str, Any]:
    """Load model config JSON."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
