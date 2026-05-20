"""Configuration loading helpers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.paths import get_project_paths, load_yaml


CONFIG_FILES = {
    "paths": "paths.yaml",
    "data": "data.yaml",
    "pairing": "pairing.yaml",
    "patching": "patching.yaml",
    "storage": "storage.yaml",
    "training_base": "training_base.yaml",
    "training_cnn": "training_cnn.yaml",
    "training_swin": "training_swin.yaml",
    "training_convnext": "training_convnext.yaml",
    "training_maxvit": "training_maxvit.yaml",
}


@dataclass(frozen=True)
class Config:
    """Small container for all YAML-backed project configuration dictionaries."""

    paths: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    pairing: dict[str, Any] = field(default_factory=dict)
    patching: dict[str, Any] = field(default_factory=dict)
    storage: dict[str, Any] = field(default_factory=dict)
    training_base: dict[str, Any] = field(default_factory=dict)
    training_cnn: dict[str, Any] = field(default_factory=dict)
    training_swin: dict[str, Any] = field(default_factory=dict)
    training_convnext: dict[str, Any] = field(default_factory=dict)
    training_maxvit: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, dict[str, Any]]:
        """Return the full configuration as a plain dictionary."""
        return {
            "paths": self.paths,
            "data": self.data,
            "pairing": self.pairing,
            "patching": self.patching,
            "storage": self.storage,
            "training_base": self.training_base,
            "training_cnn": self.training_cnn,
            "training_swin": self.training_swin,
            "training_convnext": self.training_convnext,
            "training_maxvit": self.training_maxvit,
        }

    def __getitem__(self, key: str) -> dict[str, Any]:
        """Allow dictionary-style access to a named config section."""
        return self.as_dict()[key]


def _config_dir(config_dir: str | Path = "configs") -> Path:
    """Resolve the configuration directory from the current working directory."""
    candidate = Path(config_dir)
    if candidate.is_absolute():
        return candidate

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    return Path(__file__).resolve().parents[2] / candidate


def load_config(name: str, config_dir: str | Path = "configs") -> dict[str, Any]:
    """Load one YAML configuration by logical name or YAML filename."""
    file_name = CONFIG_FILES.get(name, name)
    config_path = Path(file_name)
    if not config_path.suffix:
        config_path = config_path.with_suffix(".yaml")

    if not config_path.is_absolute():
        config_path = _config_dir(config_dir) / config_path

    return load_yaml(config_path)


def load_all_configs(config_dir: str | Path = "configs") -> Config:
    """Load all stage-one project configuration files."""
    resolved_config_dir = _config_dir(config_dir)
    paths = get_project_paths(resolved_config_dir / CONFIG_FILES["paths"])

    return Config(
        paths=paths,
        data=load_config("data", resolved_config_dir),
        pairing=load_config("pairing", resolved_config_dir),
        patching=load_config("patching", resolved_config_dir),
        storage=load_config("storage", resolved_config_dir),
        training_base=load_config("training_base", resolved_config_dir),
        training_cnn=load_config("training_cnn", resolved_config_dir),
        training_swin=load_config("training_swin", resolved_config_dir),
        training_convnext=load_config("training_convnext", resolved_config_dir),
        training_maxvit=load_config("training_maxvit", resolved_config_dir),
    )
