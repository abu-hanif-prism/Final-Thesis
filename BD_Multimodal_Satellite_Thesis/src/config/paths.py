"""Path and YAML helpers for project configuration."""

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on environment setup.
    raise ImportError("PyYAML is required to load YAML configuration files.") from exc


PATH_KEYS = {
    "google_drive_root",
    "sentinel_drive_dir",
    "dynamic_world_drive_dir",
    "local_project_root",
    "tabular_raw_path",
    "metadata_dir",
    "output_dir",
    "checkpoint_dir",
    "log_dir",
}

LOCAL_OUTPUT_DIR_KEYS = {
    "metadata_dir",
    "output_dir",
    "checkpoint_dir",
    "log_dir",
}


def _project_root() -> Path:
    """Return the repository root inferred from this module location."""
    return Path(__file__).resolve().parents[2]


def _resolve_config_path(path: str | Path) -> Path:
    """Resolve a config path from an absolute path, CWD, or project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    return _project_root() / candidate


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file using UTF-8 encoding."""
    yaml_path = _resolve_config_path(path)
    with yaml_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_project_paths(config_path: str | Path = "configs/paths.yaml") -> dict[str, Any]:
    """Load project paths and convert configured path values to Path objects.

    Local handoff/output directories are created if they do not already exist.
    """
    paths = load_yaml(config_path)

    for key in PATH_KEYS:
        if key in paths and paths[key] is not None:
            paths[key] = Path(paths[key])

    for key in LOCAL_OUTPUT_DIR_KEYS:
        if key in paths:
            paths[key].mkdir(parents=True, exist_ok=True)

    return paths
