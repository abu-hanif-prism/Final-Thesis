"""General file utilities used across the project."""

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Any, path: str | Path) -> Path:
    """Save JSON data with UTF-8 encoding."""
    json_path = Path(path)
    ensure_dir(json_path.parent)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    return json_path


def load_json(path: str | Path) -> Any:
    """Load JSON data with UTF-8 encoding."""
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a DataFrame based on the output file extension."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8")
    elif suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif suffix == ".json":
        df.to_json(output_path, orient="records", indent=2, force_ascii=False)
    else:
        raise ValueError(f"Unsupported DataFrame format: {suffix}")

    return output_path


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame based on the input file extension."""
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(input_path, encoding="utf-8")
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix == ".json":
        return pd.read_json(input_path)

    raise ValueError(f"Unsupported DataFrame format: {suffix}")


def list_files_recursive(
    root: str | Path,
    extensions: Iterable[str] | str | None = None,
) -> list[Path]:
    """List files recursively, optionally filtering by extension."""
    root_path = Path(root)
    if extensions is None:
        allowed_extensions = None
    elif isinstance(extensions, str):
        allowed_extensions = {_normalize_extension(extensions)}
    else:
        allowed_extensions = {_normalize_extension(extension) for extension in extensions}

    files = [path for path in root_path.rglob("*") if path.is_file()]
    if allowed_extensions is not None:
        files = [
            path
            for path in files
            if path.suffix.lower() in allowed_extensions
        ]

    return sorted(files)


def _normalize_extension(extension: str) -> str:
    """Normalize an extension to lowercase with a leading dot."""
    normalized = extension.lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized
