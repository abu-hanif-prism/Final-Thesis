"""Recursive inventory scanner for Sentinel GeoTIFF files."""

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data_inventory.parse_filenames import parse_satellite_filename


INVENTORY_COLUMNS = [
    "image_id",
    "filename",
    "full_path",
    "sensor",
    "resolution",
    "year",
    "season",
    "district",
    "extension",
    "parse_status",
    "error_message",
]

RASTER_EXTENSIONS = {".tif", ".tiff"}


def scan_sentinel_files(sentinel_dir: str | Path) -> pd.DataFrame:
    """Recursively scan Sentinel GeoTIFF files and return an inventory table."""
    records = [_build_record(path) for path in _iter_raster_files(sentinel_dir)]
    return pd.DataFrame(records, columns=INVENTORY_COLUMNS)


def _iter_raster_files(root: str | Path) -> Iterable[Path]:
    """Yield supported raster files under a root directory."""
    root_path = Path(root)
    if not root_path.exists():
        return []

    return sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in RASTER_EXTENSIONS
    )


def _build_record(path: Path) -> dict[str, object]:
    """Parse one raster path into the inventory output schema."""
    parsed = parse_satellite_filename(path)
    return {
        "image_id": parsed["image_id"],
        "filename": parsed["filename"],
        "full_path": str(path.resolve()),
        "sensor": parsed["sensor"],
        "resolution": parsed["resolution"],
        "year": parsed["year"],
        "season": parsed["season"],
        "district": parsed["district"],
        "extension": parsed["extension"],
        "parse_status": parsed["parse_status"],
        "error_message": parsed["error_message"],
    }
