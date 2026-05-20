"""Filename parsing utilities for seasonal satellite raster files."""

from pathlib import Path
import re
from typing import Any


VALID_YEARS = set(range(2016, 2026))
VALID_EXTENSIONS = {".tif", ".tiff"}
VALID_SENSORS = {"S2", "Sentinel", "Sentinel2", "DW", "DynamicWorld"}

SENSOR_ALIASES = {
    "s2": "S2",
    "sentinel": "Sentinel",
    "sentinel2": "Sentinel2",
    "dw": "DW",
    "dynamicworld": "DynamicWorld",
}

SEASON_ALIASES = {
    "winter": "Winter",
    "premonsoon": "PreMonsoon",
    "pre_monsoon": "PreMonsoon",
    "monsoon": "Monsoon",
    "postmonsoon": "PostMonsoon",
    "post_monsoon": "PostMonsoon",
}

SEASON_PREFIXES = sorted(SEASON_ALIASES, key=len, reverse=True)


def normalize_district_name(name: str) -> str:
    """Normalize a district name while keeping readable capitalization."""
    cleaned = str(name).strip().replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    title_cased = cleaned.title()
    underscored = title_cased.replace(" ", "_")
    return re.sub(r"_+", "_", underscored).strip("_")


def normalize_season_name(name: str) -> str:
    """Normalize supported season aliases to the canonical season name."""
    key = str(name).strip().replace(" ", "_").lower()
    key = re.sub(r"_+", "_", key)

    if key in SEASON_ALIASES:
        return SEASON_ALIASES[key]

    compact_key = key.replace("_", "")
    if compact_key in SEASON_ALIASES:
        return SEASON_ALIASES[compact_key]

    raise ValueError(
        "Invalid season "
        f"'{name}'. Expected one of: Winter, PreMonsoon, Monsoon, PostMonsoon."
    )


def build_image_id(district: str, year: int | str, season: str) -> str:
    """Build a stable image identifier from district, year, and season."""
    normalized_district = normalize_district_name(district)
    normalized_season = normalize_season_name(season)
    return f"{normalized_district}_{int(year)}_{normalized_season}"


def parse_satellite_filename(filepath: str | Path) -> dict[str, Any]:
    """Parse a satellite raster filename into normalized metadata fields.

    Parsing errors are returned in the output dictionary rather than raised.
    """
    path = Path(filepath)
    result = _empty_result(path)

    try:
        _validate_extension(path.suffix)

        sensor_raw, resolution, year_raw, season_and_district = _split_filename_stem(
            path.stem
        )
        sensor = _normalize_sensor(sensor_raw)
        year = _normalize_year(year_raw)
        season_raw, district_raw = _split_season_and_district(season_and_district)
        season = normalize_season_name(season_raw)
        district = normalize_district_name(district_raw)

        if not district:
            raise ValueError("District name is missing.")

        result.update(
            {
                "sensor": sensor,
                "resolution": resolution,
                "year": year,
                "season": season,
                "district": district,
                "image_id": build_image_id(district, year, season),
                "parse_status": "success",
                "error_message": None,
            }
        )
    except Exception as exc:  # noqa: BLE001 - parser must not crash callers.
        result["parse_status"] = "failed"
        result["error_message"] = str(exc)

    return result


def _empty_result(path: Path) -> dict[str, Any]:
    """Create the common parser output structure."""
    return {
        "filename": path.name,
        "stem": path.stem,
        "sensor": None,
        "resolution": None,
        "year": None,
        "season": None,
        "district": None,
        "image_id": None,
        "extension": path.suffix.lower(),
        "parse_status": "failed",
        "error_message": None,
    }


def _validate_extension(extension: str) -> None:
    """Validate supported raster filename extensions."""
    if extension.lower() not in VALID_EXTENSIONS:
        raise ValueError(
            f"Invalid extension '{extension}'. Expected .tif or .tiff."
        )


def _normalize_sensor(sensor: str) -> str:
    """Normalize supported sensor aliases."""
    key = sensor.strip().lower()
    if key not in SENSOR_ALIASES:
        allowed = ", ".join(sorted(VALID_SENSORS))
        raise ValueError(f"Invalid sensor '{sensor}'. Expected one of: {allowed}.")
    return SENSOR_ALIASES[key]


def _normalize_year(year: str) -> int:
    """Validate and convert the supported year range."""
    if not year.isdigit():
        raise ValueError(f"Invalid year '{year}'. Expected a four-digit year.")

    parsed_year = int(year)
    if parsed_year not in VALID_YEARS:
        raise ValueError("Invalid year. Expected a year from 2016 to 2025.")

    return parsed_year


def _split_filename_stem(stem: str) -> tuple[str, str, str, str]:
    """Split supported filename stems into sensor, resolution, year, and suffix."""
    if stem.lower().startswith("dw_allclasses_"):
        parts = stem.split("_", 4)
        if len(parts) != 5:
            raise ValueError(
                "Filename must follow "
                "DW_allclasses_{resolution}_{year}_{season}_{district}.tif format."
            )

        sensor_raw, class_token, resolution, year_raw, season_and_district = parts
        if class_token.lower() != "allclasses":
            raise ValueError("Invalid Dynamic World class token. Expected allclasses.")

        return sensor_raw, resolution, year_raw, season_and_district

    parts = stem.split("_", 3)
    if len(parts) != 4:
        raise ValueError(
            "Filename must follow "
            "{sensor}_{resolution}_{year}_{season}_{district}.tif format."
        )

    return parts[0], parts[1], parts[2], parts[3]


def _split_season_and_district(value: str) -> tuple[str, str]:
    """Split the season prefix from the district suffix."""
    normalized_value = value.strip()
    comparison_value = normalized_value.lower().replace(" ", "_")

    for prefix in SEASON_PREFIXES:
        if not comparison_value.startswith(prefix):
            continue

        prefix_length = len(prefix)
        next_char = comparison_value[prefix_length:prefix_length + 1]
        if next_char and next_char != "_":
            continue

        district = normalized_value[prefix_length:].lstrip("_ ")
        if not district:
            raise ValueError("District name is missing.")

        return normalized_value[:prefix_length], district

    raise ValueError(
        "Invalid season. Expected one of: Winter, PreMonsoon, Monsoon, PostMonsoon."
    )
