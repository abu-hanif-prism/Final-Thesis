"""Debug Dynamic World GeoTIFF band meanings using small raster windows."""

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from src.config.settings import load_all_configs  # noqa: E402
from src.utils.file_utils import ensure_dir  # noqa: E402


WINDOW_SIZE = 256
MAX_UNIQUE_DISPLAY = 20


def main() -> None:
    """Inspect several Dynamic World rasters and save a band debug report."""
    configs = load_all_configs()
    reports_dir = ensure_dir(configs.paths["output_dir"] / "reports")
    report_path = reports_dir / "dynamic_world_band_debug.txt"

    raster_paths = select_dynamic_world_files(configs.paths["metadata_dir"])
    lines = []
    lines.append("Dynamic World Band Debug Report")
    lines.append("=" * 40)
    lines.append("")

    inspected = []
    for path in raster_paths:
        raster_report = inspect_raster_bands(path)
        inspected.append(raster_report)
        lines.extend(format_raster_report(raster_report))
        lines.append("")

    conclusion = infer_band_meaning(inspected)
    lines.extend(conclusion)

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\nSaved report: {report_path}")


def select_dynamic_world_files(metadata_dir: Path) -> list[Path]:
    """Select at least three Dynamic World files from different districts/years."""
    inventory_path = metadata_dir / "inventory" / "dynamic_world_inventory.parquet"
    if not inventory_path.exists():
        raise FileNotFoundError(f"Missing Dynamic World inventory: {inventory_path}")

    inventory = pd.read_parquet(inventory_path)
    success = inventory[inventory["parse_status"] == "success"].copy()
    if success.empty:
        raise ValueError("No successfully parsed Dynamic World files found.")

    selected_rows = []
    used_districts = set()
    used_years = set()
    for _, row in success.sort_values(["district", "year", "season"]).iterrows():
        district = row["district"]
        year = int(row["year"])
        if district in used_districts or year in used_years:
            continue
        selected_rows.append(row)
        used_districts.add(district)
        used_years.add(year)
        if len(selected_rows) >= 3:
            break

    if len(selected_rows) < 3:
        selected_rows = [
            row
            for _, row in success.drop_duplicates(["district", "year"]).head(3).iterrows()
        ]

    if len(selected_rows) < 3:
        raise ValueError("Could not select at least three Dynamic World files.")

    return [Path(row["full_path"]) for row in selected_rows[:3]]


def inspect_raster_bands(path: Path) -> dict[str, object]:
    """Inspect metadata and small-window band statistics for one raster."""
    with rasterio.open(path) as dataset:
        width = int(dataset.width)
        height = int(dataset.height)
        window = Window(
            col_off=0,
            row_off=0,
            width=min(WINDOW_SIZE, width),
            height=min(WINDOW_SIZE, height),
        )
        band_reports = []
        for band_index in range(1, dataset.count + 1):
            array = dataset.read(band_index, window=window, boundless=False)
            band_reports.append(
                inspect_band_array(
                    array=array,
                    band_index=band_index,
                    description=dataset.descriptions[band_index - 1],
                    nodata=dataset.nodatavals[band_index - 1],
                )
            )

        return {
            "path": str(path),
            "width": width,
            "height": height,
            "band_count": int(dataset.count),
            "dtypes": list(dataset.dtypes),
            "nodata": list(dataset.nodatavals),
            "crs": str(dataset.crs) if dataset.crs else None,
            "descriptions": list(dataset.descriptions),
            "band_reports": band_reports,
        }


def inspect_band_array(
    array: np.ndarray,
    band_index: int,
    description: str | None,
    nodata: object,
) -> dict[str, object]:
    """Compute small-window statistics for one band."""
    values = array.astype("float64", copy=False).ravel()
    finite = values[np.isfinite(values)]
    if nodata is not None and not pd.isna(nodata):
        finite = finite[finite != float(nodata)]

    if finite.size == 0:
        return {
            "band_index": band_index,
            "description": description,
            "min": None,
            "max": None,
            "approx_unique_count": 0,
            "integer_like": False,
            "first_unique_values": [],
        }

    unique_values = np.unique(finite)
    integer_like = bool(np.all(np.isclose(unique_values, np.round(unique_values))))
    first_unique_values = []
    if integer_like:
        first_unique_values = [
            int(value) if float(value).is_integer() else float(value)
            for value in unique_values[:MAX_UNIQUE_DISPLAY]
        ]

    return {
        "band_index": band_index,
        "description": description,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "approx_unique_count": int(len(unique_values)),
        "integer_like": integer_like,
        "first_unique_values": first_unique_values,
    }


def format_raster_report(report: dict[str, object]) -> list[str]:
    """Format one raster report as text lines."""
    lines = [
        f"Raster path: {report['path']}",
        f"Width/height: {report['width']} x {report['height']}",
        f"Band count: {report['band_count']}",
        f"Dtype: {report['dtypes']}",
        f"Nodata: {report['nodata']}",
        f"CRS: {report['crs']}",
        "Bands:",
    ]
    for band in report["band_reports"]:
        unique_text = (
            f", first 20 unique values: {band['first_unique_values']}"
            if band["integer_like"]
            else ""
        )
        lines.append(
            "  "
            f"Band {band['band_index']}: "
            f"description={band['description']}, "
            f"min={band['min']}, max={band['max']}, "
            f"approx_unique_count={band['approx_unique_count']}, "
            f"integer_like={band['integer_like']}"
            f"{unique_text}"
        )
    return lines


def infer_band_meaning(reports: list[dict[str, object]]) -> list[str]:
    """Infer likely class-label and probability/one-hot bands from band stats."""
    by_band: dict[int, list[dict[str, object]]] = {}
    descriptions: dict[int, set[str]] = {}
    for report in reports:
        for band in report["band_reports"]:
            band_index = int(band["band_index"])
            by_band.setdefault(band_index, []).append(band)
            if band["description"]:
                descriptions.setdefault(band_index, set()).add(str(band["description"]))

    label_candidates = []
    probability_candidates = []
    one_hot_candidates = []
    for band_index, band_reports in by_band.items():
        mins = [band["min"] for band in band_reports if band["min"] is not None]
        maxes = [band["max"] for band in band_reports if band["max"] is not None]
        unique_counts = [int(band["approx_unique_count"]) for band in band_reports]
        all_integer_like = all(bool(band["integer_like"]) for band in band_reports)
        desc_text = " ".join(sorted(descriptions.get(band_index, []))).lower()
        min_value = min(mins) if mins else None
        max_value = max(maxes) if maxes else None
        max_unique = max(unique_counts) if unique_counts else 0

        if "label" in desc_text or (
            all_integer_like
            and min_value is not None
            and max_value is not None
            and 0 <= min_value
            and max_value <= 20
            and max_unique <= 25
        ):
            label_candidates.append(band_index)
        if (
            min_value is not None
            and max_value is not None
            and 0 <= min_value
            and max_value <= 1
            and max_unique > 20
        ):
            probability_candidates.append(band_index)
        if (
            all_integer_like
            and min_value in {0, 1}
            and max_value in {0, 1}
            and max_unique <= 2
        ):
            one_hot_candidates.append(band_index)

    recommended = label_candidates[-1] if label_candidates else None
    lines = [
        "Interpretation",
        "=" * 40,
        f"Likely class label band candidates: {label_candidates}",
        f"Likely probability band candidates: {probability_candidates}",
        f"Likely one-hot/binary band candidates: {one_hot_candidates}",
    ]
    if recommended is not None:
        lines.extend(
            [
                f"Recommended label_band for Prompt 12: {recommended}",
                "Reason: this band looks most like a discrete class-label band "
                "or is explicitly described as label.",
            ]
        )
    else:
        lines.extend(
            [
                "Recommended label_band for Prompt 12: unable to infer automatically",
                "Reason: no inspected band looked like a discrete class-label band.",
            ]
        )
    return lines


if __name__ == "__main__":
    main()
