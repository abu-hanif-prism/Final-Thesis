"""Shared constants for XAI handoff loading."""

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

DEFAULT_IMAGE_SHAPE = [13, 128, 128]
DEFAULT_NPZ_INDEX_PATH = "data/npz/final_npz_index.csv"
DEFAULT_XAI_OUTPUT_DIR = "outputs/xai"
CLASS_MAPPING = {"low": 0, "medium": 1, "high": 2}
