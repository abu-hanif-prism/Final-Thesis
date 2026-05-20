"""Project constants loaded from YAML configuration files."""

from src.config.settings import load_config

_DATA_CONFIG = load_config("data")
_PATCHING_CONFIG = load_config("patching")

BAND_NAMES = tuple(_DATA_CONFIG["band_names"])
SEASONS = tuple(_DATA_CONFIG["seasons"])
SEASON_ORDER = dict(_DATA_CONFIG["season_order"])
TARGET_COLUMN = _DATA_CONFIG["target_column"]
IMAGE_CHANNELS = int(_DATA_CONFIG["image_channels"])
PATCH_SIZE = int(_PATCHING_CONFIG["patch_size"])
