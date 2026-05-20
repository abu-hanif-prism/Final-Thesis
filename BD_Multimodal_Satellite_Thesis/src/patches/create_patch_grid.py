"""Patch coordinate grid utilities."""

import pandas as pd


def create_patch_grid(
    width: int,
    height: int,
    patch_size: int = 128,
    stride: int = 64,
) -> pd.DataFrame:
    """Generate full in-bounds top-left patch coordinates for a raster."""
    width = int(width)
    height = int(height)
    patch_size = int(patch_size)
    stride = int(stride)

    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive integers.")
    if width < patch_size or height < patch_size:
        return pd.DataFrame(columns=["x", "y", "patch_size", "stride"])

    x_values = range(0, width - patch_size + 1, stride)
    y_values = range(0, height - patch_size + 1, stride)
    rows = [
        {
            "x": int(x),
            "y": int(y),
            "patch_size": patch_size,
            "stride": stride,
        }
        for y in y_values
        for x in x_values
    ]
    grid = pd.DataFrame(rows, columns=["x", "y", "patch_size", "stride"])
    validate_patch_grid(grid, width, height, patch_size)
    return grid


def estimate_patch_count(
    width: int,
    height: int,
    patch_size: int = 128,
    stride: int = 64,
) -> int:
    """Estimate the number of full in-bounds patches for a raster."""
    width = int(width)
    height = int(height)
    patch_size = int(patch_size)
    stride = int(stride)

    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive integers.")
    if width < patch_size or height < patch_size:
        return 0

    x_count = ((width - patch_size) // stride) + 1
    y_count = ((height - patch_size) // stride) + 1
    return int(x_count * y_count)


def validate_patch_grid(
    grid_df: pd.DataFrame,
    width: int,
    height: int,
    patch_size: int,
) -> None:
    """Validate that all patch coordinates are fully inside raster bounds."""
    required_columns = {"x", "y"}
    missing = required_columns - set(grid_df.columns)
    if missing:
        raise ValueError(f"Patch grid is missing required columns: {sorted(missing)}")

    if grid_df.empty:
        return

    invalid_mask = (
        (grid_df["x"] < 0)
        | (grid_df["y"] < 0)
        | (grid_df["x"] + int(patch_size) > int(width))
        | (grid_df["y"] + int(patch_size) > int(height))
    )
    if invalid_mask.any():
        invalid_count = int(invalid_mask.sum())
        raise ValueError(f"Patch grid has {invalid_count} out-of-bounds patches.")
