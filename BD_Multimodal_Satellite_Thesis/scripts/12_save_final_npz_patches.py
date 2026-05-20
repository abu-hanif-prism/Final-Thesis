"""Save final selected multimodal training samples as compressed NPZ files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import get_project_paths
from src.patches.save_npz_patches import estimate_npz_storage_bytes, process_patch_chunk
from src.utils.file_utils import ensure_dir


FINAL_PATCH_PATH = Path("data/metadata/final/final_patch_dataset.parquet")
TABULAR_FEATURE_PATH = Path("data/tabular/processed/pair_tabular_features_scaled_tabular_complete.parquet")
FEATURE_COLUMNS_PATH = Path("data/tabular/processed/pair_tabular_feature_columns.json")
NPZ_ROOT = Path("data/npz")
INDEX_OUTPUT = NPZ_ROOT / "final_npz_index.parquet"
SUMMARY_OUTPUT = Path("outputs/reports/npz_creation_summary.csv")
ERROR_OUTPUT = Path("outputs/reports/npz_creation_errors.csv")
PROGRESS_OUTPUT = Path("outputs/progress/save_npz_progress.json")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Skip existing NPZ files and continue.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing NPZ files.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Rows processed per chunk.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum rows to process.")
    parser.add_argument("--start_index", type=int, default=None, help="Optional start row index.")
    parser.add_argument("--end_index", type=int, default=None, help="Optional end row index, exclusive.")
    return parser.parse_args()


def format_duration(seconds):
    """Format seconds as HH:MM:SS."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_feature_columns(path):
    """Load the scaled tabular feature column ordering."""
    with Path(path).open("r", encoding="utf-8") as file:
        feature_info = json.load(file)

    feature_columns = feature_info.get("scaled_feature_columns")
    if not feature_columns:
        raise ValueError("No scaled_feature_columns found in pair_tabular_feature_columns.json")
    return feature_columns


def subset_patch_dataframe(df, start_index=None, end_index=None, limit=None):
    """Apply optional row range and limit filters."""
    if start_index is not None or end_index is not None:
        start = 0 if start_index is None else int(start_index)
        end = len(df) if end_index is None else int(end_index)
        df = df.iloc[start:end]

    if limit is not None:
        df = df.head(int(limit))

    return df.reset_index(drop=True)


def write_progress(path, progress):
    """Write progress JSON using UTF-8 encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    progress["last_updated"] = datetime.now().isoformat(timespec="seconds")
    with path.open("w", encoding="utf-8") as file:
        json.dump(progress, file, indent=2)


def write_summary(path, patch_df, index_df, stats, storage_bytes, elapsed_seconds):
    """Write a compact CSV summary for NPZ creation."""
    rows = [
        {"summary_type": "metric", "value": "total_requested", "count": int(len(patch_df))},
        {"summary_type": "metric", "value": "saved_count", "count": int(stats["saved_count"])},
        {"summary_type": "metric", "value": "skipped_count", "count": int(stats["skipped_count"])},
        {"summary_type": "metric", "value": "failed_count", "count": int(stats["failed_count"])},
        {"summary_type": "metric", "value": "indexed_npz_count", "count": int(len(index_df))},
        {"summary_type": "metric", "value": "storage_mb", "count": round(storage_bytes / (1024**2), 3)},
        {"summary_type": "metric", "value": "elapsed_seconds", "count": round(elapsed_seconds, 3)},
    ]

    if "split" in index_df.columns:
        for split, count in index_df["split"].value_counts().sort_index().items():
            rows.append({"summary_type": "split", "value": split, "count": int(count)})

    if "change_class" in index_df.columns:
        for change_class, count in index_df["change_class"].value_counts().sort_index().items():
            rows.append({"summary_type": "change_class", "value": change_class, "count": int(count)})

    pd.DataFrame(rows).to_csv(path, index=False)


def print_startup_summary(patch_df, chunk_size, args):
    """Print initial run details."""
    one_patch_bytes = 13 * 128 * 128 * 2
    image_pair_bytes = 2 * one_patch_bytes
    rough_storage_gb = len(patch_df) * image_pair_bytes / (1024**3)

    print(f"Total patches to save: {len(patch_df)}")
    print(f"Chunk size: {chunk_size}")
    print(f"Resume mode: {args.resume}")
    print(f"Force overwrite: {args.force}")
    print(f"Estimated uncompressed image-pair storage: {rough_storage_gb:.2f} GB")
    print("Input split counts:")
    print(patch_df["split"].value_counts().sort_index().to_string())
    print("Input change-class counts:")
    print(patch_df["change_class"].value_counts().sort_index().to_string())


def main():
    """Run final NPZ sample creation."""
    args = parse_args()
    paths = get_project_paths()
    project_root = Path(paths["local_project_root"])

    patch_path = project_root / FINAL_PATCH_PATH
    tabular_path = project_root / TABULAR_FEATURE_PATH
    feature_columns_path = project_root / FEATURE_COLUMNS_PATH
    npz_root = project_root / NPZ_ROOT
    index_output = project_root / INDEX_OUTPUT
    summary_output = project_root / SUMMARY_OUTPUT
    error_output = project_root / ERROR_OUTPUT
    progress_output = project_root / PROGRESS_OUTPUT

    for directory in [
        npz_root / "train",
        npz_root / "val",
        npz_root / "test",
        summary_output.parent,
        error_output.parent,
        progress_output.parent,
    ]:
        ensure_dir(directory)

    patch_df = pd.read_parquet(patch_path)
    patch_df = subset_patch_dataframe(patch_df, args.start_index, args.end_index, args.limit)

    tabular_df = pd.read_parquet(tabular_path)
    feature_columns = load_feature_columns(feature_columns_path)
    if "pair_id" not in tabular_df.columns:
        raise ValueError("Tabular feature file must contain pair_id.")
    tabular_df = tabular_df.set_index("pair_id", drop=False)

    print(f"GDAL_DISABLE_READDIR_ON_OPEN={os.environ.get('GDAL_DISABLE_READDIR_ON_OPEN')}")
    print(f"Final patch input: {patch_path}")
    print(f"Tabular feature input: {tabular_path}")
    print(f"NPZ root: {npz_root}")
    print_startup_summary(patch_df, args.chunk_size, args)

    start_time = time.time()
    all_index_parts = []
    all_error_parts = []
    stats = {"saved_count": 0, "skipped_count": 0, "failed_count": 0, "processed_count": 0}
    total_rows = len(patch_df)
    total_chunks = (total_rows + args.chunk_size - 1) // args.chunk_size if total_rows else 0

    for chunk_number, start in enumerate(range(0, total_rows, args.chunk_size), start=1):
        chunk_start_time = time.time()
        end = min(start + args.chunk_size, total_rows)
        chunk_df = patch_df.iloc[start:end].copy()

        def progress_callback(chunk_stats):
            elapsed = time.time() - start_time
            chunk_elapsed = time.time() - chunk_start_time
            processed = stats["processed_count"] + chunk_stats["processed_count"]
            samples_per_second = processed / elapsed if elapsed > 0 else 0.0
            remaining = (total_rows - processed) / samples_per_second if samples_per_second > 0 else 0.0
            chunk_percent = (chunk_stats["processed_count"] / len(chunk_df)) * 100 if len(chunk_df) else 100.0

            print(
                f"Chunk {chunk_number}/{total_chunks} | "
                f"{chunk_stats['processed_count']}/{len(chunk_df)} "
                f"({chunk_percent:.1f}%) | "
                f"saved {stats['saved_count'] + chunk_stats['saved_count']} | "
                f"skipped {stats['skipped_count'] + chunk_stats['skipped_count']} | "
                f"failed {stats['failed_count'] + chunk_stats['failed_count']} | "
                f"elapsed {format_duration(chunk_elapsed)} | ETA {format_duration(remaining)}"
            )

            write_progress(
                progress_output,
                {
                    "total_rows": int(total_rows),
                    "chunk_size": int(args.chunk_size),
                    "total_chunks": int(total_chunks),
                    "current_chunk": int(chunk_number),
                    "current_chunk_processed": int(chunk_stats["processed_count"]),
                    "current_chunk_progress_percent": float(round(chunk_percent, 3)),
                    "processed_count": int(processed),
                    "saved_count": int(stats["saved_count"] + chunk_stats["saved_count"]),
                    "skipped_count": int(stats["skipped_count"] + chunk_stats["skipped_count"]),
                    "failed_count": int(stats["failed_count"] + chunk_stats["failed_count"]),
                    "elapsed_seconds": float(round(elapsed, 3)),
                    "estimated_remaining_seconds": float(round(remaining, 3)),
                    "samples_per_second": float(round(samples_per_second, 3)),
                    "command_args": vars(args),
                },
            )

        index_part, error_part, chunk_stats = process_patch_chunk(
            chunk_df,
            tabular_df,
            feature_columns,
            npz_root,
            force=args.force,
            progress_callback=progress_callback,
            progress_interval=100,
        )

        all_index_parts.append(index_part)
        if not error_part.empty:
            all_error_parts.append(error_part)

        for key in stats:
            stats[key] += chunk_stats.get(key, 0)

        elapsed = time.time() - start_time
        processed = stats["processed_count"]
        samples_per_second = processed / elapsed if elapsed > 0 else 0.0
        remaining = (total_rows - processed) / samples_per_second if samples_per_second > 0 else 0.0
        chunk_elapsed = time.time() - chunk_start_time

        print(
            f"Chunk {chunk_number}/{total_chunks} complete | "
            f"processed {processed}/{total_rows} | "
            f"saved {stats['saved_count']} | skipped {stats['skipped_count']} | "
            f"failed {stats['failed_count']} | "
            f"chunk time {format_duration(chunk_elapsed)} | "
            f"{samples_per_second:.2f} samples/sec | ETA {format_duration(remaining)}"
        )

        write_progress(
            progress_output,
            {
                "total_rows": int(total_rows),
                "chunk_size": int(args.chunk_size),
                "total_chunks": int(total_chunks),
                "current_chunk": int(chunk_number),
                "processed_count": int(processed),
                "saved_count": int(stats["saved_count"]),
                "skipped_count": int(stats["skipped_count"]),
                "failed_count": int(stats["failed_count"]),
                "elapsed_seconds": float(round(elapsed, 3)),
                "estimated_remaining_seconds": float(round(remaining, 3)),
                "samples_per_second": float(round(samples_per_second, 3)),
                "command_args": vars(args),
            },
        )

    index_columns = ["patch_id", "npz_path", "split", "change_class", "change_ratio", "pair_id", "district"]
    error_columns = ["patch_id", "pair_id", "split", "error_message"]
    index_df = pd.concat(all_index_parts, ignore_index=True) if all_index_parts else pd.DataFrame(columns=index_columns)
    error_df = pd.concat(all_error_parts, ignore_index=True) if all_error_parts else pd.DataFrame(columns=error_columns)

    index_df.to_parquet(index_output, index=False)
    error_df.to_csv(error_output, index=False)

    storage_bytes = estimate_npz_storage_bytes(index_df.to_dict("records"))
    elapsed = time.time() - start_time
    write_summary(summary_output, patch_df, index_df, stats, storage_bytes, elapsed)

    print("NPZ creation complete.")
    print(f"Saved count: {stats['saved_count']}")
    print(f"Skipped count: {stats['skipped_count']}")
    print(f"Failed count: {stats['failed_count']}")
    if not index_df.empty:
        print("Final train/val/test counts:")
        print(index_df["split"].value_counts().sort_index().to_string())
        print("Final low/medium/high counts:")
        print(index_df["change_class"].value_counts().sort_index().to_string())
    print(f"Estimated storage used: {storage_bytes / (1024**3):.3f} GB")
    print(f"Average speed: {stats['processed_count'] / elapsed if elapsed > 0 else 0.0:.2f} samples/sec")
    print(f"NPZ index saved to: {index_output}")


if __name__ == "__main__":
    main()
