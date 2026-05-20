"""Compute Dynamic World patch-level change labels with chunking and resume."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import json
import os
import sys
import time

import pandas as pd

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_all_configs  # noqa: E402
from src.patches.compute_change_ratio import compute_labels_for_chunk  # noqa: E402
from src.utils.file_utils import ensure_dir  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional local dependency.
    tqdm = None


PATCH_COLUMNS_FOR_LABELING = [
    "patch_id",
    "pair_id",
    "split",
    "pair_type",
    "time_gap_group",
    "dw_path_t1",
    "dw_path_t2",
    "x",
    "y",
    "patch_size",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for chunked label computation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Resume existing chunk run.")
    parser.add_argument("--chunk_size", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--end_index", type=int, default=None)
    # Dynamic World export layout for this project:
    # bands 1-9 = class probabilities, band 10 = dw_label_mode, band 11 = forest mask.
    parser.add_argument("--label_band", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--min_valid_pixel_ratio", type=float, default=None)
    parser.add_argument(
        "--input_patch_index",
        type=str,
        default="data/metadata/patches/patch_index_sampled.parquet",
        help=(
            "Patch index parquet to label. Defaults to sampled patch index. "
            "Use data/metadata/patches/patch_index_all.parquet to label all patches."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Recompute from scratch.")
    return parser.parse_args()


def main() -> None:
    """Run chunked Dynamic World label computation."""
    args = parse_args()
    configs = load_all_configs()
    metadata_dir = configs.paths["metadata_dir"]
    output_dir = configs.paths["output_dir"]
    patch_size = int(args.patch_size or configs.patching.get("patch_size", 128))
    min_valid_pixel_ratio = float(
        args.min_valid_pixel_ratio
        if args.min_valid_pixel_ratio is not None
        else configs.patching.get("min_valid_pixel_ratio", 0.80)
    )

    patch_dir = ensure_dir(metadata_dir / "patches")
    label_dir = ensure_dir(metadata_dir / "labels")
    chunk_dir = ensure_dir(label_dir / "chunks")
    reports_dir = ensure_dir(output_dir / "reports")
    progress_dir = ensure_dir(output_dir / "progress")
    progress_path = progress_dir / "compute_patch_labels_progress.json"

    if args.force:
        _clear_previous_outputs(label_dir, chunk_dir, patch_dir, reports_dir, progress_path)

    patch_index_path = _resolve_input_patch_index(args.input_patch_index, patch_dir)
    if not patch_index_path.exists():
        raise FileNotFoundError(
            f"Missing patch index: {patch_index_path}. "
            "Run scripts/10b_sample_patch_index.py first for the sampled index, "
            "or pass --input_patch_index data/metadata/patches/patch_index_all.parquet."
        )

    patch_index = pd.read_parquet(patch_index_path)
    patch_index = _slice_patch_index(patch_index, args)
    if "patch_size" not in patch_index.columns:
        patch_index["patch_size"] = patch_size

    chunks = _build_chunks(len(patch_index), args.chunk_size)
    progress = _load_or_create_progress(progress_path, args, len(patch_index))
    completed_chunks = {int(chunk_id) for chunk_id in progress.get("completed_chunks", [])}
    failed_chunks = list(progress.get("failed_chunks", []))
    skipped_count = 0

    print(f"Total patch rows: {len(patch_index)}")
    print(f"Input patch index path: {patch_index_path}")
    if patch_index_path.name == "patch_index_all.parquet":
        print("WARNING: Full patch index detected (~5M patches). This may take extremely long.")
    raster_pair_count = _raster_pair_count(patch_index)
    average_patches_per_raster_pair = (
        0.0 if raster_pair_count == 0 else len(patch_index) / raster_pair_count
    )
    print(f"Unique raster pair count: {raster_pair_count}")
    print(f"Average patches per raster pair: {average_patches_per_raster_pair:.2f}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Resume mode: {bool(args.resume)}")
    print(f"Using Dynamic World label band: {args.label_band}")

    run_start_time = time.time()
    iterator = chunks
    if tqdm is not None:
        iterator = tqdm(chunks, desc="Label chunks", unit="chunk")

    for chunk_id, start, end in iterator:
        chunk_path = _chunk_path(chunk_dir, chunk_id)
        if args.resume and chunk_path.exists():
            completed_chunks.add(chunk_id)
            skipped_count += 1
            _save_progress(
                _progress_payload(
                    args=args,
                    total_rows=len(patch_index),
                    completed_chunks=sorted(completed_chunks),
                    failed_chunks=failed_chunks,
                    completed_patch_count=_completed_patch_count(completed_chunks, chunks),
                    current_runtime_seconds=time.time() - run_start_time,
                ),
                progress_path,
            )
            continue

        try:
            chunk_df = patch_index.iloc[start:end][PATCH_COLUMNS_FOR_LABELING].copy()
            chunk_start_time = time.time()
            chunk_group_total = _raster_pair_count(chunk_df)
            average_chunk_patches_per_group = len(chunk_df) / max(1, chunk_group_total)
            print(
                f"\nChunk {chunk_id:06d} START\n"
                f"Rows: {len(chunk_df)}\n"
                f"Raster pair groups: {chunk_group_total}\n"
                f"Average patches per raster pair in chunk: "
                f"{average_chunk_patches_per_group:.2f}"
            )
            completed_before_chunk = _completed_patch_count(completed_chunks, chunks)

            def progress_callback(
                processed_count: int,
                raster_pair_group_index: int | None,
                raster_pair_group_total: int | None,
            ) -> None:
                _print_live_chunk_progress(
                    chunk_id=chunk_id,
                    processed_count=processed_count,
                    chunk_size=len(chunk_df),
                    chunk_start_time=chunk_start_time,
                    raster_pair_group_index=raster_pair_group_index,
                    raster_pair_group_total=raster_pair_group_total or chunk_group_total,
                )
                completed_patch_count = completed_before_chunk + processed_count
                progress_payload = _progress_payload(
                    args=args,
                    total_rows=len(patch_index),
                    completed_chunks=sorted(completed_chunks),
                    failed_chunks=failed_chunks,
                    completed_patch_count=completed_patch_count,
                    current_chunk_id=chunk_id,
                    current_chunk_progress_percent=(
                        processed_count / max(1, len(chunk_df)) * 100
                    ),
                    estimated_remaining_seconds=_estimate_remaining_seconds(
                        completed_patch_count,
                        len(patch_index),
                        run_start_time,
                    ),
                    current_runtime_seconds=time.time() - run_start_time,
                    current_raster_pair_group=raster_pair_group_index,
                    current_raster_pair_group_total=raster_pair_group_total
                    or chunk_group_total,
                )
                _save_progress(progress_payload, progress_path)

            labels = compute_labels_for_chunk(
                chunk_df,
                patch_size=patch_size,
                min_valid_pixel_ratio=min_valid_pixel_ratio,
                label_band=args.label_band,
                progress_callback=progress_callback,
                progress_interval=500,
            )
            labels.to_parquet(chunk_path, index=False)
            completed_chunks.add(chunk_id)
            failed_chunks = [
                failure
                for failure in failed_chunks
                if failure.get("chunk_id") != chunk_id
            ]
        except Exception as exc:  # noqa: BLE001 - record chunk failure and continue.
            failed_chunks = [
                failure
                for failure in failed_chunks
                if failure.get("chunk_id") != chunk_id
            ]
            failed_chunks.append({"chunk_id": chunk_id, "error_message": str(exc)})
        finally:
            chunk_runtime_seconds = (
                time.time() - chunk_start_time if "chunk_start_time" in locals() else 0.0
            )
            progress = _progress_payload(
                args=args,
                total_rows=len(patch_index),
                completed_chunks=sorted(completed_chunks),
                failed_chunks=failed_chunks,
                completed_patch_count=_completed_patch_count(completed_chunks, chunks),
                current_chunk_id=chunk_id,
                current_chunk_progress_percent=100.0
                if chunk_id in completed_chunks
                else 0.0,
                estimated_remaining_seconds=_estimate_remaining_seconds(
                    _completed_patch_count(completed_chunks, chunks),
                    len(patch_index),
                    run_start_time,
                ),
                current_runtime_seconds=time.time() - run_start_time,
                current_raster_pair_group=chunk_group_total
                if "chunk_group_total" in locals()
                else None,
                current_raster_pair_group_total=chunk_group_total
                if "chunk_group_total" in locals()
                else None,
            )
            _save_progress(progress, progress_path)
            if chunk_id in completed_chunks:
                _print_chunk_complete(
                    chunk_id=chunk_id,
                    chunk_runtime_seconds=chunk_runtime_seconds,
                    chunk_rows=end - start,
                    completed_chunks_count=len(completed_chunks),
                    total_chunks=len(chunks),
                    completed_patch_count=_completed_patch_count(completed_chunks, chunks),
                    total_patch_count=len(patch_index),
                    run_start_time=run_start_time,
                    raster_pair_group_total=chunk_group_total
                    if "chunk_group_total" in locals()
                    else None,
                )

    expected_chunk_ids = [chunk_id for chunk_id, _, _ in chunks]
    labels = _combine_chunks(chunk_dir, expected_chunk_ids)
    labels.to_parquet(label_dir / "patch_change_labels.parquet", index=False)

    labeled_patch_index = patch_index.merge(labels, on=["patch_id", "pair_id"], how="left")
    _fill_missing_label_rows(labeled_patch_index)
    labeled_patch_index.to_parquet(patch_dir / "patch_index_labeled.parquet", index=False)
    for split in ["train", "val", "test"]:
        labeled_patch_index[labeled_patch_index["split"] == split].to_parquet(
            patch_dir / f"patch_index_{split}_labeled.parquet",
            index=False,
        )

    summary = build_label_distribution_summary(labeled_patch_index, args.label_band)
    summary.to_csv(
        reports_dir / "label_distribution_summary.csv",
        index=False,
        encoding="utf-8",
    )
    errors = labeled_patch_index[
        labeled_patch_index["label_status"] != "success"
    ].copy()
    errors.to_csv(
        reports_dir / "patch_label_errors.csv",
        index=False,
        encoding="utf-8",
    )

    print_label_report(
        labels=labels,
        skipped_count=skipped_count,
        completed_count=len(completed_chunks),
        failed_count=len(failed_chunks),
    )


def build_label_distribution_summary(
    labeled_df: pd.DataFrame,
    label_band_used: int,
) -> pd.DataFrame:
    """Build label metric and distribution summary rows."""
    success = labeled_df[labeled_df["label_status"] == "success"]
    rows = [
        {
            "summary_type": "metric",
            "value": "label_band_used",
            "count": int(label_band_used),
        },
        {"summary_type": "metric", "value": "total_patches", "count": len(labeled_df)},
        {
            "summary_type": "metric",
            "value": "success_count",
            "count": int((labeled_df["label_status"] == "success").sum()),
        },
        {
            "summary_type": "metric",
            "value": "failed_count",
            "count": int((labeled_df["label_status"] == "failed").sum()),
        },
        {
            "summary_type": "metric",
            "value": "invalid_low_valid_ratio_count",
            "count": int((labeled_df["label_status"] == "invalid_low_valid_ratio").sum()),
        },
        {
            "summary_type": "metric",
            "value": "low_count",
            "count": int((labeled_df["change_class"] == "low").sum()),
        },
        {
            "summary_type": "metric",
            "value": "medium_count",
            "count": int((labeled_df["change_class"] == "medium").sum()),
        },
        {
            "summary_type": "metric",
            "value": "high_count",
            "count": int((labeled_df["change_class"] == "high").sum()),
        },
        {
            "summary_type": "metric",
            "value": "mean_change_ratio",
            "count": float(success["change_ratio"].mean(skipna=True)),
        },
        {
            "summary_type": "metric",
            "value": "median_change_ratio",
            "count": float(success["change_ratio"].median(skipna=True)),
        },
    ]
    rows.extend(_count_rows(labeled_df, ["split", "change_class"], "change_class_by_split"))
    rows.extend(
        _count_rows(labeled_df, ["pair_type", "change_class"], "change_class_by_pair_type")
    )
    rows.extend(
        _count_rows(
            labeled_df,
            ["time_gap_group", "change_class"],
            "change_class_by_time_gap_group",
        )
    )
    return pd.DataFrame(rows, columns=["summary_type", "value", "count"])


def print_label_report(
    labels: pd.DataFrame,
    skipped_count: int,
    completed_count: int,
    failed_count: int,
) -> None:
    """Print the requested label-computation summary."""
    success = labels[labels["label_status"] == "success"]
    print(f"Chunks completed: {completed_count}")
    print(f"Chunks skipped: {skipped_count}")
    print(f"Chunks failed: {failed_count}")
    print(f"Total labels created: {len(labels)}")
    print(f"Success count: {int((labels['label_status'] == 'success').sum())}")
    print(
        "Invalid count: "
        f"{int((labels['label_status'] == 'invalid_low_valid_ratio').sum())}"
    )
    print(f"Failed count: {int((labels['label_status'] == 'failed').sum())}")
    print(f"Low count: {int((labels['change_class'] == 'low').sum())}")
    print(f"Medium count: {int((labels['change_class'] == 'medium').sum())}")
    print(f"High count: {int((labels['change_class'] == 'high').sum())}")
    print(f"Mean change_ratio: {success['change_ratio'].mean(skipna=True)}")
    print(f"Median change_ratio: {success['change_ratio'].median(skipna=True)}")


def _slice_patch_index(patch_index: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Apply optional start/end/limit row slicing."""
    start = args.start_index if args.start_index is not None else 0
    end = args.end_index if args.end_index is not None else len(patch_index)
    sliced = patch_index.iloc[start:end].copy()
    if args.limit is not None:
        sliced = sliced.head(args.limit).copy()
    return sliced.reset_index(drop=True)


def _resolve_input_patch_index(input_patch_index: str, patch_dir: Path) -> Path:
    """Resolve the input patch index path from CLI argument."""
    candidate = Path(input_patch_index)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    return patch_dir / candidate.name


def _build_chunks(total_rows: int, chunk_size: int) -> list[tuple[int, int, int]]:
    """Build chunk tuples of chunk_id/start/end."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    return [
        (chunk_id, start, min(start + chunk_size, total_rows))
        for chunk_id, start in enumerate(range(0, total_rows, chunk_size))
    ]


def _raster_pair_count(df: pd.DataFrame) -> int:
    """Count unique Dynamic World raster-pair groups in a patch table."""
    required_columns = {"dw_path_t1", "dw_path_t2"}
    if df.empty or not required_columns.issubset(df.columns):
        return 0
    return int(df[["dw_path_t1", "dw_path_t2"]].drop_duplicates().shape[0])


def _print_live_chunk_progress(
    chunk_id: int,
    processed_count: int,
    chunk_size: int,
    chunk_start_time: float,
    raster_pair_group_index: int | None,
    raster_pair_group_total: int | None,
) -> None:
    """Print live progress for the current chunk."""
    elapsed = time.time() - chunk_start_time
    progress_fraction = processed_count / max(1, chunk_size)
    eta_seconds = (
        elapsed * (1.0 - progress_fraction) / progress_fraction
        if progress_fraction > 0
        else 0.0
    )
    print(
        f"\n[Chunk {chunk_id:06d}]\n"
        f"Processed: {processed_count} / {chunk_size} patches "
        f"({progress_fraction * 100:.1f}%)\n"
        f"Raster pair group: {raster_pair_group_index or 0} / "
        f"{raster_pair_group_total or 0}\n"
        f"Elapsed: {_format_duration(elapsed)}\n"
        f"ETA chunk finish: {_format_duration(eta_seconds)}"
    )


def _print_chunk_complete(
    chunk_id: int,
    chunk_runtime_seconds: float,
    chunk_rows: int,
    completed_chunks_count: int,
    total_chunks: int,
    completed_patch_count: int,
    total_patch_count: int,
    run_start_time: float,
    raster_pair_group_total: int | None,
) -> None:
    """Print summary after a chunk completes."""
    speed = chunk_rows / chunk_runtime_seconds if chunk_runtime_seconds > 0 else 0.0
    overall_progress = completed_patch_count / max(1, total_patch_count)
    estimated_remaining = _estimate_remaining_seconds(
        completed_patch_count,
        total_patch_count,
        run_start_time,
    )
    print(
        f"\nChunk {chunk_id:06d} COMPLETE\n"
        f"Chunk runtime: {_format_duration(chunk_runtime_seconds)}\n"
        f"Speed: {speed:.2f} patches/sec\n"
        f"Raster pair groups processed: {raster_pair_group_total or 0}\n"
        f"Completed chunks: {completed_chunks_count} / {total_chunks}\n"
        f"Completed patches: {completed_patch_count} / {total_patch_count}\n"
        f"Overall progress: {overall_progress * 100:.1f}%\n"
        f"Estimated remaining time: {_format_duration(estimated_remaining)}"
    )


def _completed_patch_count(
    completed_chunks: set[int],
    chunks: list[tuple[int, int, int]],
) -> int:
    """Compute completed patch rows from completed chunk IDs."""
    return int(
        sum(end - start for chunk_id, start, end in chunks if chunk_id in completed_chunks)
    )


def _estimate_remaining_seconds(
    completed_patch_count: int,
    total_patch_count: int,
    run_start_time: float,
) -> float:
    """Estimate remaining runtime from current average throughput."""
    if completed_patch_count <= 0:
        return 0.0
    elapsed = time.time() - run_start_time
    patches_per_second = completed_patch_count / max(elapsed, 1e-9)
    remaining_patches = max(0, total_patch_count - completed_patch_count)
    return remaining_patches / max(patches_per_second, 1e-9)


def _format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _combine_chunks(chunk_dir: Path, expected_chunk_ids: list[int]) -> pd.DataFrame:
    """Combine expected chunk parquet files into one label DataFrame."""
    tables = []
    missing = []
    for chunk_id in expected_chunk_ids:
        path = _chunk_path(chunk_dir, chunk_id)
        if path.exists():
            tables.append(pd.read_parquet(path))
        else:
            missing.append(chunk_id)
    if missing:
        print(f"Warning: missing chunk outputs during combine: {missing[:20]}")
    if not tables:
        raise RuntimeError("No label chunks were available to combine.")
    return pd.concat(tables, ignore_index=True)


def _fill_missing_label_rows(labeled_patch_index: pd.DataFrame) -> None:
    """Mark rows without a combined label as failed in-place."""
    missing_label_mask = labeled_patch_index["label_status"].isna()
    if not missing_label_mask.any():
        return

    labeled_patch_index.loc[missing_label_mask, "label_status"] = "failed"
    labeled_patch_index.loc[missing_label_mask, "change_class"] = "invalid"
    labeled_patch_index.loc[
        missing_label_mask,
        "label_error_message",
    ] = "Missing label output for patch, likely due to failed chunk."


def _chunk_path(chunk_dir: Path, chunk_id: int) -> Path:
    """Return one chunk output path."""
    return chunk_dir / f"labels_chunk_{chunk_id:06d}.parquet"


def _load_or_create_progress(
    progress_path: Path,
    args: argparse.Namespace,
    total_rows: int,
) -> dict[str, object]:
    """Load resume progress if requested, otherwise create a fresh payload."""
    if args.resume and progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return _progress_payload(args, total_rows, [], [])


def _progress_payload(
    args: argparse.Namespace,
    total_rows: int,
    completed_chunks: list[int],
    failed_chunks: list[dict[str, object]],
    completed_patch_count: int = 0,
    current_chunk_id: int | None = None,
    current_chunk_progress_percent: float = 0.0,
    estimated_remaining_seconds: float = 0.0,
    current_runtime_seconds: float = 0.0,
    current_raster_pair_group: int | None = None,
    current_raster_pair_group_total: int | None = None,
) -> dict[str, object]:
    """Create a progress JSON payload."""
    overall_progress_percent = (
        0.0 if total_rows == 0 else completed_patch_count / total_rows * 100
    )
    return {
        "total_rows": int(total_rows),
        "chunk_size": int(args.chunk_size),
        "completed_chunks": completed_chunks,
        "failed_chunks": failed_chunks,
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "command_args": vars(args),
        "completed_patch_count": int(completed_patch_count),
        "total_patch_count": int(total_rows),
        "overall_progress_percent": float(overall_progress_percent),
        "current_chunk_id": current_chunk_id,
        "current_chunk_progress_percent": float(current_chunk_progress_percent),
        "estimated_remaining_seconds": float(estimated_remaining_seconds),
        "current_runtime_seconds": float(current_runtime_seconds),
        "current_raster_pair_group": current_raster_pair_group,
        "current_raster_pair_group_total": current_raster_pair_group_total,
    }


def _save_progress(progress: dict[str, object], progress_path: Path) -> None:
    """Write progress JSON with UTF-8 encoding."""
    with progress_path.open("w", encoding="utf-8") as file:
        json.dump(progress, file, indent=2, ensure_ascii=False)


def _clear_previous_outputs(
    label_dir: Path,
    chunk_dir: Path,
    patch_dir: Path,
    reports_dir: Path,
    progress_path: Path,
) -> None:
    """Remove previous label outputs created by this script."""
    for path in chunk_dir.glob("labels_chunk_*.parquet"):
        path.unlink()
    for path in [
        label_dir / "patch_change_labels.parquet",
        patch_dir / "patch_index_labeled.parquet",
        patch_dir / "patch_index_train_labeled.parquet",
        patch_dir / "patch_index_val_labeled.parquet",
        patch_dir / "patch_index_test_labeled.parquet",
        reports_dir / "label_distribution_summary.csv",
        reports_dir / "patch_label_errors.csv",
        progress_path,
    ]:
        if path.exists():
            path.unlink()


def _count_rows(
    df: pd.DataFrame,
    columns: list[str],
    summary_type: str,
) -> list[dict[str, object]]:
    """Return grouped count rows for summary CSVs."""
    if df.empty or not set(columns).issubset(df.columns):
        return []
    grouped = df.groupby(columns, dropna=False).size().reset_index(name="count")
    return [
        {
            "summary_type": summary_type,
            "value": "|".join(str(row[column]) for column in columns),
            "count": int(row["count"]),
        }
        for _, row in grouped.iterrows()
    ]


if __name__ == "__main__":
    main()
