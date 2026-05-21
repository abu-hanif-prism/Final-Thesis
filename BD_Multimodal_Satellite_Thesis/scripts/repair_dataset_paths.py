"""Repair copied metadata paths after moving the dataset to local disk.

This script only replaces the dataset root prefix:

    G:\\My Drive\\BD_satalite_thesis_data
    G:/My Drive/BD_satalite_thesis_data

with:

    D:\\BD_satalite_thesis_data

It scans metadata files, creates backups before writes, and reports path
existence checks for common path columns.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - script can still repair non-YAML files.
    yaml = None


OLD_PREFIXES = (
    r"G:\My Drive\BD_satalite_thesis_data",
    "G:/My Drive/BD_satalite_thesis_data",
)
NEW_PREFIX = r"D:\BD_satalite_thesis_data"

TARGET_DIRS = (
    Path("data/metadata"),
    Path("data/tabular/processed"),
)
BACKUP_DIR = Path("data/metadata/path_repair_backups")
EXTENSIONS = {".parquet", ".csv", ".json", ".yaml", ".yml", ".txt"}

PATH_COLUMNS = {
    "full_path",
    "sentinel_path",
    "dw_path",
    "sentinel_path_t1",
    "sentinel_path_t2",
    "dw_path_t1",
    "dw_path_t2",
    "npz_path",
}

IMPORTANT_FILES = (
    Path("data/metadata/patches/patch_index_sampled.parquet"),
    Path("data/metadata/final/final_patch_dataset.parquet"),
    Path("data/metadata/pairs/constrained_pair_index_tabular_complete.parquet"),
)


@dataclass
class FileResult:
    path: Path
    scanned: bool = True
    contains_old_path: bool = False
    modified: bool = False
    replacements: int = 0
    remaining_old_occurrences: int = 0
    path_values_checked: int = 0
    paths_exist: int = 0
    paths_missing: int = 0
    backup_path: Path | None = None
    error: str | None = None


@dataclass
class Summary:
    files_scanned: int = 0
    files_containing_old_path: int = 0
    files_modified: int = 0
    total_replacements: int = 0
    remaining_old_occurrences: int = 0
    path_values_checked: int = 0
    paths_exist: int = 0
    paths_missing: int = 0
    backups_created: list[Path] = field(default_factory=list)
    backups_reused: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair Google Drive dataset paths in metadata files."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry_run", action="store_true", help="Report changes only.")
    mode.add_argument("--apply", action="store_true", help="Apply path replacements.")
    parser.add_argument(
        "--force_backup",
        action="store_true",
        help="Overwrite existing backups before modifying files.",
    )
    return parser.parse_args()


def old_count(text: str) -> int:
    return sum(text.count(prefix) for prefix in OLD_PREFIXES)


def repair_text_value(value: str) -> tuple[str, int]:
    replacements = old_count(value)
    repaired = value
    for prefix in OLD_PREFIXES:
        repaired = repaired.replace(prefix, NEW_PREFIX)
    return repaired, replacements


def iter_target_files(root: Path) -> list[Path]:
    files: list[Path] = []
    backup_root = root / BACKUP_DIR
    for target_dir in TARGET_DIRS:
        directory = root / target_dir
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if backup_root in path.parents:
                continue
            if ".backup" in path.name:
                continue
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                files.append(path)
    return sorted(set(files))


def backup_path_for(root: Path, path: Path) -> Path:
    rel = path.relative_to(root)
    backup_name = "_".join(rel.parts)
    return root / BACKUP_DIR / backup_name


def ensure_backup(root: Path, path: Path, force_backup: bool) -> tuple[Path, str]:
    backup_path = backup_path_for(root, path)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if backup_path.exists() and not force_backup:
        return backup_path, "reused"
    shutil.copy2(path, backup_path)
    return backup_path, "created"


def string_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            columns.append(str(col))
    return columns


def check_path_values(df: pd.DataFrame) -> tuple[int, int, int]:
    checked = 0
    exists = 0
    missing = 0
    seen: set[str] = set()
    for col in PATH_COLUMNS.intersection(map(str, df.columns)):
        for value in df[col].dropna().astype(str).unique():
            if not value.startswith(NEW_PREFIX):
                continue
            if value in seen:
                continue
            seen.add(value)
            checked += 1
            if Path(value).exists():
                exists += 1
            else:
                missing += 1
    return checked, exists, missing


def repair_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    replacements = 0
    repaired = df.copy()
    for col in string_columns(repaired):
        series = repaired[col].astype("string")
        for prefix in OLD_PREFIXES:
            mask = series.str.contains(prefix, regex=False, na=False)
            replacements += int(mask.sum())
            series = series.str.replace(prefix, NEW_PREFIX, regex=False)
        repaired[col] = series

    remaining = 0
    for col in string_columns(repaired):
        series = repaired[col].astype("string")
        for prefix in OLD_PREFIXES:
            remaining += int(series.str.contains(prefix, regex=False, na=False).sum())
    return repaired, replacements, remaining


def repair_nested_strings(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        return repair_text_value(value)
    if isinstance(value, list):
        repaired_items = []
        replacements = 0
        for item in value:
            repaired, count = repair_nested_strings(item)
            repaired_items.append(repaired)
            replacements += count
        return repaired_items, replacements
    if isinstance(value, dict):
        repaired_dict = {}
        replacements = 0
        for key, item in value.items():
            repaired, count = repair_nested_strings(item)
            repaired_dict[key] = repaired
            replacements += count
        return repaired_dict, replacements
    return value, 0


def remaining_in_nested(value: Any) -> int:
    if isinstance(value, str):
        return old_count(value)
    if isinstance(value, list):
        return sum(remaining_in_nested(item) for item in value)
    if isinstance(value, dict):
        return sum(remaining_in_nested(item) for item in value.values())
    return 0


def repair_parquet(path: Path, apply: bool) -> FileResult:
    result = FileResult(path=path)
    df = pd.read_parquet(path)
    repaired, replacements, remaining = repair_dataframe(df)
    result.replacements = replacements
    result.contains_old_path = replacements > 0
    result.remaining_old_occurrences = remaining
    checked, exists, missing = check_path_values(repaired)
    result.path_values_checked = checked
    result.paths_exist = exists
    result.paths_missing = missing
    if apply and replacements:
        repaired.to_parquet(path, index=False)
        result.modified = True
    return result


def repair_csv(path: Path, apply: bool) -> FileResult:
    result = FileResult(path=path)
    df = pd.read_csv(path, dtype="string", keep_default_na=False)
    repaired, replacements, remaining = repair_dataframe(df)
    result.replacements = replacements
    result.contains_old_path = replacements > 0
    result.remaining_old_occurrences = remaining
    checked, exists, missing = check_path_values(repaired)
    result.path_values_checked = checked
    result.paths_exist = exists
    result.paths_missing = missing
    if apply and replacements:
        repaired.to_csv(path, index=False)
        result.modified = True
    return result


def repair_json(path: Path, apply: bool) -> FileResult:
    result = FileResult(path=path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    repaired, replacements = repair_nested_strings(data)
    result.replacements = replacements
    result.contains_old_path = replacements > 0
    result.remaining_old_occurrences = remaining_in_nested(repaired)
    if apply and replacements:
        with path.open("w", encoding="utf-8") as file:
            json.dump(repaired, file, indent=2, ensure_ascii=False)
            file.write("\n")
        result.modified = True
    return result


def repair_yaml(path: Path, apply: bool) -> FileResult:
    result = FileResult(path=path)
    if yaml is None:
        result.error = "PyYAML is not installed"
        return result
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    repaired, replacements = repair_nested_strings(data)
    result.replacements = replacements
    result.contains_old_path = replacements > 0
    result.remaining_old_occurrences = remaining_in_nested(repaired)
    if apply and replacements:
        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(repaired, file, sort_keys=False, allow_unicode=True)
        result.modified = True
    return result


def repair_txt(path: Path, apply: bool) -> FileResult:
    result = FileResult(path=path)
    text = path.read_text(encoding="utf-8")
    repaired, replacements = repair_text_value(text)
    result.replacements = replacements
    result.contains_old_path = replacements > 0
    result.remaining_old_occurrences = old_count(repaired)
    if apply and replacements:
        path.write_text(repaired, encoding="utf-8")
        result.modified = True
    return result


def repair_file(path: Path, apply: bool) -> FileResult:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return repair_parquet(path, apply)
    if suffix == ".csv":
        return repair_csv(path, apply)
    if suffix == ".json":
        return repair_json(path, apply)
    if suffix in {".yaml", ".yml"}:
        return repair_yaml(path, apply)
    if suffix == ".txt":
        return repair_txt(path, apply)
    return FileResult(path=path)


def add_to_summary(summary: Summary, result: FileResult) -> None:
    summary.files_scanned += int(result.scanned)
    summary.files_containing_old_path += int(result.contains_old_path)
    summary.files_modified += int(result.modified)
    summary.total_replacements += result.replacements
    summary.remaining_old_occurrences += result.remaining_old_occurrences
    summary.path_values_checked += result.path_values_checked
    summary.paths_exist += result.paths_exist
    summary.paths_missing += result.paths_missing
    if result.error:
        summary.errors.append(f"{result.path}: {result.error}")


def current_remaining_old_occurrences(result: FileResult, apply_changes: bool) -> int:
    if apply_changes and result.modified:
        return result.remaining_old_occurrences
    return result.replacements if result.contains_old_path else result.remaining_old_occurrences


def print_validation_examples(root: Path) -> None:
    print("\nValidation examples:")
    for rel_path in IMPORTANT_FILES:
        path = root / rel_path
        if not path.exists():
            print(f"- {rel_path}: missing file")
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - report-only branch.
            print(f"- {rel_path}: could not read parquet: {exc}")
            continue
        shown = 0
        print(f"- {rel_path}:")
        for col in PATH_COLUMNS.intersection(map(str, df.columns)):
            values = df[col].dropna().astype(str)
            for value in values[values.str.startswith(NEW_PREFIX)].head(10):
                old_removed = all(prefix not in value for prefix in OLD_PREFIXES)
                print(f"  {col} | old prefix removed: {old_removed} | exists: {Path(value).exists()}")
                print(f"    {value}")
                shown += 1
                if shown >= 10:
                    break
            if shown >= 10:
                break
        if shown == 0:
            print("  no repaired path examples found")


def main() -> None:
    args = parse_args()
    apply_changes = bool(args.apply)
    root = Path.cwd()
    files = iter_target_files(root)
    summary = Summary()
    changed_files: list[Path] = []
    needing_repair: list[Path] = []

    mode = "apply" if apply_changes else "dry_run"
    print(f"Mode: {mode}")
    print(f"Dataset replacement: {OLD_PREFIXES} -> {NEW_PREFIX}")

    for path in files:
        try:
            probe = repair_file(path, apply=False)
            if probe.contains_old_path:
                needing_repair.append(path)
            if apply_changes and probe.replacements:
                backup_path, backup_status = ensure_backup(root, path, args.force_backup)
                result = repair_file(path, apply=True)
                result.backup_path = backup_path
                if result.modified:
                    changed_files.append(path)
                    if backup_status == "created":
                        summary.backups_created.append(backup_path)
                    else:
                        summary.backups_reused.append(backup_path)
            else:
                result = probe
        except Exception as exc:  # Keep scanning other files.
            result = FileResult(path=path, error=str(exc))
        result.remaining_old_occurrences = current_remaining_old_occurrences(result, apply_changes)
        add_to_summary(summary, result)

    print("\nSummary:")
    print(f"total files scanned: {summary.files_scanned}")
    print(f"files containing old path: {summary.files_containing_old_path}")
    print(f"files modified: {summary.files_modified}")
    print(f"total replacements: {summary.total_replacements}")
    print(f"remaining old path occurrences: {summary.remaining_old_occurrences}")
    print(f"path-like values checked: {summary.path_values_checked}")
    print(f"paths that exist: {summary.paths_exist}")
    print(f"paths missing: {summary.paths_missing}")

    if needing_repair:
        print("\nFiles needing repair:")
        for path in needing_repair:
            print(f"- {path.relative_to(root)}")

    if changed_files:
        print("\nFiles modified:")
        for path in changed_files:
            print(f"- {path.relative_to(root)}")

    if summary.backups_created:
        print("\nBackups created:")
        for path in summary.backups_created:
            print(f"- {path.relative_to(root)}")

    if summary.backups_reused:
        print("\nBackups already existed and were reused:")
        for path in summary.backups_reused:
            print(f"- {path.relative_to(root)}")

    if summary.errors:
        print("\nErrors:")
        for error in summary.errors:
            print(f"- {error}")

    if apply_changes:
        print_validation_examples(root)


if __name__ == "__main__":
    main()
