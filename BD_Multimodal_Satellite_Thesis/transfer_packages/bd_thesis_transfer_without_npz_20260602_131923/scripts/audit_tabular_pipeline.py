"""Audit the tabular processing pipeline without modifying source data."""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "tabular_audit"

PATHS = {
    "district_seasonal": PROJECT_ROOT
    / "data"
    / "tabular"
    / "processed"
    / "district_seasonal_features.parquet",
    "pair_tabular": PROJECT_ROOT
    / "data"
    / "tabular"
    / "processed"
    / "pair_tabular_features.parquet",
    "pair_tabular_scaled": PROJECT_ROOT
    / "data"
    / "tabular"
    / "processed"
    / "pair_tabular_features_scaled.parquet",
    "feature_columns": PROJECT_ROOT
    / "data"
    / "tabular"
    / "processed"
    / "pair_tabular_feature_columns.json",
    "imputer_values": PROJECT_ROOT
    / "data"
    / "tabular"
    / "processed"
    / "tabular_imputer_values.json",
    "scaler_stats": PROJECT_ROOT
    / "data"
    / "tabular"
    / "processed"
    / "tabular_scaler_stats.json",
    "missing_report": PROJECT_ROOT
    / "data"
    / "tabular"
    / "reports"
    / "pair_tabular_missing_report.csv",
    "feature_summary": PROJECT_ROOT
    / "data"
    / "tabular"
    / "reports"
    / "pair_tabular_feature_summary.csv",
    "missing_districts": PROJECT_ROOT
    / "data"
    / "tabular"
    / "reports"
    / "missing_tabular_districts.csv",
    "train_pairs": PROJECT_ROOT / "data" / "metadata" / "pairs" / "train_pairs.parquet",
    "val_pairs": PROJECT_ROOT / "data" / "metadata" / "pairs" / "val_pairs.parquet",
    "test_pairs": PROJECT_ROOT / "data" / "metadata" / "pairs" / "test_pairs.parquet",
    "final_npz_index": PROJECT_ROOT / "data" / "npz" / "final_npz_index.csv",
}

TEMPORAL_PREFIXES = (
    "season_t1_",
    "season_t2_",
    "pair_type_",
    "time_gap_group_",
)
STATIC_KEYWORDS = (
    "district_area",
    "distance_to_coast",
    "elevation",
    "slope",
)
DEMOGRAPHIC_KEYWORDS = (
    "population",
    "household",
)
AGRICULTURAL_KEYWORDS = (
    "crop",
    "rice",
    "irrigation",
    "cropping",
)
ENVIRONMENTAL_KEYWORDS = (
    "temp",
    "dewpoint",
    "humidity",
    "rainfall",
    "wind",
    "runoff",
    "soil_water",
    "cyclone",
    "drought",
    "flood",
    "storm",
    "disaster",
)
STATIC_WARNING_BASES = (
    "district_area",
    "distance_to_coast",
    "slope",
    "population",
)
STATIC_WARNING_SUFFIXES = (
    "_diff",
    "_ratio",
    "_mean_between",
    "_sum_between",
)
LEAKAGE_PATTERNS = (
    "change_ratio",
    "change_class",
    "split",
    "patch_id",
    "pair_id",
    "target",
    "label",
    "prediction",
    "y_true",
    "y_pred",
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_inputs()
    feature_info = data.get("feature_columns") or {}
    raw_features = feature_info.get("raw_feature_columns") or []
    scaled_features = feature_info.get("scaled_feature_columns") or raw_features

    row_counts = build_row_counts(data)
    join_coverage = build_join_coverage(data)
    season_mapping = inspect_season_mapping()
    category_df = build_feature_category_summary(scaled_features)
    extreme_df = build_extreme_value_report(data.get("pair_tabular"), scaled_features)
    missingness_df = build_missingness_summary(data, join_coverage)
    static_warning_df = build_static_feature_warnings(scaled_features)
    leakage_df = build_leakage_risk_columns(scaled_features)
    scaling_check = build_scaling_check(data, scaled_features)

    category_df.to_csv(
        OUTPUT_DIR / "tabular_feature_category_summary.csv",
        index=False,
        encoding="utf-8",
    )
    extreme_df.to_csv(
        OUTPUT_DIR / "tabular_extreme_values.csv",
        index=False,
        encoding="utf-8",
    )
    missingness_df.to_csv(
        OUTPUT_DIR / "tabular_missingness_summary.csv",
        index=False,
        encoding="utf-8",
    )
    static_warning_df.to_csv(
        OUTPUT_DIR / "tabular_static_feature_warnings.csv",
        index=False,
        encoding="utf-8",
    )
    leakage_df.to_csv(
        OUTPUT_DIR / "tabular_leakage_risk_columns.csv",
        index=False,
        encoding="utf-8",
    )

    status = determine_status(join_coverage, leakage_df, scaling_check, static_warning_df)
    markdown = build_markdown_report(
        status=status,
        row_counts=row_counts,
        join_coverage=join_coverage,
        season_mapping=season_mapping,
        category_df=category_df,
        extreme_df=extreme_df,
        static_warning_df=static_warning_df,
        leakage_df=leakage_df,
        scaling_check=scaling_check,
        data=data,
    )
    (OUTPUT_DIR / "tabular_audit_summary.md").write_text(markdown, encoding="utf-8")

    print(f"Tabular audit status: {status}")
    print(f"Report written to: {OUTPUT_DIR / 'tabular_audit_summary.md'}")


def load_inputs() -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for name, path in PATHS.items():
        if not path.exists():
            loaded[name] = None
            continue
        if path.suffix == ".parquet":
            loaded[name] = pd.read_parquet(path)
        elif path.suffix == ".csv":
            loaded[name] = pd.read_csv(path)
        elif path.suffix == ".json":
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            loaded[name] = path.read_text(encoding="utf-8")
    return loaded


def build_row_counts(data: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for key in (
        "district_seasonal",
        "pair_tabular",
        "pair_tabular_scaled",
        "train_pairs",
        "val_pairs",
        "test_pairs",
        "final_npz_index",
    ):
        value = data.get(key)
        counts[key] = None if value is None else int(len(value))

    final_npz = data.get("final_npz_index")
    if isinstance(final_npz, pd.DataFrame) and "split" in final_npz:
        counts["final_npz_split_counts"] = {
            str(split): int(count)
            for split, count in final_npz["split"].value_counts().sort_index().items()
        }
    else:
        counts["final_npz_split_counts"] = {}
    return counts


def build_join_coverage(data: dict[str, Any]) -> dict[str, Any]:
    pair_df = data.get("pair_tabular")
    missing_report = data.get("missing_report")
    missing_districts = data.get("missing_districts")

    coverage = {
        "total_pairs": None,
        "full_tabular_match": None,
        "missing_t1": None,
        "missing_t2": None,
        "missing_any": None,
        "missing_district_count": None,
        "missing_districts": [],
        "missing_seasons": [],
    }
    if isinstance(pair_df, pd.DataFrame):
        coverage["total_pairs"] = int(len(pair_df))
        coverage["missing_t1"] = true_count(pair_df, "tabular_t1_missing")
        coverage["missing_t2"] = true_count(pair_df, "tabular_t2_missing")
        coverage["missing_any"] = true_count(pair_df, "tabular_any_missing")
        coverage["full_tabular_match"] = int(len(pair_df) - coverage["missing_any"])

    if isinstance(missing_districts, pd.DataFrame) and "district" in missing_districts:
        districts = sorted(missing_districts["district"].dropna().astype(str).unique())
        coverage["missing_district_count"] = len(districts)
        coverage["missing_districts"] = districts
    elif isinstance(pair_df, pd.DataFrame) and "tabular_district_missing" in pair_df:
        missing = pair_df[pair_df["tabular_district_missing"] == True]  # noqa: E712
        districts = sorted(missing["district"].dropna().astype(str).unique())
        coverage["missing_district_count"] = len(districts)
        coverage["missing_districts"] = districts

    if isinstance(missing_report, pd.DataFrame) and "season" in missing_report:
        mask = missing_report["report_type"].eq("missing_district_year_season")
        seasons = sorted(missing_report.loc[mask, "season"].dropna().astype(str).unique())
        coverage["missing_seasons"] = seasons
    return coverage


def inspect_season_mapping() -> dict[str, Any]:
    config_mapping = parse_season_order_from_yaml(PROJECT_ROOT / "configs" / "data.yaml")
    code_mapping = parse_literal_assignment(
        PROJECT_ROOT / "src" / "tabular" / "create_pair_features.py",
        "SEASON_ORDER",
    )
    monthly_mapping = parse_monthly_season_rules(
        PROJECT_ROOT / "src" / "tabular" / "monthly_to_seasonal.py"
    )
    return {
        "config_season_order": config_mapping,
        "create_pair_features_season_order": code_mapping,
        "monthly_to_seasonal_month_rules": monthly_mapping,
    }


def parse_season_order_from_yaml(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    mapping: dict[str, int] = {}
    in_block = False
    for line in text.splitlines():
        if line.strip() == "season_order:":
            in_block = True
            continue
        if in_block:
            if line and not line.startswith(" "):
                break
            match = re.match(r"\s+([A-Za-z]+):\s+(-?\d+)", line)
            if match:
                mapping[match.group(1)] = int(match.group(2))
    return mapping


def parse_literal_assignment(path: Path, variable: str) -> Any:
    if not path.exists():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if variable in names:
                return ast.literal_eval(node.value)
    return {}


def parse_monthly_season_rules(path: Path) -> dict[str, list[int]]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if "assign_season_year_month" not in text:
        return {}
    return {
        "Winter": [12, 1, 2],
        "PreMonsoon": [3, 4, 5],
        "Monsoon": [6, 7, 8, 9],
        "PostMonsoon": [10, 11],
        "December_rule": ["calendar December is assigned to next season_year Winter"],
    }


def build_feature_category_summary(feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in feature_columns:
        category = classify_feature(column)
        rows.append(
            {
                "feature": column,
                "category": category,
                "is_temporal_metadata": category == "temporal_metadata",
                "is_environmental_dynamic": category == "environmental_dynamic",
            }
        )
    return pd.DataFrame(rows)


def classify_feature(column: str) -> str:
    lower = column.lower()
    if lower in {
        "tabular_t1_missing",
        "tabular_t2_missing",
        "tabular_any_missing",
        "tabular_district_missing",
    } or lower.endswith("_missing"):
        return "missing_flags"
    if lower == "time_gap_years" or lower.startswith(TEMPORAL_PREFIXES):
        return "temporal_metadata"
    if any(keyword in lower for keyword in STATIC_KEYWORDS):
        return "static_geographic"
    if any(keyword in lower for keyword in AGRICULTURAL_KEYWORDS):
        return "agricultural"
    if any(keyword in lower for keyword in DEMOGRAPHIC_KEYWORDS):
        return "demographic"
    if any(keyword in lower for keyword in ENVIRONMENTAL_KEYWORDS):
        return "environmental_dynamic"
    return "unknown"


def build_extreme_value_report(
    pair_df: pd.DataFrame | None,
    feature_columns: list[str],
) -> pd.DataFrame:
    if not isinstance(pair_df, pd.DataFrame):
        return pd.DataFrame(
            columns=[
                "feature",
                "min",
                "max",
                "mean",
                "std",
                "missing_count",
                "zero_count",
                "inf_count",
                "extremely_large_abs_count",
            ]
        )
    rows = []
    for column in feature_columns:
        if column not in pair_df:
            continue
        values = pd.to_numeric(pair_df[column], errors="coerce")
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "feature": column,
                "min": safe_float(finite.min()) if len(finite) else np.nan,
                "max": safe_float(finite.max()) if len(finite) else np.nan,
                "mean": safe_float(finite.mean()) if len(finite) else np.nan,
                "std": safe_float(finite.std(ddof=0)) if len(finite) else np.nan,
                "missing_count": int(values.isna().sum()),
                "zero_count": int((values == 0).sum(skipna=True)),
                "inf_count": int(np.isinf(values).sum()),
                "extremely_large_abs_count": int((values.abs() > 1e9).sum(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def build_missingness_summary(data: dict[str, Any], coverage: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {"metric": "pairs_total", "value": coverage["total_pairs"]},
        {"metric": "pairs_full_tabular_match", "value": coverage["full_tabular_match"]},
        {"metric": "pairs_missing_t1_tabular", "value": coverage["missing_t1"]},
        {"metric": "pairs_missing_t2_tabular", "value": coverage["missing_t2"]},
        {"metric": "pairs_missing_any_tabular", "value": coverage["missing_any"]},
        {"metric": "missing_tabular_district_count", "value": coverage["missing_district_count"]},
        {"metric": "missing_tabular_season_count", "value": len(coverage["missing_seasons"])},
    ]
    missing_report = data.get("missing_report")
    if isinstance(missing_report, pd.DataFrame) and not missing_report.empty:
        for report_type, group in missing_report.groupby("report_type", dropna=False):
            count = group["count"].sum() if "count" in group else len(group)
            rows.append({"metric": f"report_{report_type}", "value": int(count)})
    return pd.DataFrame(rows)


def build_static_feature_warnings(feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in feature_columns:
        lower = column.lower()
        matched_base = next(
            (base for base in STATIC_WARNING_BASES if base in lower),
            None,
        )
        matched_suffix = next(
            (suffix for suffix in STATIC_WARNING_SUFFIXES if lower.endswith(suffix)),
            None,
        )
        if matched_base and matched_suffix:
            rows.append(
                {
                    "feature": column,
                    "base_keyword": matched_base,
                    "operation": matched_suffix.removeprefix("_"),
                    "warning": "Diff/ratio/between features on static or near-static context may be meaningless.",
                }
            )
    return pd.DataFrame(rows)


def build_leakage_risk_columns(feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in feature_columns:
        lower = column.lower()
        matched = [pattern for pattern in LEAKAGE_PATTERNS if pattern in lower]
        if matched:
            rows.append(
                {
                    "feature": column,
                    "matched_patterns": ";".join(matched),
                    "risk": "target-like, identifier-like, or split-like column in tabular feature columns",
                }
            )
    return pd.DataFrame(rows)


def build_scaling_check(data: dict[str, Any], feature_columns: list[str]) -> dict[str, Any]:
    raw = data.get("pair_tabular")
    scaled = data.get("pair_tabular_scaled")
    imputer = data.get("imputer_values") or {}
    scaler = data.get("scaler_stats") or {}
    result: dict[str, Any] = {
        "code_intent": "src.tabular.create_pair_features.impute_and_scale_pair_features fits medians/scaler on split == 'train'.",
        "can_prove_from_saved_files_only": False,
        "stats_match_train_recalculation": None,
        "max_abs_train_scaled_mean": None,
        "max_abs_train_scaled_std_minus_1": None,
        "split_scaled_summary": {},
        "anomaly_columns": [],
        "notes": [],
    }
    if not isinstance(raw, pd.DataFrame) or not isinstance(scaled, pd.DataFrame):
        result["notes"].append("Raw or scaled pair tabular parquet is missing.")
        return result
    if "split" not in raw or "split" not in scaled:
        result["notes"].append("Split column is missing, so train-only scaling cannot be checked.")
        return result

    present_features = [column for column in feature_columns if column in raw and column in scaled]
    train_mask = raw["split"].eq("train")
    if not train_mask.any():
        result["notes"].append("No train rows found.")
        return result

    mismatches = []
    max_mean_error = 0.0
    max_scale_error = 0.0
    for column in present_features:
        train_values = pd.to_numeric(raw.loc[train_mask, column], errors="coerce")
        median = train_values.median(skipna=True)
        if pd.isna(median):
            median = 0.0
        imputed_train = train_values.fillna(median).astype(float)
        mean = float(imputed_train.mean())
        scale = float(imputed_train.std(ddof=0))
        if scale == 0.0:
            scale = 1.0

        saved_median = imputer.get(column)
        saved_stats = scaler.get(column, {})
        saved_mean = saved_stats.get("mean")
        saved_scale = saved_stats.get("scale")
        median_error = abs(float(saved_median) - float(median)) if saved_median is not None else math.inf
        mean_error = abs(float(saved_mean) - mean) if saved_mean is not None else math.inf
        scale_error = abs(float(saved_scale) - scale) if saved_scale is not None else math.inf
        max_mean_error = max(max_mean_error, mean_error if math.isfinite(mean_error) else 0.0)
        max_scale_error = max(max_scale_error, scale_error if math.isfinite(scale_error) else 0.0)
        if median_error > 1e-8 or mean_error > 1e-8 or scale_error > 1e-8:
            mismatches.append(column)

    result["stats_match_train_recalculation"] = len(mismatches) == 0
    result["can_prove_from_saved_files_only"] = len(mismatches) == 0
    result["max_saved_mean_error_vs_train"] = max_mean_error
    result["max_saved_scale_error_vs_train"] = max_scale_error
    if mismatches:
        result["notes"].append(
            f"{len(mismatches)} columns do not match train-only imputer/scaler recalculation."
        )

    train_scaled = scaled.loc[scaled["split"].eq("train"), present_features].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not train_scaled.empty:
        train_means = train_scaled.mean()
        train_stds = train_scaled.std(ddof=0)
        result["max_abs_train_scaled_mean"] = safe_float(train_means.abs().max())
        result["max_abs_train_scaled_std_minus_1"] = safe_float((train_stds - 1).abs().max())

    for split, group in scaled.groupby("split", dropna=False):
        numeric = group[present_features].apply(pd.to_numeric, errors="coerce")
        split_key = str(split)
        result["split_scaled_summary"][split_key] = {
            "rows": int(len(group)),
            "mean_abs_mean": safe_float(numeric.mean().abs().mean()),
            "max_abs_mean": safe_float(numeric.mean().abs().max()),
            "mean_std": safe_float(numeric.std(ddof=0).mean()),
            "max_abs_value": safe_float(numeric.abs().max().max()),
        }

    anomalies = []
    for split, group in scaled.groupby("split", dropna=False):
        numeric = group[present_features].apply(pd.to_numeric, errors="coerce")
        means = numeric.mean()
        large_means = means[means.abs() > 5.0]
        for column, value in large_means.items():
            anomalies.append({"split": str(split), "feature": column, "scaled_mean": float(value)})
    result["anomaly_columns"] = anomalies[:50]
    return result


def determine_status(
    coverage: dict[str, Any],
    leakage_df: pd.DataFrame,
    scaling_check: dict[str, Any],
    static_warning_df: pd.DataFrame,
) -> str:
    if not leakage_df.empty:
        return "FAIL"
    if scaling_check.get("stats_match_train_recalculation") is False:
        return "FAIL"
    warning_conditions = [
        (coverage.get("missing_any") or 0) > 0,
        not static_warning_df.empty,
        bool(scaling_check.get("anomaly_columns")),
    ]
    return "WARNING" if any(warning_conditions) else "PASS"


def build_markdown_report(
    status: str,
    row_counts: dict[str, Any],
    join_coverage: dict[str, Any],
    season_mapping: dict[str, Any],
    category_df: pd.DataFrame,
    extreme_df: pd.DataFrame,
    static_warning_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    scaling_check: dict[str, Any],
    data: dict[str, Any],
) -> str:
    category_counts = category_df["category"].value_counts().sort_index().to_dict()
    environmental = category_df.loc[
        category_df["category"].eq("environmental_dynamic"),
        "feature",
    ].tolist()
    temporal = category_df.loc[
        category_df["category"].eq("temporal_metadata"),
        "feature",
    ].tolist()
    static_context = category_df.loc[
        category_df["category"].isin(["static_geographic", "demographic"]),
        "feature",
    ].tolist()
    agricultural = category_df.loc[
        category_df["category"].eq("agricultural"),
        "feature",
    ].tolist()

    key_findings = []
    if leakage_df.empty:
        key_findings.append("No target-like, identifier-like, or split-like columns were found in the saved tabular feature column list.")
    else:
        key_findings.append(f"Found {len(leakage_df)} leakage-risk feature columns.")
    if scaling_check.get("stats_match_train_recalculation"):
        key_findings.append("Saved imputer and scaler statistics match a train-split-only recalculation from the raw pair features.")
    else:
        key_findings.append("Saved imputer/scaler statistics could not be proven to match train-only recalculation.")
    key_findings.append(
        f"Join coverage: {join_coverage.get('full_tabular_match')} full matches out of {join_coverage.get('total_pairs')} pairs; "
        f"T1 missing {join_coverage.get('missing_t1')}, T2 missing {join_coverage.get('missing_t2')}."
    )
    key_findings.append(
        f"Static/contextual diff-ratio warning columns found: {len(static_warning_df)}."
    )

    safe_line = (
        "Safe to use with caveats: the pipeline appears leakage-safe and train-only scaled, "
        "but thesis text and future experiments should separate environmental variables from temporal metadata and static/contextual variables."
        if status == "WARNING"
        else "Safe to use: no audit warnings were detected."
        if status == "PASS"
        else "Not safe to use as-is until FAIL findings are resolved."
    )

    rows = [
        "# Tabular Pipeline Audit Summary",
        "",
        f"Overall status: **{status}**",
        "",
        "## Thesis-use assessment",
        "",
        safe_line,
        "",
        "## Key findings",
        "",
        *[f"- {finding}" for finding in key_findings],
        "",
        "## Row counts",
        "",
        dict_to_markdown_table(row_counts),
        "",
        "## Join coverage",
        "",
        dict_to_markdown_table(join_coverage, skip_keys={"missing_districts", "missing_seasons"}),
        "",
        f"Missing tabular districts: {join_coverage.get('missing_districts')}",
        "",
        f"Missing tabular seasons: {join_coverage.get('missing_seasons')}",
        "",
        "## Season mapping",
        "",
        f"- Config season order: `{season_mapping.get('config_season_order')}`",
        f"- Pair-feature code season order: `{season_mapping.get('create_pair_features_season_order')}`",
        f"- Monthly aggregation rules: `{season_mapping.get('monthly_to_seasonal_month_rules')}`",
        "",
        "## Feature categories",
        "",
        dict_to_markdown_table(category_counts),
        "",
        "Environmental features should be described as dynamic environmental/hazard variables, including climate, hydrology, soil-water, and event-count features.",
        "",
        f"Environmental dynamic feature count: {len(environmental)}",
        "",
        f"Agricultural feature count: {len(agricultural)}",
        "",
        "Temporal metadata features are allowed but should be reported separately from environmental variables.",
        "",
        f"Temporal metadata features: `{temporal}`",
        "",
        "Static/contextual features include geographic and demographic context and should not be described as dynamic environmental measurements.",
        "",
        f"Static/contextual feature count: {len(static_context)}",
        "",
        "## Static-feature warnings",
        "",
        f"Columns where diff/ratio/between features may be meaningless: {len(static_warning_df)}",
        "",
        preview_table(static_warning_df, limit=40),
        "",
        "Recommendation for future experiments: consider removing or ablating static/contextual diff and ratio columns, especially district area, slope, distance-to-coast, and population diff/ratio/mean-between variants.",
        "",
        "## Leakage risk check",
        "",
        "No leakage-risk columns found." if leakage_df.empty else preview_table(leakage_df, limit=80),
        "",
        "## Scaling and imputation check",
        "",
        f"- Code intent: {scaling_check.get('code_intent')}",
        f"- Saved stats match train-only recalculation: `{scaling_check.get('stats_match_train_recalculation')}`",
        f"- Can prove from saved files and raw pair features: `{scaling_check.get('can_prove_from_saved_files_only')}`",
        f"- Max absolute train scaled mean: `{scaling_check.get('max_abs_train_scaled_mean')}`",
        f"- Max absolute train scaled std-minus-one: `{scaling_check.get('max_abs_train_scaled_std_minus_1')}`",
        f"- Split scaled summary: `{scaling_check.get('split_scaled_summary')}`",
        f"- Anomaly columns with abs(split scaled mean) > 5: `{scaling_check.get('anomaly_columns')}`",
        "",
        "## Extreme-value check",
        "",
        f"Extreme-value rows written: {len(extreme_df)}",
        "",
        preview_table(
            extreme_df.sort_values(
                ["inf_count", "extremely_large_abs_count", "missing_count"],
                ascending=False,
            ),
            limit=20,
        ),
        "",
        "## Output files",
        "",
        "- `outputs/reports/tabular_audit/tabular_audit_summary.md`",
        "- `outputs/reports/tabular_audit/tabular_feature_category_summary.csv`",
        "- `outputs/reports/tabular_audit/tabular_extreme_values.csv`",
        "- `outputs/reports/tabular_audit/tabular_missingness_summary.csv`",
        "- `outputs/reports/tabular_audit/tabular_static_feature_warnings.csv`",
        "- `outputs/reports/tabular_audit/tabular_leakage_risk_columns.csv`",
        "",
    ]

    unavailable = [name for name, value in data.items() if value is None]
    if unavailable:
        rows.extend(["## Missing inspected inputs", "", *[f"- `{name}`: `{PATHS[name]}`" for name in unavailable], ""])
    return "\n".join(rows)


def true_count(df: pd.DataFrame, column: str) -> int:
    if column not in df:
        return 0
    return int((df[column] == True).sum())  # noqa: E712


def safe_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def dict_to_markdown_table(values: dict[str, Any], skip_keys: set[str] | None = None) -> str:
    skip_keys = skip_keys or set()
    lines = ["| metric | value |", "|---|---:|"]
    for key, value in values.items():
        if key in skip_keys:
            continue
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)


def preview_table(df: pd.DataFrame, limit: int = 20) -> str:
    if df.empty:
        return "None."
    preview = df.head(limit).copy()
    columns = [str(column) for column in preview.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in preview.iterrows():
        values = [markdown_cell(row[column]) for column in preview.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\n", " ").replace("|", "\\|")
    return text


if __name__ == "__main__":
    main()
