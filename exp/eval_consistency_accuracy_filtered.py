from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Basic helpers
# ============================================================


def normalize_transition_key(path_like: Any, remove_final_suffix: bool = True) -> str:
    key = Path(str(path_like).strip()).name
    if remove_final_suffix:
        lowered = key.lower()
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".txt"):
            if lowered.endswith(ext):
                key = key[: -len(ext)]
                break
    return key



def extract_prefix_and_frame(path_like: Any) -> Tuple[str, int]:
    key = normalize_transition_key(path_like)
    if "_frame" not in key:
        return key, -1
    prefix, tail = key.split("_frame", 1)
    digits: List[str] = []
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    frame_idx = int("".join(digits)) if digits else -1
    return prefix, frame_idx



def threshold_to_suffix(x: float) -> str:
    s = f"{float(x):.3f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")



def safe_method_slug(text: str) -> str:
    out: List[str] = []
    for ch in str(text):
        if ch.isalnum() or ch == "_":
            out.append(ch)
        elif ch == ".":
            out.append("p")
        elif ch == "-":
            out.append("m")
        else:
            out.append("_")
    return "".join(out)


# ============================================================
# Filter helpers
# ============================================================


def load_filter_tokens(filter_arg: Optional[str]) -> Optional[List[str]]:
    if filter_arg is None:
        return None

    p = Path(str(filter_arg))
    if p.exists() and p.is_file():
        txt = p.read_text(encoding="utf-8")
    else:
        txt = str(filter_arg)

    txt = txt.replace("\n", ",")
    tokens = [x.strip() for x in txt.split(",") if x.strip()]
    return tokens if tokens else None



def prefix_pass_filter(prefix: str, tokens: Optional[List[str]], mode: Optional[str]) -> bool:
    if not tokens or not mode:
        return True

    if mode == "exact":
        # blacklist: exact prefixes are excluded entirely
        return prefix not in tokens

    if mode == "contains":
        # whitelist: keep only prefixes containing any token
        return any(t in prefix for t in tokens)

    return True


# ============================================================
# CLI parsers
# ============================================================


def parse_alpha_beta_pairs(text: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    if not text:
        return pairs
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            a_str, b_str = item.split(":", 1)
        elif "/" in item:
            a_str, b_str = item.split("/", 1)
        else:
            raise ValueError(f"Invalid alpha:beta pair: {item}")
        pairs.append((float(a_str), float(b_str)))
    return pairs


# ============================================================
# Stage A inputs: prediction consistency + ssim
# ============================================================


def get_pred_frame_consistency(entry: Any) -> Optional[float]:
    if not isinstance(entry, dict):
        return None
    if entry.get("frame_consistency") is not None:
        return float(entry["frame_consistency"])
    if entry.get("score") is not None:
        return float(entry["score"])
    return None



def load_prediction_transition_table(frame_path: str, ssim_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    with open(frame_path, "r", encoding="utf-8") as f:
        object_consistency_dict = json.load(f)

    ssim_df = pd.read_csv(ssim_path)
    missing = {"file_name", "ssim_value"} - set(ssim_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in SSIM CSV: {sorted(missing)}")

    object_lookup: Dict[str, Dict[str, Any]] = {}
    debug: Dict[str, Any] = {
        "raw_object_entries": len(object_consistency_dict),
        "raw_ssim_rows": int(len(ssim_df)),
        "object_missing_consistency": [],
        "object_duplicate_keys": [],
        "ssim_duplicate_keys": [],
        "ssim_non_numeric_or_nan": [],
    }

    seen_obj = set()
    for raw_key, entry in object_consistency_dict.items():
        clean_key = normalize_transition_key(raw_key)
        if clean_key in seen_obj:
            debug["object_duplicate_keys"].append(clean_key)
            continue
        seen_obj.add(clean_key)

        consistency = get_pred_frame_consistency(entry)
        if consistency is None or pd.isna(consistency):
            debug["object_missing_consistency"].append(clean_key)
            continue

        prefix, frame_idx = extract_prefix_and_frame(clean_key)
        transition_from = normalize_transition_key(entry.get("transition_from", raw_key))
        transition_to = None
        next_frame_idx = -1
        if entry.get("transition_to") is not None:
            transition_to = normalize_transition_key(entry["transition_to"])
            _, next_frame_idx = extract_prefix_and_frame(transition_to)

        object_lookup[clean_key] = {
            "transition_key": clean_key,
            "prefix": prefix,
            "frame_idx": int(frame_idx),
            "next_frame_idx": int(next_frame_idx),
            "transition_from": transition_from,
            "transition_to": transition_to,
            "pred_frame_consistency": float(consistency),
            "pred_num_matches": int(entry.get("num_matches", 0)) if entry.get("num_matches") is not None else None,
            "pred_mean_match_quality": float(entry.get("mean_match_quality")) if entry.get("mean_match_quality") is not None else np.nan,
        }

    ssim_lookup: Dict[str, float] = {}
    seen_ssim = set()
    for _, row in ssim_df.iterrows():
        clean_key = normalize_transition_key(row["file_name"])
        if clean_key in seen_ssim:
            debug["ssim_duplicate_keys"].append(clean_key)
            continue
        seen_ssim.add(clean_key)
        try:
            val = float(row["ssim_value"])
            if pd.isna(val):
                raise ValueError("NaN")
        except Exception:
            debug["ssim_non_numeric_or_nan"].append(clean_key)
            continue
        ssim_lookup[clean_key] = val

    common_keys = sorted(set(object_lookup.keys()) & set(ssim_lookup.keys()))
    dataset: List[Dict[str, Any]] = []
    for key in common_keys:
        row = dict(object_lookup[key])
        row["ssim"] = float(ssim_lookup[key])
        dataset.append(row)

    df = pd.DataFrame(dataset)
    if not df.empty:
        df = df.sort_values(["prefix", "frame_idx", "transition_key"]).reset_index(drop=True)

    debug["joined_samples"] = int(len(df))
    debug["num_prefixes"] = int(df["prefix"].nunique()) if not df.empty else 0
    debug["object_only_keys_sample"] = sorted(list(set(object_lookup.keys()) - set(ssim_lookup.keys())))[:20]
    debug["ssim_only_keys_sample"] = sorted(list(set(ssim_lookup.keys()) - set(object_lookup.keys())))[:20]
    return df, debug


# ============================================================
# Stage B inputs: grouped GT
# ============================================================


def load_grouped_gt_transition_tables(
    grouped_gt_root: str,
    gt_consistency_threshold: Optional[float] = None,
    exclude_terminal: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    root = Path(grouped_gt_root)
    if not root.exists():
        raise FileNotFoundError(f"Grouped GT root not found: {grouped_gt_root}")

    gt_by_prefix: Dict[str, pd.DataFrame] = {}
    debug: Dict[str, Any] = {
        "grouped_gt_root": str(root),
        "prefix_dirs": [],
        "skipped_dirs": [],
    }

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        csv_path = child / "gt_transition_reference.csv"
        if not csv_path.exists():
            debug["skipped_dirs"].append(child.name)
            continue

        df = pd.read_csv(csv_path)
        if "transition_key" not in df.columns:
            if "transition_from" in df.columns:
                df["transition_key"] = df["transition_from"].map(normalize_transition_key)
            else:
                debug["skipped_dirs"].append(child.name)
                continue
        else:
            df["transition_key"] = df["transition_key"].map(normalize_transition_key)

        if "prefix" not in df.columns:
            df["prefix"] = df["transition_key"].map(lambda x: extract_prefix_and_frame(x)[0])

        if "is_terminal" not in df.columns:
            df["is_terminal"] = False
        df["is_terminal"] = df["is_terminal"].fillna(False).astype(bool)

        if "valid_transition" not in df.columns:
            df["valid_transition"] = True
        df["valid_transition"] = df["valid_transition"].fillna(True).astype(bool)

        if "gt_frame_consistency" not in df.columns:
            raise ValueError(f"Missing gt_frame_consistency in {csv_path}")

        if "gt_is_consistent_binary" not in df.columns:
            if gt_consistency_threshold is None:
                raise ValueError(
                    f"Missing gt_is_consistent_binary in {csv_path}; use --main-binary-threshold or provide gt binary column"
                )
            df["gt_is_consistent_binary"] = (
                pd.to_numeric(df["gt_frame_consistency"], errors="coerce") >= gt_consistency_threshold
            ).astype(int)
        else:
            df["gt_is_consistent_binary"] = pd.to_numeric(df["gt_is_consistent_binary"], errors="coerce").fillna(0).astype(int)

        if exclude_terminal:
            df = df[~df["is_terminal"]].copy()
        df = df[df["valid_transition"]].copy()

        if df.empty:
            debug["skipped_dirs"].append(child.name)
            continue

        prefix_values = [p for p in df["prefix"].dropna().astype(str).unique().tolist() if p]
        if len(prefix_values) == 1:
            prefix = prefix_values[0]
        else:
            name = child.name
            prefix = name[3:] if len(name) > 3 and name[:2].isdigit() and name[2] == "_" else name
            df["prefix"] = prefix

        gt_by_prefix[prefix] = df.sort_values(["transition_key"]).reset_index(drop=True)
        debug["prefix_dirs"].append({"prefix": prefix, "dir": child.name, "rows": int(len(df))})

    debug["num_prefixes"] = len(gt_by_prefix)
    return gt_by_prefix, debug


# ============================================================
# Stage A grouped calibration results (per-prefix only)
# ============================================================


def load_grouped_calibration_results(calibration_root: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    root = Path(calibration_root)
    if not root.exists():
        raise FileNotFoundError(f"Calibration root not found: {calibration_root}")

    calib_by_prefix: Dict[str, Dict[str, Any]] = {}
    loaded_rows: List[Dict[str, Any]] = []

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        summary_path = child / "calibration_summary.json"
        if not summary_path.exists():
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        prefix = summary.get("prefix")
        if not prefix:
            name = child.name
            prefix = name[3:] if len(name) > 3 and name[:2].isdigit() and name[2] == "_" else name

        alpha = float(summary["alpha"])
        beta = float(summary["beta"])
        calib_by_prefix[prefix] = {
            "prefix": prefix,
            "alpha": alpha,
            "beta": beta,
            "source": str(summary_path),
            "mode": summary.get("mode"),
        }
        loaded_rows.append({
            "prefix": prefix,
            "alpha": alpha,
            "beta": beta,
            "source": str(summary_path),
            "mode": summary.get("mode"),
        })

    debug = {
        "calibration_root": str(root),
        "load_mode": "per_prefix_calibration_summary_only",
        "num_prefixes": len(calib_by_prefix),
        "prefixes": sorted(calib_by_prefix.keys()),
        "loaded_rows": loaded_rows,
    }
    return calib_by_prefix, debug


# ============================================================
# SSIM features
# ============================================================


def compute_ssim_features_for_series(ssim_series: pd.Series) -> Dict[str, Any]:
    x = pd.to_numeric(ssim_series, errors="coerce").dropna().astype(float)
    if len(x) == 0:
        return {
            "ssim_num_samples": 0,
            "ssim_mean": np.nan,
            "ssim_var": np.nan,
            "ssim_std": np.nan,
            "ssim_min": np.nan,
            "ssim_max": np.nan,
            "ssim_range": np.nan,
            "ssim_median": np.nan,
            "ssim_q25": np.nan,
            "ssim_q75": np.nan,
            "ssim_iqr": np.nan,
            "ssim_cv": np.nan,
            "ssim_volatility_mean_abs_diff": np.nan,
            "ssim_volatility_std_diff": np.nan,
            "ssim_volatility_max_abs_diff": np.nan,
        }

    diffs = np.diff(x.to_numpy(dtype=float)) if len(x) >= 2 else np.array([], dtype=float)
    abs_diffs = np.abs(diffs) if len(diffs) else np.array([], dtype=float)
    mean_val = float(x.mean())
    std_val = float(x.std(ddof=0))
    q25 = float(x.quantile(0.25))
    q75 = float(x.quantile(0.75))

    return {
        "ssim_num_samples": int(len(x)),
        "ssim_mean": mean_val,
        "ssim_var": float(x.var(ddof=0)),
        "ssim_std": std_val,
        "ssim_min": float(x.min()),
        "ssim_max": float(x.max()),
        "ssim_range": float(x.max() - x.min()),
        "ssim_median": float(x.median()),
        "ssim_q25": q25,
        "ssim_q75": q75,
        "ssim_iqr": float(q75 - q25),
        "ssim_cv": float(std_val / mean_val) if mean_val != 0 else np.nan,
        "ssim_volatility_mean_abs_diff": float(abs_diffs.mean()) if len(abs_diffs) else np.nan,
        "ssim_volatility_std_diff": float(diffs.std(ddof=0)) if len(diffs) else np.nan,
        "ssim_volatility_max_abs_diff": float(abs_diffs.max()) if len(abs_diffs) else np.nan,
    }



def compute_prefix_ssim_features(pred_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if pred_df.empty:
        return out
    for prefix, sub in pred_df.groupby("prefix", dropna=False):
        s = sub.sort_values(["frame_idx", "transition_key"])["ssim"]
        out[str(prefix)] = compute_ssim_features_for_series(s)
    return out


# ============================================================
# Methods and metrics
# ============================================================


def build_fixed_alpha_beta_method_name(alpha: float, beta: float) -> Tuple[str, str]:
    slug = f"fixed_ab_a{threshold_to_suffix(alpha)}_b{threshold_to_suffix(beta)}"
    display = f"Fixed Alpha/Beta (a={alpha:.2f}, b={beta:.2f})"
    return slug, display



def build_method_specs(eval_mode: str, fixed_alpha_beta_pairs: Iterable[Tuple[float, float]]) -> List[Dict[str, Any]]:
    methods: List[Dict[str, Any]] = []
    if eval_mode == "all":
        for alpha, beta in fixed_alpha_beta_pairs:
            method_slug, display_name = build_fixed_alpha_beta_method_name(alpha, beta)
            methods.append({
                "method_type": "fixed_alpha_beta",
                "display_name": display_name,
                "method_slug": safe_method_slug(method_slug),
                "alpha": float(alpha),
                "beta": float(beta),
            })

    methods.append({
        "method_type": "calibrated_alpha_beta",
        "display_name": "Calibrated Alpha/Beta",
        "method_slug": "calibrated_alpha_beta",
    })
    return methods



def apply_method_predictions(
    df: pd.DataFrame,
    method: Dict[str, Any],
    calibrated_alpha: Optional[float] = None,
    calibrated_beta: Optional[float] = None,
) -> pd.DataFrame:
    out = df.copy()
    method_type = method["method_type"]

    if method_type == "fixed_alpha_beta":
        alpha = float(method["alpha"])
        beta = float(method["beta"])
        out["decision_boundary"] = alpha * out["ssim"] + beta
        out["pred_is_consistent_binary"] = (out["pred_frame_consistency"] >= out["decision_boundary"]).astype(int)
    elif method_type == "calibrated_alpha_beta":
        if calibrated_alpha is None or calibrated_beta is None:
            raise ValueError("Missing calibrated alpha/beta")
        out["decision_boundary"] = float(calibrated_alpha) * out["ssim"] + float(calibrated_beta)
        out["pred_is_consistent_binary"] = (out["pred_frame_consistency"] >= out["decision_boundary"]).astype(int)
    else:
        raise ValueError(f"Unsupported method_type: {method_type}")

    out["correct"] = (out["pred_is_consistent_binary"] == out["gt_is_consistent_binary"]).astype(int)
    return out



def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n = int(len(y_true))

    accuracy = (tp + tn) / n if n else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    balanced_accuracy = 0.5 * (recall + specificity) if pd.notna(recall) and pd.notna(specificity) else np.nan

    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy) if pd.notna(accuracy) else None,
        "precision": float(precision) if pd.notna(precision) else None,
        "recall": float(recall) if pd.notna(recall) else None,
        "f1": float(f1) if pd.notna(f1) else None,
        "specificity": float(specificity) if pd.notna(specificity) else None,
        "balanced_accuracy": float(balanced_accuracy) if pd.notna(balanced_accuracy) else None,
    }



def compute_continuous_alignment_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "continuous_mae": None,
            "continuous_rmse": None,
            "continuous_bias": None,
            "continuous_corr": None,
        }

    y_true = pd.to_numeric(df["gt_frame_consistency"], errors="coerce")
    y_pred = pd.to_numeric(df["pred_frame_consistency"], errors="coerce")
    valid = ~(y_true.isna() | y_pred.isna())
    y_true = y_true[valid].to_numpy(dtype=float)
    y_pred = y_pred[valid].to_numpy(dtype=float)
    if len(y_true) == 0:
        return {
            "continuous_mae": None,
            "continuous_rmse": None,
            "continuous_bias": None,
            "continuous_corr": None,
        }

    err = y_pred - y_true
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) >= 2 else np.nan
    return {
        "continuous_mae": float(np.mean(np.abs(err))),
        "continuous_rmse": float(np.sqrt(np.mean(err ** 2))),
        "continuous_bias": float(np.mean(err)),
        "continuous_corr": float(corr) if pd.notna(corr) else None,
    }


# ============================================================
# Fairness constraints for fixed alpha/beta baselines
# ============================================================


def boundary_is_valid_for_prefix(alpha: float, beta: float, ssim_min: float, ssim_max: float) -> bool:
    y_min = alpha * ssim_min + beta
    y_max = alpha * ssim_max + beta
    return (0.0 <= y_min <= 1.0) and (0.0 <= y_max <= 1.0)



def pred_positive_rate_from_df(evaluated_df: pd.DataFrame) -> float:
    return float(pd.to_numeric(evaluated_df["pred_is_consistent_binary"], errors="coerce").fillna(0).astype(int).mean())


# ============================================================
# Join prediction + GT
# ============================================================


def join_prefix_prediction_with_gt(pred_df: pd.DataFrame, gt_df: pd.DataFrame, prefix: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    pred_p = pred_df[pred_df["prefix"] == prefix].copy()
    gt_p = gt_df.copy()

    rename_gt = {}
    if "frame_idx" in gt_p.columns:
        rename_gt["frame_idx"] = "gt_frame_idx"
    if "next_frame_idx" in gt_p.columns:
        rename_gt["next_frame_idx"] = "gt_next_frame_idx"
    if rename_gt:
        gt_p = gt_p.rename(columns=rename_gt)

    gt_keep_cols = [
        c for c in [
            "transition_key",
            "prefix",
            "gt_frame_consistency",
            "gt_is_consistent_binary",
            "gt_num_t",
            "gt_num_t1",
            "gt_num_matches",
            "gt_mean_match_quality",
            "gt_frame_idx",
            "gt_next_frame_idx",
        ]
        if c in gt_p.columns
    ]

    pred_keys = set(pred_p["transition_key"].tolist())
    gt_keys = set(gt_p["transition_key"].tolist())
    common_keys = pred_keys & gt_keys

    joined = pred_p.merge(
        gt_p[gt_keep_cols],
        on=[c for c in ["transition_key", "prefix"] if c in pred_p.columns and c in gt_p.columns],
        how="inner",
        suffixes=("", "_gt"),
    )

    sort_cols: List[str] = []
    if "frame_idx" in joined.columns:
        sort_cols.append("frame_idx")
    elif "gt_frame_idx" in joined.columns:
        sort_cols.append("gt_frame_idx")
    sort_cols.append("transition_key")
    joined = joined.sort_values(sort_cols).reset_index(drop=True)

    debug = {
        "prefix": prefix,
        "pred_rows": int(len(pred_p)),
        "gt_rows": int(len(gt_p)),
        "common_rows": int(len(joined)),
        "pred_only_keys": sorted(list(pred_keys - gt_keys))[:20],
        "gt_only_keys": sorted(list(gt_keys - pred_keys))[:20],
        "sample_common_keys": sorted(list(common_keys))[:10],
        "joined_columns": joined.columns.tolist(),
    }
    return joined, debug


# ============================================================
# Main evaluation
# ============================================================


def evaluate_grouped_consistency(
    frame_path: str,
    ssim_path: str,
    grouped_gt_root: str,
    calibration_root: str,
    save_path: str,
    eval_mode: str,
    fixed_alpha_beta_pairs: List[Tuple[float, float]],
    main_binary_threshold: float,
    exclude_empty_empty_from_main_eval: bool,
    filter_arg: Optional[str],
    filter_mode: Optional[str],
    enable_fixed_method_fair_filter: bool,
    positive_rate_tolerance: float,
) -> Dict[str, Any]:
    output_root = Path(save_path)
    output_root.mkdir(parents=True, exist_ok=True)

    pred_df, pred_debug = load_prediction_transition_table(frame_path, ssim_path)
    gt_by_prefix, gt_debug = load_grouped_gt_transition_tables(
        grouped_gt_root,
        gt_consistency_threshold=main_binary_threshold,
        exclude_terminal=True,
    )
    calib_by_prefix, calib_debug = load_grouped_calibration_results(calibration_root)

    if pred_df.empty:
        raise ValueError("No joined prediction transitions available from frame_path + ssim_path")
    if not gt_by_prefix:
        raise ValueError("No grouped GT transition tables available")

    ssim_features_by_prefix = compute_prefix_ssim_features(pred_df)

    pred_prefixes = sorted(pred_df["prefix"].unique().tolist())
    gt_prefixes = sorted(gt_by_prefix.keys())
    calib_prefixes = sorted(calib_by_prefix.keys())

    common_prefixes_all = sorted(set(pred_prefixes) & set(gt_prefixes))
    filter_tokens = load_filter_tokens(filter_arg)
    common_prefixes = [p for p in common_prefixes_all if prefix_pass_filter(p, filter_tokens, filter_mode)]
    excluded_by_filter = [p for p in common_prefixes_all if p not in common_prefixes]

    if not common_prefixes:
        raise ValueError("No prefixes remained after applying filter / intersection")

    methods = build_method_specs(eval_mode, fixed_alpha_beta_pairs)
    method_configs_path = output_root / "method_configs.json"
    with open(method_configs_path, "w", encoding="utf-8") as f:
        json.dump(methods, f, indent=2, ensure_ascii=False)

    per_prefix_metrics_rows: List[Dict[str, Any]] = []
    prefix_ssim_feature_rows: List[Dict[str, Any]] = []
    all_transition_tables: Dict[str, List[pd.DataFrame]] = {m["method_slug"]: [] for m in methods}
    prefix_summaries: List[Dict[str, Any]] = []

    for idx, prefix in enumerate(common_prefixes):
        prefix_dir = output_root / f"{idx:02d}_{prefix}"
        prefix_dir.mkdir(parents=True, exist_ok=True)

        joined_df, join_debug = join_prefix_prediction_with_gt(pred_df, gt_by_prefix[prefix], prefix)
        if joined_df.empty:
            prefix_summaries.append({
                "prefix": prefix,
                "output_dir": str(prefix_dir),
                "skipped": True,
                "reason": "no_common_transition_keys",
                "join_debug": join_debug,
            })
            continue

        base_df = joined_df.copy()
        if exclude_empty_empty_from_main_eval and "gt_num_t" in base_df.columns and "gt_num_t1" in base_df.columns:
            mask_empty_empty = (pd.to_numeric(base_df["gt_num_t"], errors="coerce").fillna(-1) == 0) & (
                pd.to_numeric(base_df["gt_num_t1"], errors="coerce").fillna(-1) == 0
            )
            base_df = base_df.loc[~mask_empty_empty].reset_index(drop=True)

        if base_df.empty:
            prefix_summaries.append({
                "prefix": prefix,
                "output_dir": str(prefix_dir),
                "skipped": True,
                "reason": "empty_after_main_eval_filter",
                "join_debug": join_debug,
            })
            continue

        base_df["gt_is_consistent_binary"] = base_df["gt_is_consistent_binary"].astype(int)
        transition_table = base_df.copy()

        prefix_ssim_features = ssim_features_by_prefix.get(prefix, compute_ssim_features_for_series(pd.Series(dtype="float64")))
        prefix_ssim_feature_rows.append({"prefix": prefix, **prefix_ssim_features})
        for feat_name, feat_value in prefix_ssim_features.items():
            transition_table[feat_name] = feat_value

        method_metrics_rows: List[Dict[str, Any]] = []
        available_methods: List[str] = []
        skipped_methods: List[Dict[str, Any]] = []

        calib_alpha = None
        calib_beta = None
        calibration_source = None
        if prefix in calib_by_prefix:
            calib_alpha = float(calib_by_prefix[prefix]["alpha"])
            calib_beta = float(calib_by_prefix[prefix]["beta"])
            calibration_source = calib_by_prefix[prefix].get("source")

        continuous_metrics = compute_continuous_alignment_metrics(base_df)
        prefix_ssim_min = float(pd.to_numeric(base_df["ssim"], errors="coerce").min())
        prefix_ssim_max = float(pd.to_numeric(base_df["ssim"], errors="coerce").max())

        calibrated_ref_positive_rate: Optional[float] = None
        if calib_alpha is not None and calib_beta is not None:
            calibrated_ref_df = apply_method_predictions(
                base_df,
                {"method_type": "calibrated_alpha_beta", "method_slug": "calibrated_alpha_beta", "display_name": "Calibrated Alpha/Beta"},
                calibrated_alpha=calib_alpha,
                calibrated_beta=calib_beta,
            )
            calibrated_ref_positive_rate = pred_positive_rate_from_df(calibrated_ref_df)

        for method in methods:
            method_type = method["method_type"]
            method_slug = method["method_slug"]
            method_display = method["display_name"]

            if method_type == "calibrated_alpha_beta":
                if calib_alpha is None or calib_beta is None:
                    skipped_methods.append({
                        "method_slug": method_slug,
                        "display_name": method_display,
                        "reason": "missing_calibration_for_prefix",
                    })
                    continue
                evaluated_df = apply_method_predictions(
                    base_df,
                    method,
                    calibrated_alpha=calib_alpha,
                    calibrated_beta=calib_beta,
                )
                row_alpha = float(calib_alpha)
                row_beta = float(calib_beta)
                current_positive_rate = pred_positive_rate_from_df(evaluated_df)
            else:
                evaluated_df = apply_method_predictions(base_df, method)
                row_alpha = float(method["alpha"])
                row_beta = float(method["beta"])
                current_positive_rate = pred_positive_rate_from_df(evaluated_df)

                if enable_fixed_method_fair_filter:
                    if not boundary_is_valid_for_prefix(row_alpha, row_beta, prefix_ssim_min, prefix_ssim_max):
                        skipped_methods.append({
                            "method_slug": method_slug,
                            "display_name": method_display,
                            "reason": "invalid_boundary_range",
                            "alpha": row_alpha,
                            "beta": row_beta,
                            "prefix_ssim_min": prefix_ssim_min,
                            "prefix_ssim_max": prefix_ssim_max,
                        })
                        continue

                    if calibrated_ref_positive_rate is not None:
                        if abs(current_positive_rate - calibrated_ref_positive_rate) > positive_rate_tolerance:
                            skipped_methods.append({
                                "method_slug": method_slug,
                                "display_name": method_display,
                                "reason": "pred_positive_rate_too_far_from_calibrated",
                                "alpha": row_alpha,
                                "beta": row_beta,
                                "pred_positive_rate": current_positive_rate,
                                "calibrated_pred_positive_rate": calibrated_ref_positive_rate,
                                "tolerance": positive_rate_tolerance,
                            })
                            continue

            metrics = compute_binary_metrics(
                evaluated_df["gt_is_consistent_binary"].to_numpy(),
                evaluated_df["pred_is_consistent_binary"].to_numpy(),
            )

            metrics_row = {
                "prefix": prefix,
                "method_slug": method_slug,
                "method_display_name": method_display,
                "method_type": method_type,
                "alpha": row_alpha,
                "beta": row_beta,
                "loaded_calibration_alpha": float(calib_alpha) if calib_alpha is not None else None,
                "loaded_calibration_beta": float(calib_beta) if calib_beta is not None else None,
                "calibration_source": calibration_source,
                "pred_positive_rate": current_positive_rate,
                "calibrated_pred_positive_rate": calibrated_ref_positive_rate,
                "positive_rate_gap_vs_calibrated": (abs(current_positive_rate - calibrated_ref_positive_rate)
                                                     if calibrated_ref_positive_rate is not None else None),
                "main_binary_threshold": float(main_binary_threshold),
                **metrics,
                **continuous_metrics,
                **prefix_ssim_features,
            }

            method_metrics_rows.append(metrics_row)
            per_prefix_metrics_rows.append(metrics_row)
            all_transition_tables[method_slug].append(evaluated_df.assign(prefix=prefix))
            available_methods.append(method_slug)

            transition_table[f"boundary__{method_slug}"] = evaluated_df["decision_boundary"].to_numpy()
            transition_table[f"pred__{method_slug}"] = evaluated_df["pred_is_consistent_binary"].to_numpy()
            transition_table[f"correct__{method_slug}"] = evaluated_df["correct"].to_numpy()

        transition_table_path = prefix_dir / "evaluation_transition_table.csv"
        transition_table.to_csv(transition_table_path, index=False)

        method_metrics_df = pd.DataFrame(method_metrics_rows)
        metrics_csv_path = prefix_dir / "evaluation_metrics.csv"
        method_metrics_df.to_csv(metrics_csv_path, index=False)

        prefix_summary = {
            "prefix": prefix,
            "output_dir": str(prefix_dir),
            "transition_table_csv": str(transition_table_path),
            "metrics_csv": str(metrics_csv_path),
            "join_debug": join_debug,
            "available_methods": available_methods,
            "skipped_methods": skipped_methods,
            "calibration": calib_by_prefix.get(prefix),
            "num_rows": int(len(base_df)),
            "ssim_features": prefix_ssim_features,
            "fair_filter": {
                "enabled": bool(enable_fixed_method_fair_filter),
                "positive_rate_tolerance": float(positive_rate_tolerance),
                "calibrated_pred_positive_rate": calibrated_ref_positive_rate,
                "prefix_ssim_min": prefix_ssim_min,
                "prefix_ssim_max": prefix_ssim_max,
            },
        }
        prefix_summary_path = prefix_dir / "evaluation_summary.json"
        with open(prefix_summary_path, "w", encoding="utf-8") as f:
            json.dump(prefix_summary, f, indent=2, ensure_ascii=False)
        prefix_summary["summary_json"] = str(prefix_summary_path)
        prefix_summaries.append(prefix_summary)

    per_prefix_metrics_df = pd.DataFrame(per_prefix_metrics_rows)
    per_prefix_metrics_path = output_root / "per_prefix_metrics.csv"
    per_prefix_metrics_df.to_csv(per_prefix_metrics_path, index=False)

    prefix_ssim_features_path = output_root / "prefix_ssim_features.csv"
    prefix_ssim_features_df = pd.DataFrame(prefix_ssim_feature_rows)
    if not prefix_ssim_features_df.empty:
        prefix_ssim_features_df = prefix_ssim_features_df.sort_values(["prefix"]).reset_index(drop=True)
    prefix_ssim_features_df.to_csv(prefix_ssim_features_path, index=False)

    global_rows: List[Dict[str, Any]] = []
    for method in methods:
        method_slug = method["method_slug"]
        tables = all_transition_tables.get(method_slug, [])
        if not tables:
            continue
        concat_df = pd.concat(tables, axis=0, ignore_index=True)
        metrics = compute_binary_metrics(
            concat_df["gt_is_consistent_binary"].to_numpy(),
            concat_df["pred_is_consistent_binary"].to_numpy(),
        )
        global_rows.append({
            "method_slug": method_slug,
            "method_display_name": method["display_name"],
            "method_type": method["method_type"],
            "alpha": method.get("alpha"),
            "beta": method.get("beta"),
            "pred_positive_rate": pred_positive_rate_from_df(concat_df),
            **metrics,
            "num_prefixes_evaluated": int(concat_df["prefix"].nunique()) if "prefix" in concat_df.columns else len(tables),
        })

    global_metrics_df = pd.DataFrame(global_rows)
    global_metrics_path = output_root / "global_metrics.csv"
    global_metrics_df.to_csv(global_metrics_path, index=False)

    summary = {
        "frame_path": str(frame_path),
        "ssim_path": str(ssim_path),
        "grouped_gt_root": str(grouped_gt_root),
        "calibration_root": str(calibration_root),
        "save_path": str(output_root),
        "eval_mode": eval_mode,
        "fixed_alpha_beta_pairs": fixed_alpha_beta_pairs,
        "main_binary_threshold": float(main_binary_threshold),
        "exclude_empty_empty_from_main_eval": bool(exclude_empty_empty_from_main_eval),
        "fair_filter": {
            "enabled": bool(enable_fixed_method_fair_filter),
            "positive_rate_tolerance": float(positive_rate_tolerance),
        },
        "filter": {
            "tokens": filter_tokens,
            "mode": filter_mode,
            "common_prefixes_before_filter": common_prefixes_all,
            "prefixes_excluded_by_filter": excluded_by_filter,
            "prefixes_used_after_filter": common_prefixes,
        },
        "pred_debug": pred_debug,
        "gt_debug": gt_debug,
        "calibration_debug": calib_debug,
        "pred_prefixes": pred_prefixes,
        "gt_prefixes": gt_prefixes,
        "calibration_prefixes": calib_prefixes,
        "common_prefixes_before_filter": common_prefixes_all,
        "common_prefixes": common_prefixes,
        "num_processed_prefixes": sum(1 for item in prefix_summaries if not item.get("skipped", False)),
        "num_skipped_prefixes": sum(1 for item in prefix_summaries if item.get("skipped", False)),
        "per_prefix_metrics_csv": str(per_prefix_metrics_path),
        "global_metrics_csv": str(global_metrics_path),
        "method_configs_json": str(method_configs_path),
        "prefix_ssim_features_csv": str(prefix_ssim_features_path),
        "ssim_features_by_prefix": {k: v for k, v in ssim_features_by_prefix.items() if k in common_prefixes},
        "results": prefix_summaries,
    }
    index_path = output_root / "index_summary.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# ============================================================
# CLI
# ============================================================


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate calibrated frame-level consistency against grouped GT references. Only per-prefix calibration_summary.json files are used."
    )
    parser.add_argument("--frame-path", required=True, help="Prediction object consistency JSON")
    parser.add_argument("--ssim-path", required=True, help="SSIM CSV")
    parser.add_argument("--grouped-gt-root", required=True, help="Root directory of grouped GT references")
    parser.add_argument("--calibration-root", required=True, help="Root directory of grouped calibration results")
    parser.add_argument("--save-path", required=True, help="Output root directory for evaluation results")

    parser.add_argument(
        "--eval-mode",
        choices=["all", "calibrated_only"],
        default="all",
        help="all = fixed alpha/beta + calibrated; calibrated_only = only calibrated alpha/beta",
    )
    parser.add_argument(
        "--fixed-alpha-beta-pairs",
        default="0.0:0.1,0.0:0.2,0.0:0.3,0.0:0.4,0.0:0.5,0.5:0.2,0.5:0.1,1.0:0.2,1.0:0.1",
        help="Comma-separated a:b pairs for Fixed Alpha/Beta baselines",
    )
    parser.add_argument(
        "--main-binary-threshold",
        type=float,
        default=0.5,
        help="Fallback threshold for GT binary label if grouped GT csv does not already contain gt_is_consistent_binary",
    )
    parser.add_argument(
        "--exclude-empty-empty-from-main-eval",
        action="store_true",
        help="Exclude transitions where GT object counts at t and t+1 are both zero from evaluation",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Comma-separated prefix keywords or a file containing them",
    )
    parser.add_argument(
        "--filter-mode",
        type=str,
        choices=["exact", "contains"],
        default=None,
        help="exact: exclude exact prefixes entirely; contains: keep only prefixes containing any filter token",
    )
    parser.add_argument(
        "--enable-fixed-method-fair-filter",
        action="store_true",
        help="Apply fairness constraints to fixed alpha/beta baselines before they enter any output.",
    )
    parser.add_argument(
        "--positive-rate-tolerance",
        type=float,
        default=0.05,
        help="Keep fixed alpha/beta methods only if their predicted positive rate is within this tolerance of the calibrated method for the same prefix.",
    )
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    fixed_alpha_beta_pairs = parse_alpha_beta_pairs(args.fixed_alpha_beta_pairs)

    summary = evaluate_grouped_consistency(
        frame_path=args.frame_path,
        ssim_path=args.ssim_path,
        grouped_gt_root=args.grouped_gt_root,
        calibration_root=args.calibration_root,
        save_path=args.save_path,
        eval_mode=args.eval_mode,
        fixed_alpha_beta_pairs=fixed_alpha_beta_pairs,
        main_binary_threshold=args.main_binary_threshold,
        exclude_empty_empty_from_main_eval=args.exclude_empty_empty_from_main_eval,
        filter_arg=args.filter,
        filter_mode=args.filter_mode,
        enable_fixed_method_fair_filter=args.enable_fixed_method_fair_filter,
        positive_rate_tolerance=args.positive_rate_tolerance,
    )

    print("Saved evaluation index summary to:", Path(args.save_path) / "index_summary.json")
    print("Processed prefixes:", summary["num_processed_prefixes"])
