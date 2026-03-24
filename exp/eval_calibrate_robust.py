import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".txt")


def normalize_transition_key(path_str: str, remove_final_suffix: bool = True) -> str:
    key = os.path.basename(str(path_str).strip())
    if remove_final_suffix:
        lowered = key.lower()
        for ext in VALID_EXTS:
            if lowered.endswith(ext):
                key = key[: -len(ext)]
                break
    return key


def extract_prefix_from_key(key: str) -> Optional[str]:
    key = normalize_transition_key(key, remove_final_suffix=True)
    if "_frame" not in key:
        return None
    return key.split("_frame", 1)[0]


def extract_frame_consistency(entry) -> Optional[float]:
    if entry is None or not isinstance(entry, dict):
        return None
    value = entry.get("frame_consistency", entry.get("score"))
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def safe_float(x, default=None):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def parse_alpha_beta_pair(text: str) -> Tuple[float, float]:
    s = str(text).strip()
    if not s:
        raise ValueError("fallback fixed alpha/beta cannot be empty")
    if ":" in s:
        a_str, b_str = s.split(":", 1)
    elif "," in s:
        a_str, b_str = s.split(",", 1)
    elif "/" in s:
        a_str, b_str = s.split("/", 1)
    else:
        raise ValueError(
            f"Invalid alpha/beta pair '{text}'. Use format alpha:beta, alpha,beta, or alpha/beta"
        )
    return float(a_str.strip()), float(b_str.strip())


@dataclass
class PrefixCalibrationResult:
    prefix: str
    output_dir: str
    mode: str
    alpha: float
    beta: float
    best_q: Optional[float]
    n_raw: int
    n_filtered: int
    n_removed_outliers: int
    target_coverage: float
    empirical_coverage: Optional[float]
    pinball_loss: Optional[float]
    score: Optional[float]
    fallback_reason: Optional[str]
    fallback_fixed_alpha: float
    fallback_fixed_beta: float
    raw_consistency_min: Optional[float]
    raw_consistency_p50: Optional[float]
    raw_consistency_p90: Optional[float]
    raw_consistency_max: Optional[float]
    raw_consistency_std: Optional[float]
    raw_ssim_min: Optional[float]
    raw_ssim_max: Optional[float]
    filtered_consistency_min: Optional[float]
    filtered_consistency_p50: Optional[float]
    filtered_consistency_p90: Optional[float]
    filtered_consistency_max: Optional[float]
    filtered_consistency_std: Optional[float]
    filtered_ssim_min: Optional[float]
    filtered_ssim_max: Optional[float]


def load_object_consistency(frame_path: str) -> Dict[str, dict]:
    with open(frame_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Object consistency JSON must be a dict keyed by transition key.")
    return data


def load_ssim_csv(ssim_path: str) -> pd.DataFrame:
    df = pd.read_csv(ssim_path)
    required_cols = {"file_name", "ssim_value"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in SSIM CSV: {sorted(missing_cols)}")
    return df


def build_prefix_dataset(
    prefix: str,
    object_consistency_dict: Dict[str, dict],
    ssim_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    obj_rows = []
    obj_seen = set()
    obj_duplicates = 0
    obj_missing_consistency = 0

    for raw_key, entry in object_consistency_dict.items():
        clean_key = normalize_transition_key(raw_key, True)
        pfx = extract_prefix_from_key(clean_key)
        if pfx != prefix:
            continue
        if clean_key in obj_seen:
            obj_duplicates += 1
            continue
        obj_seen.add(clean_key)

        consistency = extract_frame_consistency(entry)
        if consistency is None:
            obj_missing_consistency += 1
            continue

        obj_rows.append({"transition_key": clean_key, "prefix": prefix, "consistency": consistency})

    ssim_rows = []
    ssim_seen = set()
    ssim_duplicates = 0
    ssim_bad_value = 0

    for _, row in ssim_df.iterrows():
        clean_key = normalize_transition_key(row["file_name"], True)
        pfx = extract_prefix_from_key(clean_key)
        if pfx != prefix:
            continue
        if clean_key in ssim_seen:
            ssim_duplicates += 1
            continue
        ssim_seen.add(clean_key)

        ssim_val = safe_float(row["ssim_value"], None)
        if ssim_val is None:
            ssim_bad_value += 1
            continue

        ssim_rows.append({"transition_key": clean_key, "prefix": prefix, "ssim": ssim_val})

    obj_df = pd.DataFrame(obj_rows)
    ssim_sub_df = pd.DataFrame(ssim_rows)

    if obj_df.empty or ssim_sub_df.empty:
        merged = pd.DataFrame(columns=["transition_key", "prefix", "ssim", "consistency"])
    else:
        merged = pd.merge(obj_df, ssim_sub_df, on=["transition_key", "prefix"], how="inner")
        merged = merged[["transition_key", "prefix", "ssim", "consistency"]].copy()
        merged = merged.sort_values(["prefix", "transition_key"]).reset_index(drop=True)

    stats = {
        "obj_unique": int(len(obj_df)),
        "ssim_unique": int(len(ssim_sub_df)),
        "matched": int(len(merged)),
        "obj_duplicates": int(obj_duplicates),
        "ssim_duplicates": int(ssim_duplicates),
        "obj_missing_consistency": int(obj_missing_consistency),
        "ssim_bad_value": int(ssim_bad_value),
        "obj_only": int(max(len(obj_df) - len(merged), 0)),
        "ssim_only": int(max(len(ssim_sub_df) - len(merged), 0)),
    }
    return merged, stats


def summarize_series(s: pd.Series) -> Dict[str, Optional[float]]:
    if s is None or len(s) == 0:
        return {"min": None, "p50": None, "p90": None, "max": None, "std": None}
    arr = s.to_numpy(dtype=float)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.quantile(arr, 0.5)),
        "p90": float(np.quantile(arr, 0.9)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def filter_outliers_percentile(
    df: pd.DataFrame,
    lower_q: float,
    upper_q: float,
    apply_to_ssim: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if df.empty:
        return df.copy(), {}

    c_low = float(df["consistency"].quantile(lower_q))
    c_high = float(df["consistency"].quantile(upper_q))
    mask = (df["consistency"] >= c_low) & (df["consistency"] <= c_high)
    info = {"consistency_lower": c_low, "consistency_upper": c_high}

    if apply_to_ssim:
        s_low = float(df["ssim"].quantile(lower_q))
        s_high = float(df["ssim"].quantile(upper_q))
        mask &= (df["ssim"] >= s_low) & (df["ssim"] <= s_high)
        info["ssim_lower"] = s_low
        info["ssim_upper"] = s_high

    return df.loc[mask].copy().reset_index(drop=True), info


def detect_degenerate_prefix(
    df: pd.DataFrame,
    min_samples: int,
    median_threshold: float,
    p90_threshold: float,
    std_threshold: float,
) -> Optional[str]:
    if len(df) < min_samples:
        return f"too_few_samples(n={len(df)} < {min_samples})"
    cons = df["consistency"]
    s = summarize_series(cons)
    if s["p50"] is not None and s["p50"] < median_threshold:
        return f"low_median_consistency(p50={s['p50']:.4f} < {median_threshold})"
    if s["p90"] is not None and s["p90"] < p90_threshold:
        return f"low_p90_consistency(p90={s['p90']:.4f} < {p90_threshold})"
    if s["std"] is not None and s["std"] < std_threshold:
        return f"low_consistency_std(std={s['std']:.4f} < {std_threshold})"
    return None


def fit_quantile_boundary(df: pd.DataFrame, candidate_quantiles: List[float], target_coverage: float) -> Dict[str, float]:
    if len(df) < 3:
        raise ValueError(f"Not enough samples to fit quantile regression: {len(df)}")

    best = None
    model = smf.quantreg("consistency ~ ssim", df)
    for q in candidate_quantiles:
        try:
            res = model.fit(q=q)
            alpha = float(res.params.get("ssim", 0.0))
            beta = float(res.params.get("Intercept", 0.0))
            y_hat = np.clip(alpha * df["ssim"].to_numpy() + beta, 0.0, 1.0)
            coverage = float(np.mean(df["consistency"].to_numpy() >= y_hat))
            cov_error = abs(coverage - target_coverage)
            pbl = pinball_loss(df["consistency"].to_numpy(), y_hat, q)
            score = cov_error + 0.1 * pbl
            cand = {
                "q": float(q),
                "alpha": alpha,
                "beta": beta,
                "coverage": coverage,
                "pinball_loss": pbl,
                "score": score,
            }
            if best is None or cand["score"] < best["score"]:
                best = cand
        except Exception:
            continue
    if best is None:
        raise RuntimeError("All quantile regression fits failed.")
    return best


def save_calibration_plot(df_raw: pd.DataFrame, df_used: pd.DataFrame, alpha: float, beta: float, best_q: Optional[float], out_path: str, title: str):
    if plt is None or df_raw.empty:
        return
    x_raw = df_raw["ssim"].to_numpy(dtype=float)
    y_raw = df_raw["consistency"].to_numpy(dtype=float)
    x_min = float(np.min(x_raw))
    x_max = float(np.max(x_raw))
    if x_max <= x_min:
        x_max = x_min + 1e-6
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = np.clip(alpha * x_vals + beta, 0.0, 1.0)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_raw, y_raw, alpha=0.25, label="Raw points")
    if not df_used.empty:
        plt.scatter(df_used["ssim"].to_numpy(dtype=float), df_used["consistency"].to_numpy(dtype=float), alpha=0.70, label="Used for fit")
    label = f"Boundary (q={best_q:.2f})" if best_q is not None else "Fallback boundary"
    plt.plot(x_vals, y_vals, linewidth=2, label=label)
    plt.xlabel("SSIM")
    plt.ylabel("Frame-level consistency")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def robust_group_calibration(args) -> Dict[str, object]:
    os.makedirs(args.save_path, exist_ok=True)

    object_consistency_dict = load_object_consistency(args.frame_path)
    ssim_df = load_ssim_csv(args.ssim_path)

    object_prefixes = sorted({p for p in (extract_prefix_from_key(k) for k in object_consistency_dict.keys()) if p is not None})
    ssim_prefixes = sorted({p for p in (extract_prefix_from_key(k) for k in ssim_df["file_name"].astype(str).tolist()) if p is not None})

    common_prefixes = sorted(set(object_prefixes) & set(ssim_prefixes))
    object_only_prefixes = sorted(set(object_prefixes) - set(ssim_prefixes))
    ssim_only_prefixes = sorted(set(ssim_prefixes) - set(object_prefixes))

    global_fit = None
    global_fit_df = []
    if args.fallback_mode == "global":
        for prefix in common_prefixes:
            df_prefix, _ = build_prefix_dataset(prefix, object_consistency_dict, ssim_df)
            if df_prefix.empty:
                continue
            df_filtered, _ = filter_outliers_percentile(
                df_prefix,
                args.outlier_lower_q,
                args.outlier_upper_q,
                apply_to_ssim=args.outlier_filter_ssim,
            )
            degenerate_reason = detect_degenerate_prefix(
                df_filtered,
                min_samples=args.min_valid_samples,
                median_threshold=args.degenerate_median_threshold,
                p90_threshold=args.degenerate_p90_threshold,
                std_threshold=args.degenerate_std_threshold,
            )
            if degenerate_reason is None and not df_filtered.empty:
                global_fit_df.append(df_filtered)
        if global_fit_df:
            global_df = pd.concat(global_fit_df, ignore_index=True)
            global_fit = fit_quantile_boundary(
                global_df,
                candidate_quantiles=args.candidate_quantiles,
                target_coverage=args.target_coverage,
            )

    per_prefix_results: List[PrefixCalibrationResult] = []
    per_prefix_metrics_rows = []

    for idx, prefix in enumerate(common_prefixes):
        prefix_dir = os.path.join(args.save_path, f"{idx:02d}_{prefix}")
        os.makedirs(prefix_dir, exist_ok=True)

        raw_df, align_stats = build_prefix_dataset(prefix, object_consistency_dict, ssim_df)
        raw_stats_cons = summarize_series(raw_df["consistency"] if not raw_df.empty else pd.Series(dtype=float))
        raw_stats_ssim = summarize_series(raw_df["ssim"] if not raw_df.empty else pd.Series(dtype=float))

        filtered_df, filter_info = filter_outliers_percentile(
            raw_df,
            args.outlier_lower_q,
            args.outlier_upper_q,
            apply_to_ssim=args.outlier_filter_ssim,
        )
        degenerate_reason = detect_degenerate_prefix(
            filtered_df,
            min_samples=args.min_valid_samples,
            median_threshold=args.degenerate_median_threshold,
            p90_threshold=args.degenerate_p90_threshold,
            std_threshold=args.degenerate_std_threshold,
        )

        mode = "calibrated"
        alpha = None
        beta = None
        best_q = None
        empirical_coverage = None
        fit_pinball = None
        fit_score = None
        fallback_reason = None

        if degenerate_reason is None:
            try:
                best = fit_quantile_boundary(filtered_df, args.candidate_quantiles, args.target_coverage)
                alpha = float(best["alpha"])
                beta = float(best["beta"])
                best_q = float(best["q"])
                empirical_coverage = float(best["coverage"])
                fit_pinball = float(best["pinball_loss"])
                fit_score = float(best["score"])
            except Exception as e:
                degenerate_reason = f"fit_failed({e})"

        if degenerate_reason is not None:
            fallback_reason = degenerate_reason
            if args.fallback_mode == "fixed":
                mode = "fallback_fixed"
                alpha = float(args.fallback_fixed_alpha)
                beta = float(args.fallback_fixed_beta)
            elif args.fallback_mode == "global" and global_fit is not None:
                mode = "fallback_global"
                alpha = float(global_fit["alpha"])
                beta = float(global_fit["beta"])
                best_q = float(global_fit["q"])
                empirical_coverage = float(global_fit["coverage"])
                fit_pinball = float(global_fit["pinball_loss"])
                fit_score = float(global_fit["score"])
            else:
                mode = "fallback_fixed"
                alpha = float(args.fallback_fixed_alpha)
                beta = float(args.fallback_fixed_beta)
                if args.fallback_mode == "global" and global_fit is None:
                    fallback_reason += ";global_fit_unavailable"

        filtered_stats_cons = summarize_series(filtered_df["consistency"] if not filtered_df.empty else pd.Series(dtype=float))
        filtered_stats_ssim = summarize_series(filtered_df["ssim"] if not filtered_df.empty else pd.Series(dtype=float))

        raw_df.to_csv(os.path.join(prefix_dir, "calibration_dataset_raw.csv"), index=False)
        filtered_df.to_csv(os.path.join(prefix_dir, "calibration_dataset_used.csv"), index=False)

        summary_payload = {
            "prefix": prefix,
            "mode": mode,
            "alpha": float(alpha),
            "beta": float(beta),
            "best_q": best_q,
            "target_coverage": args.target_coverage,
            "empirical_coverage": empirical_coverage,
            "pinball_loss": fit_pinball,
            "score": fit_score,
            "fallback_reason": fallback_reason,
            "fallback_mode": args.fallback_mode,
            "fallback_fixed_alpha": float(args.fallback_fixed_alpha),
            "fallback_fixed_beta": float(args.fallback_fixed_beta),
            "fallback_fixed_alpha_beta": {"alpha": float(args.fallback_fixed_alpha), "beta": float(args.fallback_fixed_beta)},
            "outlier_filter": {
                "mode": "percentile",
                "lower_q": args.outlier_lower_q,
                "upper_q": args.outlier_upper_q,
                "apply_to_ssim": bool(args.outlier_filter_ssim),
                "filter_info": filter_info,
            },
            "degenerate_checks": {
                "min_valid_samples": args.min_valid_samples,
                "median_threshold": args.degenerate_median_threshold,
                "p90_threshold": args.degenerate_p90_threshold,
                "std_threshold": args.degenerate_std_threshold,
            },
            "alignment_stats": align_stats,
            "raw_stats": {"consistency": raw_stats_cons, "ssim": raw_stats_ssim},
            "used_stats": {"consistency": filtered_stats_cons, "ssim": filtered_stats_ssim},
        }
        with open(os.path.join(prefix_dir, "calibration_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)

        if args.plot:
            save_calibration_plot(
                df_raw=raw_df,
                df_used=filtered_df,
                alpha=float(alpha),
                beta=float(beta),
                best_q=best_q,
                out_path=os.path.join(prefix_dir, "best_consistency_boundary.png"),
                title=f"{prefix} | {mode}",
            )

        result = PrefixCalibrationResult(
            prefix=prefix,
            output_dir=prefix_dir,
            mode=mode,
            alpha=float(alpha),
            beta=float(beta),
            best_q=None if best_q is None else float(best_q),
            n_raw=int(len(raw_df)),
            n_filtered=int(len(filtered_df)),
            n_removed_outliers=int(max(len(raw_df) - len(filtered_df), 0)),
            target_coverage=float(args.target_coverage),
            empirical_coverage=empirical_coverage,
            pinball_loss=fit_pinball,
            score=fit_score,
            fallback_reason=fallback_reason,
            fallback_fixed_alpha=float(args.fallback_fixed_alpha),
            fallback_fixed_beta=float(args.fallback_fixed_beta),
            raw_consistency_min=raw_stats_cons["min"],
            raw_consistency_p50=raw_stats_cons["p50"],
            raw_consistency_p90=raw_stats_cons["p90"],
            raw_consistency_max=raw_stats_cons["max"],
            raw_consistency_std=raw_stats_cons["std"],
            raw_ssim_min=raw_stats_ssim["min"],
            raw_ssim_max=raw_stats_ssim["max"],
            filtered_consistency_min=filtered_stats_cons["min"],
            filtered_consistency_p50=filtered_stats_cons["p50"],
            filtered_consistency_p90=filtered_stats_cons["p90"],
            filtered_consistency_max=filtered_stats_cons["max"],
            filtered_consistency_std=filtered_stats_cons["std"],
            filtered_ssim_min=filtered_stats_ssim["min"],
            filtered_ssim_max=filtered_stats_ssim["max"],
        )
        per_prefix_results.append(result)
        per_prefix_metrics_rows.append(asdict(result))

    per_prefix_df = pd.DataFrame(per_prefix_metrics_rows)
    per_prefix_csv = os.path.join(args.save_path, "per_prefix_calibration_metrics.csv")
    per_prefix_df.to_csv(per_prefix_csv, index=False)

    results_payload = [
        {
            "prefix": r.prefix,
            "output_dir": r.output_dir,
            "mode": r.mode,
            "alpha": r.alpha,
            "beta": r.beta,
            "fallback_reason": r.fallback_reason,
            "source": os.path.join(r.output_dir, "calibration_summary.json"),
        }
        for r in per_prefix_results
    ]

    index_summary = {
        "frame_path": args.frame_path,
        "ssim_path": args.ssim_path,
        "save_path": args.save_path,
        "num_object_prefixes": len(object_prefixes),
        "num_ssim_prefixes": len(ssim_prefixes),
        "num_common_prefixes": len(common_prefixes),
        "object_only_prefixes": object_only_prefixes,
        "ssim_only_prefixes": ssim_only_prefixes,
        "fallback_mode": args.fallback_mode,
        "fallback_fixed_alpha": float(args.fallback_fixed_alpha),
        "fallback_fixed_beta": float(args.fallback_fixed_beta),
        "fallback_fixed_alpha_beta": {"alpha": float(args.fallback_fixed_alpha), "beta": float(args.fallback_fixed_beta)},
        "global_fit": global_fit,
        "candidate_quantiles": args.candidate_quantiles,
        "target_coverage": args.target_coverage,
        "outlier_filter": {
            "mode": "percentile",
            "lower_q": args.outlier_lower_q,
            "upper_q": args.outlier_upper_q,
            "apply_to_ssim": bool(args.outlier_filter_ssim),
        },
        "degenerate_checks": {
            "min_valid_samples": args.min_valid_samples,
            "median_threshold": args.degenerate_median_threshold,
            "p90_threshold": args.degenerate_p90_threshold,
            "std_threshold": args.degenerate_std_threshold,
        },
        "num_prefixes_calibrated": int(sum(r.mode == "calibrated" for r in per_prefix_results)),
        "num_prefixes_fallback_fixed": int(sum(r.mode == "fallback_fixed" for r in per_prefix_results)),
        "num_prefixes_fallback_global": int(sum(r.mode == "fallback_global" for r in per_prefix_results)),
        "per_prefix_metrics_csv": per_prefix_csv,
        "results": results_payload,
        "outputs": results_payload,
    }
    with open(os.path.join(args.save_path, "index_summary.json"), "w", encoding="utf-8") as f:
        json.dump(index_summary, f, ensure_ascii=False, indent=2)

    return index_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Robust grouped calibration for frame-level consistency boundaries.")
    parser.add_argument("--frame-path", required=True, help="Path to object consistency JSON")
    parser.add_argument("--ssim-path", required=True, help="Path to SSIM CSV")
    parser.add_argument("--save-path", required=True, help="Output directory root")
    parser.add_argument("--candidate-quantiles", default="0.05,0.10,0.15,0.20,0.25,0.30", help="Comma-separated quantiles to try")
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--plot", action="store_true", help="Save per-prefix plots")
    parser.add_argument("--outlier-lower-q", type=float, default=0.05, help="Lower percentile for outlier filtering")
    parser.add_argument("--outlier-upper-q", type=float, default=0.95, help="Upper percentile for outlier filtering")
    parser.add_argument("--outlier-filter-ssim", action="store_true", help="Also filter extreme SSIM values by the same percentiles")
    parser.add_argument("--min-valid-samples", type=int, default=30)
    parser.add_argument("--degenerate-median-threshold", type=float, default=0.10)
    parser.add_argument("--degenerate-p90-threshold", type=float, default=0.20)
    parser.add_argument("--degenerate-std-threshold", type=float, default=0.03)
    parser.add_argument("--fallback-mode", choices=["fixed", "global"], default="fixed", help="How to handle degenerate prefixes")
    parser.add_argument("--fallback-fixed-threshold", type=str, default=None, help="Deprecated alias; linear boundary alpha:beta")
    parser.add_argument("--fallback-fixed-alpha-beta", type=str, default="1.0:0.1", help="Used when fallback-mode=fixed; linear boundary alpha:beta, default 1.0:0.1")

    args = parser.parse_args()
    args.candidate_quantiles = [float(x) for x in args.candidate_quantiles.split(",") if str(x).strip()]
    if not args.candidate_quantiles:
        raise ValueError("candidate_quantiles is empty")
    if not (0.0 <= args.outlier_lower_q < args.outlier_upper_q <= 1.0):
        raise ValueError("Require 0 <= outlier_lower_q < outlier_upper_q <= 1")

    fixed_pair_text = args.fallback_fixed_alpha_beta
    if args.fallback_fixed_threshold is not None:
        fixed_pair_text = args.fallback_fixed_threshold
    args.fallback_fixed_alpha, args.fallback_fixed_beta = parse_alpha_beta_pair(fixed_pair_text)
    return args


if __name__ == "__main__":
    args = parse_args()
    summary = robust_group_calibration(args)
    # print(json.dumps(summary, ensure_ascii=False, indent=2))