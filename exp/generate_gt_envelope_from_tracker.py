from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm  # type: ignore
    STATSMODELS_AVAILABLE = True
except Exception:
    sm = None
    STATSMODELS_AVAILABLE = False


def normalize_transition_key(path_like: str, remove_final_suffix: bool = True) -> str:
    key = Path(str(path_like).strip()).name
    if remove_final_suffix:
        lowered = key.lower()
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".txt"):
            if lowered.endswith(ext):
                key = key[: -len(ext)]
                break
    return key


def load_ssim_mapping(ssim_path: Path) -> Dict[str, float]:
    df = pd.read_csv(ssim_path)
    cols = {c.lower(): c for c in df.columns}

    # preferred: transition_key + ssim_value
    if "transition_key" in cols and "ssim_value" in cols:
        out = {}
        for _, row in df.iterrows():
            key = normalize_transition_key(str(row[cols["transition_key"]]), remove_final_suffix=False)
            out[key] = float(row[cols["ssim_value"]])
        return out

    # common pairwise format: file_name, next_file_name, ssim_value
    if "file_name" in cols and "next_file_name" in cols and "ssim_value" in cols:
        out = {}
        for _, row in df.iterrows():
            key = normalize_transition_key(str(row[cols["file_name"]]))
            out[key] = float(row[cols["ssim_value"]])
        return out

    # fallback: file_name + ssim_value
    if "file_name" in cols and "ssim_value" in cols:
        out = {}
        for _, row in df.iterrows():
            key = normalize_transition_key(str(row[cols["file_name"]]))
            out[key] = float(row[cols["ssim_value"]])
        return out

    raise ValueError(
        "Unsupported SSIM CSV format. Expected one of: "
        "(transition_key, ssim_value), (file_name, next_file_name, ssim_value), or (file_name, ssim_value)."
    )


def iter_tracker_object_jsons(grouped_gt_root: Path) -> Iterable[Tuple[str, Path]]:
    for child in sorted(grouped_gt_root.iterdir()):
        if not child.is_dir():
            continue
        object_json = child / "gt_object_matches.json"
        if object_json.exists():
            yield child.name, object_json


def build_gt_envelope_points(grouped_gt_root: Path, ssim_map: Dict[str, float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, object_json_path in iter_tracker_object_jsons(grouped_gt_root):
        with open(object_json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        for transition_key, record in obj.items():
            if bool(record.get("is_terminal", False)):
                continue
            if not bool(record.get("valid_transition", True)):
                continue

            prefix = str(record.get("prefix", ""))
            key_norm = normalize_transition_key(str(transition_key), remove_final_suffix=False)
            ssim = ssim_map.get(key_norm)
            if ssim is None:
                # try normalized without extension just in case
                ssim = ssim_map.get(normalize_transition_key(str(transition_key)))
            if ssim is None:
                continue

            matches = record.get("matches", []) or []
            for m in matches:
                rows.append(
                    {
                        "prefix": prefix,
                        "transition_key": key_norm,
                        "ssim": float(ssim),
                        "iou_gt": float(m.get("iou", np.nan)),
                        "class_id": int(m.get("class_id", -1)),
                        "box_t_idx": int(m.get("box_t_idx", -1)),
                        "box_t1_idx": int(m.get("box_t1_idx", -1)),
                        "match_cost": float(m.get("cost", np.nan)),
                        "center_dist": float(m.get("center_dist", np.nan)),
                        "scale_change": float(m.get("scale_change", np.nan)),
                        "aspect_change": float(m.get("aspect_change", np.nan)),
                        "gt_num_t": int(record.get("gt_num_t", 0)),
                        "gt_num_t1": int(record.get("gt_num_t1", 0)),
                        "gt_num_matches": int(record.get("gt_num_matches", 0)),
                        "gt_match_coverage": float(record.get("gt_match_coverage", np.nan)),
                        "gt_frame_consistency": float(record.get("gt_frame_consistency", np.nan)),
                        "gt_mean_match_iou": float(record.get("gt_mean_match_iou", record.get("gt_mean_match_quality", np.nan))),
                        "exclude_from_main_eval": bool(record.get("exclude_from_main_eval", False)),
                        "gt_is_empty_empty": bool(record.get("gt_is_empty_empty", False)),
                    }
                )

    return pd.DataFrame(rows)


def fit_linear_quantile(points: pd.DataFrame, quantile: float) -> Dict[str, Any]:
    if len(points) < 8:
        return {
            "alpha": np.nan,
            "beta": np.nan,
            "n_points": int(len(points)),
            "quantile": float(quantile),
            "fit_ok": False,
            "reason": "too_few_points",
        }

    x = points["ssim"].astype(float).to_numpy()
    y = points["iou_gt"].astype(float).to_numpy()

    if np.nanstd(x) < 1e-8:
        return {
            "alpha": np.nan,
            "beta": np.nan,
            "n_points": int(len(points)),
            "quantile": float(quantile),
            "fit_ok": False,
            "reason": "ssim_nearly_constant",
        }

    if STATSMODELS_AVAILABLE:
        try:
            X = sm.add_constant(x)
            model = sm.QuantReg(y, X)
            res = model.fit(q=quantile)
            beta = float(res.params[0])
            alpha = float(res.params[1])
            return {
                "alpha": alpha,
                "beta": beta,
                "n_points": int(len(points)),
                "quantile": float(quantile),
                "fit_ok": True,
                "reason": "ok_statsmodels",
            }
        except Exception as e:
            reason = f"statsmodels_failed:{type(e).__name__}"
    else:
        reason = "statsmodels_unavailable"

    # Fallback: lower-quantile binning + OLS on binwise quantiles
    try:
        tmp = points[["ssim", "iou_gt"]].dropna().copy()
        tmp["bin"] = pd.qcut(tmp["ssim"], q=min(10, max(3, len(tmp) // 20)), duplicates="drop")
        grp = tmp.groupby("bin", observed=False).agg(ssim=("ssim", "median"), iou_q=("iou_gt", lambda s: float(np.quantile(s, quantile)))).reset_index(drop=True)
        if len(grp) < 3:
            return {
                "alpha": np.nan,
                "beta": np.nan,
                "n_points": int(len(points)),
                "quantile": float(quantile),
                "fit_ok": False,
                "reason": f"{reason}|fallback_too_few_bins",
            }
        x2 = grp["ssim"].to_numpy(dtype=float)
        y2 = grp["iou_q"].to_numpy(dtype=float)
        alpha, beta = np.polyfit(x2, y2, 1)
        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "n_points": int(len(points)),
            "quantile": float(quantile),
            "fit_ok": True,
            "reason": f"{reason}|fallback_binned_ols",
        }
    except Exception as e:
        return {
            "alpha": np.nan,
            "beta": np.nan,
            "n_points": int(len(points)),
            "quantile": float(quantile),
            "fit_ok": False,
            "reason": f"{reason}|fallback_failed:{type(e).__name__}",
        }


def build_envelopes(points_df: pd.DataFrame, quantiles: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_prefix_rows: List[Dict[str, Any]] = []
    global_rows: List[Dict[str, Any]] = []

    for q in quantiles:
        global_rows.append({"prefix": "__GLOBAL__", **fit_linear_quantile(points_df, q)})
        for prefix, sub in points_df.groupby("prefix"):
            per_prefix_rows.append({"prefix": prefix, **fit_linear_quantile(sub, q)})

    return pd.DataFrame(per_prefix_rows), pd.DataFrame(global_rows)


def line_distance_metrics(alpha_ref: float, beta_ref: float, alpha_cmp: float, beta_cmp: float, ssim_min: float = 0.4, ssim_max: float = 1.0, num: int = 50) -> Dict[str, float]:
    grid = np.linspace(ssim_min, ssim_max, num=num)
    y_ref = alpha_ref * grid + beta_ref
    y_cmp = alpha_cmp * grid + beta_cmp
    dif = np.abs(y_ref - y_cmp)
    return {
        "line_mae": float(np.mean(dif)),
        "line_max_err": float(np.max(dif)),
    }


def compare_with_calibration(per_prefix_gt: pd.DataFrame, calibration_csv: Path) -> pd.DataFrame:
    cal = pd.read_csv(calibration_csv).copy()
    need = {"prefix", "alpha", "beta"}
    if not need.issubset(cal.columns):
        raise ValueError(f"Calibration CSV must contain columns {sorted(need)}")

    merged = per_prefix_gt.merge(cal[["prefix", "alpha", "beta"]].rename(columns={"alpha": "alpha_cal", "beta": "beta_cal"}), on="prefix", how="inner")
    rows: List[Dict[str, Any]] = []
    for _, r in merged.iterrows():
        if pd.isna(r["alpha"]) or pd.isna(r["beta"]) or pd.isna(r["alpha_cal"]) or pd.isna(r["beta_cal"]):
            continue
        d = line_distance_metrics(float(r["alpha"]), float(r["beta"]), float(r["alpha_cal"]), float(r["beta_cal"]))
        rows.append({
            "prefix": r["prefix"],
            "quantile": float(r["quantile"]),
            "alpha_gt": float(r["alpha"]),
            "beta_gt": float(r["beta"]),
            "alpha_cal": float(r["alpha_cal"]),
            "beta_cal": float(r["beta_cal"]),
            "abs_alpha_diff": float(abs(float(r["alpha"]) - float(r["alpha_cal"]))),
            "abs_beta_diff": float(abs(float(r["beta"]) - float(r["beta_cal"]))),
            **d,
        })
    return pd.DataFrame(rows)


def parse_quantiles(text: str) -> List[float]:
    out = []
    for p in str(text).split(","):
        p = p.strip()
        if not p:
            continue
        q = float(p)
        if not (0.0 < q < 1.0):
            raise ValueError(f"Invalid quantile: {q}")
        out.append(q)
    if not out:
        raise ValueError("No valid quantiles provided")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GT envelope points and quantile-regression envelopes from tracker-based grouped_gt results.")
    parser.add_argument("--grouped-gt-root", required=True, help="Root directory produced by generate_gt_consistency_reference_tracker.py")
    parser.add_argument("--ssim-path", required=True, help="CSV containing transition SSIM values")
    parser.add_argument("--save-path", required=True, help="Output directory")
    parser.add_argument("--quantiles", default="0.05,0.1", help="Comma-separated quantiles, e.g. 0.05,0.1")
    parser.add_argument("--calibration-csv", default=None, help="Optional CSV with per-prefix calibrated alpha/beta to compare against GT envelopes")
    args = parser.parse_args()

    grouped_gt_root = Path(args.grouped_gt_root)
    ssim_path = Path(args.ssim_path)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    quantiles = parse_quantiles(args.quantiles)
    ssim_map = load_ssim_mapping(ssim_path)
    points_df = build_gt_envelope_points(grouped_gt_root, ssim_map)
    if points_df.empty:
        raise RuntimeError("No GT envelope points could be built. Check grouped_gt_root and SSIM alignment.")

    points_csv = save_path / "gt_envelope_points.csv"
    points_df.to_csv(points_csv, index=False)

    per_prefix_gt, global_gt = build_envelopes(points_df, quantiles)
    per_prefix_csv = save_path / "per_prefix_gt_envelopes.csv"
    global_csv = save_path / "global_gt_envelopes.csv"
    per_prefix_gt.to_csv(per_prefix_csv, index=False)
    global_gt.to_csv(global_csv, index=False)

    summary: Dict[str, Any] = {
        "grouped_gt_root": str(grouped_gt_root),
        "ssim_path": str(ssim_path),
        "save_path": str(save_path),
        "quantiles": quantiles,
        "num_points": int(len(points_df)),
        "num_prefixes": int(points_df["prefix"].nunique()),
        "statsmodels_available": bool(STATSMODELS_AVAILABLE),
        "outputs": {
            "gt_envelope_points_csv": str(points_csv),
            "per_prefix_gt_envelopes_csv": str(per_prefix_csv),
            "global_gt_envelopes_csv": str(global_csv),
        },
    }

    if args.calibration_csv:
        sim_df = compare_with_calibration(per_prefix_gt, Path(args.calibration_csv))
        sim_csv = save_path / "per_prefix_gt_vs_calibrated_similarity.csv"
        sim_df.to_csv(sim_csv, index=False)
        summary["outputs"]["per_prefix_gt_vs_calibrated_similarity_csv"] = str(sim_csv)
        summary["num_similarity_rows"] = int(len(sim_df))

    with open(save_path / "gt_envelope_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote GT envelope outputs to: {save_path}")


if __name__ == "__main__":
    main()
