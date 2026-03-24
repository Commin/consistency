
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


REQUIRED_METHOD_COLS = {
    "prefix",
    "method_slug",
    "method_display_name",
    "method_type",
    "alpha",
    "beta",
}


def find_per_prefix_metrics_csvs(root_path: Path) -> List[Path]:
    return sorted(p for p in root_path.rglob("per_prefix_metrics.csv") if p.is_file())


def line_distance_metrics(
    alpha_ref: float,
    beta_ref: float,
    alpha_cmp: float,
    beta_cmp: float,
    ssim_min: float = 0.4,
    ssim_max: float = 1.0,
    num: int = 100,
) -> Dict[str, float]:
    grid = np.linspace(ssim_min, ssim_max, num=num)
    y_ref = alpha_ref * grid + beta_ref
    y_cmp = alpha_cmp * grid + beta_cmp
    dif = y_cmp - y_ref
    abs_dif = np.abs(dif)
    return {
        "line_mae": float(np.mean(abs_dif)),
        "line_rmse": float(np.sqrt(np.mean(dif ** 2))),
        "line_max_err": float(np.max(abs_dif)),
        "signed_line_bias": float(np.mean(dif)),
    }


def normalize_method_rows(per_prefix_metrics_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(per_prefix_metrics_csv)
    missing = REQUIRED_METHOD_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{per_prefix_metrics_csv} is missing columns: {sorted(missing)}")

    keep_cols = list(REQUIRED_METHOD_COLS)
    out = df[keep_cols].copy()
    out["alpha"] = pd.to_numeric(out["alpha"], errors="coerce")
    out["beta"] = pd.to_numeric(out["beta"], errors="coerce")
    out = out.dropna(subset=["prefix", "method_slug", "alpha", "beta"])

    out = out.sort_values(["prefix", "method_slug"]).drop_duplicates(["prefix", "method_slug"], keep="first")
    out["source_per_prefix_metrics_csv"] = str(per_prefix_metrics_csv)
    out["source_run_dir"] = str(per_prefix_metrics_csv.parent)
    return out.reset_index(drop=True)


def load_gt_envelopes(gt_envelope_csv: Path, quantile: float) -> pd.DataFrame:
    gt = pd.read_csv(gt_envelope_csv).copy()
    need = {"prefix", "alpha", "beta", "quantile", "fit_ok"}
    missing = need - set(gt.columns)
    if missing:
        raise ValueError(f"{gt_envelope_csv} is missing columns: {sorted(missing)}")

    gt = gt[gt["prefix"] != "__GLOBAL__"].copy()
    gt["quantile"] = pd.to_numeric(gt["quantile"], errors="coerce")
    gt["alpha"] = pd.to_numeric(gt["alpha"], errors="coerce")
    gt["beta"] = pd.to_numeric(gt["beta"], errors="coerce")
    gt = gt[(gt["fit_ok"].astype(bool)) & np.isclose(gt["quantile"], quantile)].copy()

    keep_cols = [c for c in ["prefix", "alpha", "beta", "quantile", "n_points", "reason"] if c in gt.columns]
    gt = gt[keep_cols].rename(columns={"alpha": "alpha_gt", "beta": "beta_gt", "n_points": "gt_n_points", "reason": "gt_fit_reason"})
    return gt.reset_index(drop=True)


def load_gt_points_optional(gt_points_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    if gt_points_csv is None:
        return None
    pts = pd.read_csv(gt_points_csv).copy()
    need = {"prefix", "ssim", "iou_gt"}
    missing = need - set(pts.columns)
    if missing:
        raise ValueError(f"{gt_points_csv} is missing columns: {sorted(missing)}")
    pts["ssim"] = pd.to_numeric(pts["ssim"], errors="coerce")
    pts["iou_gt"] = pd.to_numeric(pts["iou_gt"], errors="coerce")
    return pts.dropna(subset=["prefix", "ssim", "iou_gt"]).copy()


def compute_pointwise_fit_metrics(points_df: pd.DataFrame, alpha: float, beta: float, target_quantile: Optional[float]) -> Dict[str, float]:
    pred = alpha * points_df["ssim"].to_numpy(dtype=float) + beta
    y = points_df["iou_gt"].to_numpy(dtype=float)
    err = pred - y
    below_rate = float(np.mean(y < pred))
    out = {
        "point_line_mae": float(np.mean(np.abs(err))),
        "point_line_rmse": float(np.sqrt(np.mean(err ** 2))),
        "point_signed_bias": float(np.mean(err)),
        "below_line_rate": below_rate,
        "above_line_rate": float(np.mean(y >= pred)),
    }
    if target_quantile is not None:
        out["abs_below_rate_vs_quantile"] = float(abs(below_rate - target_quantile))
    return out


def compare_one_method_against_gt(
    method_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    gt_points_df: Optional[pd.DataFrame],
    target_quantile: float,
    ssim_min: float,
    ssim_max: float,
) -> pd.DataFrame:
    merged = method_df.merge(gt_df, on="prefix", how="inner")
    rows: List[Dict[str, Any]] = []

    for _, r in merged.iterrows():
        alpha_m = float(r["alpha"])
        beta_m = float(r["beta"])
        alpha_gt = float(r["alpha_gt"])
        beta_gt = float(r["beta_gt"])

        row: Dict[str, Any] = {
            "prefix": r["prefix"],
            "method_slug": r["method_slug"],
            "method_display_name": r["method_display_name"],
            "method_type": r["method_type"],
            "alpha_method": alpha_m,
            "beta_method": beta_m,
            "alpha_gt": alpha_gt,
            "beta_gt": beta_gt,
            "abs_alpha_diff": float(abs(alpha_m - alpha_gt)),
            "abs_beta_diff": float(abs(beta_m - beta_gt)),
            "target_gt_quantile": float(target_quantile),
            "source_per_prefix_metrics_csv": r["source_per_prefix_metrics_csv"],
            "source_run_dir": r["source_run_dir"],
        }
        if "gt_n_points" in r.index and pd.notna(r.get("gt_n_points", np.nan)):
            row["gt_n_points"] = int(r["gt_n_points"])
        row.update(line_distance_metrics(alpha_gt, beta_gt, alpha_m, beta_m, ssim_min=ssim_min, ssim_max=ssim_max))

        if gt_points_df is not None:
            pts = gt_points_df[gt_points_df["prefix"] == r["prefix"]]
            if len(pts) > 0:
                row.update(compute_pointwise_fit_metrics(pts, alpha_m, beta_m, target_quantile=target_quantile))
                row["gt_points_used"] = int(len(pts))
            else:
                row["gt_points_used"] = 0

        rows.append(row)

    return pd.DataFrame(rows)


def compute_similarity_score_from_df(df: pd.DataFrame) -> pd.Series:
    score = (df["line_mae"] + df["line_rmse"]) / 2.0
    if "abs_below_rate_vs_quantile" in df.columns:
        extra = df["abs_below_rate_vs_quantile"].astype(float)
        score = np.where(extra.notna(), (score + extra) / 2.0, score)
        return pd.Series(score, index=df.index, dtype=float)
    return pd.Series(score, index=df.index, dtype=float)


def summarize_method_similarity(per_prefix_similarity: pd.DataFrame) -> pd.DataFrame:
    if per_prefix_similarity.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    group_cols = ["source_run_dir", "source_per_prefix_metrics_csv", "method_slug", "method_display_name", "method_type"]
    for keys, sub in per_prefix_similarity.groupby(group_cols, dropna=False):
        src_dir, src_csv, slug, disp, mtype = keys
        row: Dict[str, Any] = {
            "source_run_dir": src_dir,
            "source_per_prefix_metrics_csv": src_csv,
            "method_slug": slug,
            "method_display_name": disp,
            "method_type": mtype,
            "num_prefixes_compared": int(sub["prefix"].nunique()),
            "mean_abs_alpha_diff": float(sub["abs_alpha_diff"].mean()),
            "mean_abs_beta_diff": float(sub["abs_beta_diff"].mean()),
            "mean_line_mae": float(sub["line_mae"].mean()),
            "mean_line_rmse": float(sub["line_rmse"].mean()),
            "mean_line_max_err": float(sub["line_max_err"].mean()),
            "mean_signed_line_bias": float(sub["signed_line_bias"].mean()),
            "median_line_mae": float(sub["line_mae"].median()),
            "median_line_rmse": float(sub["line_rmse"].median()),
        }
        ranking_terms = [row["mean_line_mae"], row["mean_line_rmse"]]
        if "point_line_mae" in sub.columns:
            row["mean_point_line_mae"] = float(sub["point_line_mae"].mean())
            row["mean_point_line_rmse"] = float(sub["point_line_rmse"].mean())
            row["mean_below_line_rate"] = float(sub["below_line_rate"].mean())
            if "abs_below_rate_vs_quantile" in sub.columns:
                row["mean_abs_below_rate_vs_quantile"] = float(sub["abs_below_rate_vs_quantile"].mean())
                ranking_terms.append(row["mean_abs_below_rate_vs_quantile"])
        row["similarity_score"] = float(np.mean(ranking_terms))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank_within_run"] = out.groupby("source_run_dir")["similarity_score"].rank(method="dense", ascending=True).astype(int)
    return out.sort_values(["source_run_dir", "rank_within_run", "method_slug"]).reset_index(drop=True)


def summarize_prefix_similarity(per_prefix_similarity: pd.DataFrame) -> pd.DataFrame:
    if per_prefix_similarity.empty:
        return pd.DataFrame()

    out = per_prefix_similarity.copy()
    out["similarity_score"] = compute_similarity_score_from_df(out)
    out["rank_within_prefix"] = out.groupby(["source_run_dir", "prefix"])["similarity_score"].rank(method="dense", ascending=True).astype(int)
    sort_cols = ["source_run_dir", "prefix", "rank_within_prefix", "method_slug"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def best_method_per_run(per_method_summary: pd.DataFrame) -> pd.DataFrame:
    if per_method_summary.empty:
        return pd.DataFrame()
    idx = per_method_summary.groupby("source_run_dir")["similarity_score"].idxmin()
    out = per_method_summary.loc[idx].copy()
    return out.sort_values(["source_run_dir", "similarity_score", "method_slug"]).reset_index(drop=True)


def best_method_per_prefix(per_prefix_summary: pd.DataFrame) -> pd.DataFrame:
    if per_prefix_summary.empty:
        return pd.DataFrame()
    idx = per_prefix_summary.groupby(["source_run_dir", "prefix"])["similarity_score"].idxmin()
    out = per_prefix_summary.loc[idx].copy()
    return out.sort_values(["source_run_dir", "prefix", "similarity_score", "method_slug"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GT envelope lines with method envelope alpha/beta stored in recursively found per_prefix_metrics.csv files."
    )
    parser.add_argument("--root-path", required=True, help="Root path under which per_prefix_metrics.csv files will be searched recursively")
    parser.add_argument("--gt-envelope-csv", required=True, help="per_prefix_gt_envelopes.csv produced by exp_generate_gt_envelope_from_tracker_v2.py")
    parser.add_argument("--save-path", required=True, help="Output directory")
    parser.add_argument("--gt-quantile", type=float, default=0.05, help="Which GT envelope quantile to use, e.g. 0.05")
    parser.add_argument("--gt-points-csv", default=None, help="Optional gt_envelope_points.csv for pointwise fit metrics")
    parser.add_argument("--method-type-filter", default=None, help="Optional comma-separated method_type filters, e.g. calibrated_alpha_beta,fixed_alpha_beta")
    parser.add_argument("--method-slug-filter", default=None, help="Optional comma-separated method_slug filters")
    parser.add_argument("--ssim-grid-min", type=float, default=0.4)
    parser.add_argument("--ssim-grid-max", type=float, default=1.0)
    return parser.parse_args()


def split_csv_set(text: Optional[str]) -> Optional[set[str]]:
    if text is None:
        return None
    items = {x.strip() for x in str(text).split(",") if x.strip()}
    return items or None


def main() -> None:
    args = parse_args()
    root_path = Path(args.root_path)
    gt_envelope_csv = Path(args.gt_envelope_csv)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    gt_df = load_gt_envelopes(gt_envelope_csv, quantile=float(args.gt_quantile))
    gt_points_df = load_gt_points_optional(Path(args.gt_points_csv)) if args.gt_points_csv else None

    metrics_csvs = find_per_prefix_metrics_csvs(root_path)
    if not metrics_csvs:
        raise FileNotFoundError(f"No per_prefix_metrics.csv found under: {root_path}")

    method_type_filter = split_csv_set(args.method_type_filter)
    method_slug_filter = split_csv_set(args.method_slug_filter)

    all_details: List[pd.DataFrame] = []
    discovered_runs: List[Dict[str, Any]] = []

    for csv_path in metrics_csvs:
        method_df = normalize_method_rows(csv_path)
        if method_type_filter is not None:
            method_df = method_df[method_df["method_type"].isin(method_type_filter)].copy()
        if method_slug_filter is not None:
            method_df = method_df[method_df["method_slug"].isin(method_slug_filter)].copy()
        if method_df.empty:
            continue

        discovered_runs.append({
            "per_prefix_metrics_csv": str(csv_path),
            "run_dir": str(csv_path.parent),
            "num_rows_after_filter": int(len(method_df)),
            "num_method_slugs": int(method_df["method_slug"].nunique()),
            "num_prefixes": int(method_df["prefix"].nunique()),
        })

        detail_df = compare_one_method_against_gt(
            method_df=method_df,
            gt_df=gt_df,
            gt_points_df=gt_points_df,
            target_quantile=float(args.gt_quantile),
            ssim_min=float(args.ssim_grid_min),
            ssim_max=float(args.ssim_grid_max),
        )
        if not detail_df.empty:
            all_details.append(detail_df)

    if not all_details:
        raise RuntimeError("No comparable rows remained after filtering / merging with GT envelopes.")

    per_prefix_similarity = pd.concat(all_details, ignore_index=True)
    per_method_summary = summarize_method_similarity(per_prefix_similarity)
    per_prefix_summary = summarize_prefix_similarity(per_prefix_similarity)
    best_run_df = best_method_per_run(per_method_summary)
    best_prefix_df = best_method_per_prefix(per_prefix_summary)

    per_prefix_csv = save_path / "per_prefix_gt_vs_method_envelope_similarity.csv"
    per_method_csv = save_path / "per_method_gt_vs_method_envelope_similarity_summary.csv"
    per_prefix_summary_csv = save_path / "per_prefix_gt_vs_method_envelope_similarity_summary.csv"
    best_run_csv = save_path / "best_method_per_run_by_gt_envelope_similarity.csv"
    best_prefix_csv = save_path / "best_method_per_prefix_by_gt_envelope_similarity.csv"

    per_prefix_similarity.to_csv(per_prefix_csv, index=False)
    per_method_summary.to_csv(per_method_csv, index=False)
    per_prefix_summary.to_csv(per_prefix_summary_csv, index=False)
    best_run_df.to_csv(best_run_csv, index=False)
    best_prefix_df.to_csv(best_prefix_csv, index=False)

    summary = {
        "root_path": str(root_path),
        "gt_envelope_csv": str(gt_envelope_csv),
        "gt_points_csv": str(args.gt_points_csv) if args.gt_points_csv else None,
        "save_path": str(save_path),
        "gt_quantile": float(args.gt_quantile),
        "ssim_grid_min": float(args.ssim_grid_min),
        "ssim_grid_max": float(args.ssim_grid_max),
        "num_gt_prefixes": int(gt_df["prefix"].nunique()),
        "num_per_prefix_metrics_csv_found": int(len(metrics_csvs)),
        "num_runs_used": int(len(discovered_runs)),
        "num_similarity_rows": int(len(per_prefix_similarity)),
        "num_method_summary_rows": int(len(per_method_summary)),
        "num_prefix_summary_rows": int(len(per_prefix_summary)),
        "num_best_run_rows": int(len(best_run_df)),
        "num_best_prefix_rows": int(len(best_prefix_df)),
        "method_type_filter": sorted(method_type_filter) if method_type_filter else None,
        "method_slug_filter": sorted(method_slug_filter) if method_slug_filter else None,
        "outputs": {
            "per_prefix_similarity_csv": str(per_prefix_csv),
            "per_method_summary_csv": str(per_method_csv),
            "per_prefix_summary_csv": str(per_prefix_summary_csv),
            "best_method_per_run_csv": str(best_run_csv),
            "best_method_per_prefix_csv": str(best_prefix_csv),
        },
        "discovered_runs": discovered_runs,
    }
    with open(save_path / "gt_vs_method_envelope_similarity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote outputs to: {save_path}")


if __name__ == "__main__":
    main()
