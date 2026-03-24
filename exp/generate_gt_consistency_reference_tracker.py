from pathlib import Path
import json
import math
import pandas as pd
import argparse
from typing import Dict, List, Tuple, Any, Optional

DEFAULT_BINARY_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    linear_sum_assignment = None


# =========================================================
# Utilities reused from original script
# =========================================================

def normalize_transition_key(path_like: str, remove_final_suffix: bool = True) -> str:
    key = Path(str(path_like).strip()).name
    if remove_final_suffix:
        lowered = key.lower()
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".txt"):
            if lowered.endswith(ext):
                key = key[: -len(ext)]
                break
    return key


def extract_prefix_and_frame(path_like: str) -> Tuple[str, int]:
    key = normalize_transition_key(path_like)
    if "_frame" not in key:
        return key, -1
    prefix, tail = key.split("_frame", 1)
    digits = []
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    frame_idx = int("".join(digits)) if digits else -1
    return prefix, frame_idx


def parse_yolo_gt_txt(txt_path: Path) -> List[List[float]]:
    """
    Return boxes in YOLO normalized xywh format:
    [cls_id, x_center, y_center, w, h]
    """
    boxes: List[List[float]] = []
    if not txt_path.exists():
        return boxes
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue
            boxes.append([cls_id, x, y, w, h])
    return boxes


def iou_xywh(box_a_xywh: List[float], box_b_xywh: List[float]) -> float:
    xa, ya, wa, ha = map(float, box_a_xywh[:4])
    xb, yb, wb, hb = map(float, box_b_xywh[:4])

    xa1, ya1 = xa - wa / 2.0, ya - ha / 2.0
    xa2, ya2 = xa + wa / 2.0, ya + ha / 2.0
    xb1, yb1 = xb - wb / 2.0, yb - hb / 2.0
    xb2, yb2 = xb + wb / 2.0, yb + hb / 2.0

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def add_binary_threshold_columns(row: Dict[str, Any], score: float, thresholds: List[float]) -> None:
    for thr in thresholds:
        thr_tag = str(thr).replace(".", "p")
        row[f"gt_is_consistent_binary_t{thr_tag}"] = int(score >= thr)


# =========================================================
# Lightweight tracker-specific utilities
# =========================================================

def center_distance_xywh(box_a_xywh: List[float], box_b_xywh: List[float]) -> float:
    xa, ya, _, _ = map(float, box_a_xywh[:4])
    xb, yb, _, _ = map(float, box_b_xywh[:4])
    dx = xa - xb
    dy = ya - yb
    return math.sqrt(dx * dx + dy * dy)


def scale_change_xywh(box_a_xywh: List[float], box_b_xywh: List[float]) -> float:
    _, _, wa, ha = map(float, box_a_xywh[:4])
    _, _, wb, hb = map(float, box_b_xywh[:4])

    area_a = max(1e-8, wa * ha)
    area_b = max(1e-8, wb * hb)
    return abs(math.log(area_b / area_a))


def aspect_ratio_change_xywh(box_a_xywh: List[float], box_b_xywh: List[float]) -> float:
    _, _, wa, ha = map(float, box_a_xywh[:4])
    _, _, wb, hb = map(float, box_b_xywh[:4])

    ar_a = max(1e-8, wa) / max(1e-8, ha)
    ar_b = max(1e-8, wb) / max(1e-8, hb)
    return abs(math.log(ar_b / ar_a))


def pair_cost_components(
    box_t: List[float],
    box_t1: List[float],
    max_center_dist: float = 0.12,
    max_scale_change: float = 1.2,
    max_aspect_change: float = 1.0,
    w_iou: float = 0.50,
    w_center: float = 0.30,
    w_scale: float = 0.15,
    w_aspect: float = 0.05,
) -> Optional[Dict[str, float]]:
    """
    Return None if pair is implausible (gated out),
    else return cost components.

    All coordinates are YOLO-normalized, so center distance is already normalized.
    """
    iou = iou_xywh(box_t, box_t1)
    center_dist = center_distance_xywh(box_t, box_t1)
    scale_change = scale_change_xywh(box_t, box_t1)
    aspect_change = aspect_ratio_change_xywh(box_t, box_t1)

    # soft tracker but with simple gating to prevent absurd matches
    if center_dist > max_center_dist:
        return None
    if scale_change > max_scale_change:
        return None
    if aspect_change > max_aspect_change:
        return None

    cost_iou = 1.0 - iou
    cost_center = min(1.0, center_dist / max_center_dist)
    cost_scale = min(1.0, scale_change / max_scale_change)
    cost_aspect = min(1.0, aspect_change / max_aspect_change)

    total_cost = (
        w_iou * cost_iou
        + w_center * cost_center
        + w_scale * cost_scale
        + w_aspect * cost_aspect
    )

    return {
        "iou": float(iou),
        "center_dist": float(center_dist),
        "scale_change": float(scale_change),
        "aspect_change": float(aspect_change),
        "cost": float(total_cost),
    }


def _hungarian_match_lightweight_tracker(
    gt_t: List[List[float]],
    gt_t1: List[List[float]],
    max_center_dist: float = 0.12,
    max_scale_change: float = 1.2,
    max_aspect_change: float = 1.0,
    max_total_cost: float = 0.85,
    w_iou: float = 0.50,
    w_center: float = 0.30,
    w_scale: float = 0.15,
    w_aspect: float = 0.05,
):
    """
    Match GT objects between adjacent frames with:
    - class-aware gating
    - center-distance-aware gating
    - scale-aware gating
    - IoU only as a cost term

    Returns list of matches:
      [(cost, iou, center_dist, scale_change, aspect_change, i, j), ...]
    """
    n_t = len(gt_t)
    n_t1 = len(gt_t1)
    if n_t == 0 or n_t1 == 0:
        return []

    pair_info = {}
    large_cost = 1e6

    if SCIPY_AVAILABLE:
        import numpy as np
        size = max(n_t, n_t1)
        cost_mat = np.full((size, size), fill_value=large_cost, dtype=float)

        for i, b_t in enumerate(gt_t):
            cls_t = int(b_t[0])
            for j, b_t1 in enumerate(gt_t1):
                cls_t1 = int(b_t1[0])
                if cls_t != cls_t1:
                    continue

                comp = pair_cost_components(
                    box_t=b_t[1:5],
                    box_t1=b_t1[1:5],
                    max_center_dist=max_center_dist,
                    max_scale_change=max_scale_change,
                    max_aspect_change=max_aspect_change,
                    w_iou=w_iou,
                    w_center=w_center,
                    w_scale=w_scale,
                    w_aspect=w_aspect,
                )
                if comp is None:
                    continue
                if comp["cost"] > max_total_cost:
                    continue

                cost_mat[i, j] = comp["cost"]
                pair_info[(i, j)] = comp

        rows, cols = linear_sum_assignment(cost_mat)
        matches = []
        for i, j in zip(rows.tolist(), cols.tolist()):
            if i >= n_t or j >= n_t1:
                continue
            if (i, j) not in pair_info:
                continue
            comp = pair_info[(i, j)]
            matches.append((
                float(comp["cost"]),
                float(comp["iou"]),
                float(comp["center_dist"]),
                float(comp["scale_change"]),
                float(comp["aspect_change"]),
                i,
                j,
            ))

        matches.sort(key=lambda x: x[0])
        return matches

    # greedy fallback
    candidates = []
    for i, b_t in enumerate(gt_t):
        cls_t = int(b_t[0])
        for j, b_t1 in enumerate(gt_t1):
            cls_t1 = int(b_t1[0])
            if cls_t != cls_t1:
                continue

            comp = pair_cost_components(
                box_t=b_t[1:5],
                box_t1=b_t1[1:5],
                max_center_dist=max_center_dist,
                max_scale_change=max_scale_change,
                max_aspect_change=max_aspect_change,
                w_iou=w_iou,
                w_center=w_center,
                w_scale=w_scale,
                w_aspect=w_aspect,
            )
            if comp is None:
                continue
            if comp["cost"] > max_total_cost:
                continue

            candidates.append((
                float(comp["cost"]),
                float(comp["iou"]),
                float(comp["center_dist"]),
                float(comp["scale_change"]),
                float(comp["aspect_change"]),
                i,
                j,
            ))

    candidates.sort(key=lambda x: x[0])
    used_t, used_t1, matches = set(), set(), []
    for item in candidates:
        _, _, _, _, _, i, j = item
        if i in used_t or j in used_t1:
            continue
        used_t.add(i)
        used_t1.add(j)
        matches.append(item)
    return matches


def gt_match_pairs_lightweight_tracker(
    gt_t: List[List[float]],
    gt_t1: List[List[float]],
    frame_consistency_mode: str = "f1_like",
    max_center_dist: float = 0.12,
    max_scale_change: float = 1.2,
    max_aspect_change: float = 1.0,
    max_total_cost: float = 0.85,
    w_iou: float = 0.50,
    w_center: float = 0.30,
    w_scale: float = 0.15,
    w_aspect: float = 0.05,
):
    """
    Replacement for gt_match_pairs(...)

    Notes:
    - no hard IoU threshold
    - uses Hungarian over multi-factor cost
    - still outputs legacy-compatible fields
    """
    n_t = len(gt_t)
    n_t1 = len(gt_t1)
    empty_empty = (n_t == 0 and n_t1 == 0)

    raw_matches = _hungarian_match_lightweight_tracker(
        gt_t=gt_t,
        gt_t1=gt_t1,
        max_center_dist=max_center_dist,
        max_scale_change=max_scale_change,
        max_aspect_change=max_aspect_change,
        max_total_cost=max_total_cost,
        w_iou=w_iou,
        w_center=w_center,
        w_scale=w_scale,
        w_aspect=w_aspect,
    )

    match_pairs = []
    matched_ious = []
    matched_costs = []
    matched_center_dists = []
    matched_scale_changes = []
    matched_aspect_changes = []

    for cost, iou, center_dist, scale_change, aspect_change, i, j in raw_matches:
        matched_ious.append(float(iou))
        matched_costs.append(float(cost))
        matched_center_dists.append(float(center_dist))
        matched_scale_changes.append(float(scale_change))
        matched_aspect_changes.append(float(aspect_change))
        match_pairs.append({
            "box_t_idx": int(i),
            "box_t1_idx": int(j),
            "cost": float(cost),
            "iou": float(iou),
            "center_dist": float(center_dist),
            "scale_change": float(scale_change),
            "aspect_change": float(aspect_change),
            "class_id": int(gt_t[i][0]),
        })

    matched_count = len(match_pairs)

    # legacy max-normalized coverage
    denom_max = max(n_t, n_t1, 1)
    match_coverage = float(matched_count / denom_max)

    mean_match_iou = float(sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0
    mean_match_cost = float(sum(matched_costs) / len(matched_costs)) if matched_costs else 0.0
    mean_center_dist = float(sum(matched_center_dists) / len(matched_center_dists)) if matched_center_dists else 0.0
    mean_scale_change = float(sum(matched_scale_changes) / len(matched_scale_changes)) if matched_scale_changes else 0.0
    mean_aspect_change = float(sum(matched_aspect_changes) / len(matched_aspect_changes)) if matched_aspect_changes else 0.0

    # New GT consistency:
    # use symmetric F1-like coverage by default
    if empty_empty:
        gt_frame_consistency = 1.0
    elif frame_consistency_mode == "max_norm":
        gt_frame_consistency = float(matched_count / denom_max)
    else:
        denom_sym = max(n_t + n_t1, 1)
        gt_frame_consistency = float((2.0 * matched_count) / denom_sym)

    return {
        "match_pairs": match_pairs,
        "matched_count": int(matched_count),
        "match_coverage": float(match_coverage),
        "mean_match_iou": float(mean_match_iou),
        "mean_match_cost": float(mean_match_cost),
        "mean_center_dist": float(mean_center_dist),
        "mean_scale_change": float(mean_scale_change),
        "mean_aspect_change": float(mean_aspect_change),
        "gt_frame_consistency": float(gt_frame_consistency),
        "empty_empty_transition": bool(empty_empty),
        "matching_algorithm": "hungarian_lightweight_tracker" if SCIPY_AVAILABLE else "greedy_lightweight_tracker",
    }


# =========================================================
# Main generation logic
# =========================================================

def generate_gt_consistency_reference_for_prefix(
    txt_files: List[Path],
    output_dir: Path,
    prefix: str,
    gt_consistency_threshold: float = 0.5,
    include_terminal: bool = False,
    binary_thresholds: List[float] = None,
    frame_consistency_mode: str = "f1_like",
    max_center_dist: float = 0.12,
    max_scale_change: float = 1.2,
    max_aspect_change: float = 1.0,
    max_total_cost: float = 0.85,
    w_iou: float = 0.50,
    w_center: float = 0.30,
    w_scale: float = 0.15,
    w_aspect: float = 0.05,
) -> Dict[str, Any]:
    if binary_thresholds is None:
        binary_thresholds = list(DEFAULT_BINARY_THRESHOLDS)

    if not txt_files:
        raise ValueError(f"No GT txt files for prefix: {prefix}")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_csv_path = output_dir / "gt_transition_reference.csv"
    object_json_path = output_dir / "gt_object_matches.json"
    summary_json_path = output_dir / "gt_summary.json"

    sorted_txts = sorted(txt_files, key=lambda p: extract_prefix_and_frame(p)[1])

    frame_rows: List[Dict[str, Any]] = []
    object_results: Dict[str, Dict[str, Any]] = {}
    duplicate_transition_keys = []

    for i in range(len(sorted_txts) - 1):
        txt_t = sorted_txts[i]
        txt_t1 = sorted_txts[i + 1]

        prefix_t, frame_t = extract_prefix_and_frame(txt_t)
        prefix_t1, frame_t1 = extract_prefix_and_frame(txt_t1)
        key_t = normalize_transition_key(txt_t)
        key_t1 = normalize_transition_key(txt_t1)

        gt_t = parse_yolo_gt_txt(txt_t)
        gt_t1 = parse_yolo_gt_txt(txt_t1)
        valid_transition = bool(prefix_t == prefix_t1 == prefix and frame_t >= 0 and frame_t1 > frame_t)

        if valid_transition:
            match_info = gt_match_pairs_lightweight_tracker(
                gt_t=gt_t,
                gt_t1=gt_t1,
                frame_consistency_mode=frame_consistency_mode,
                max_center_dist=max_center_dist,
                max_scale_change=max_scale_change,
                max_aspect_change=max_aspect_change,
                max_total_cost=max_total_cost,
                w_iou=w_iou,
                w_center=w_center,
                w_scale=w_scale,
                w_aspect=w_aspect,
            )
        else:
            match_info = {
                "match_pairs": [],
                "matched_count": 0,
                "match_coverage": 0.0,
                "mean_match_iou": 0.0,
                "mean_match_cost": 0.0,
                "mean_center_dist": 0.0,
                "mean_scale_change": 0.0,
                "mean_aspect_change": 0.0,
                "gt_frame_consistency": 0.0,
                "empty_empty_transition": False,
                "matching_algorithm": "hungarian_lightweight_tracker" if SCIPY_AVAILABLE else "greedy_lightweight_tracker",
            }

        legacy_binary = int(match_info["gt_frame_consistency"] >= gt_consistency_threshold)
        eval_excluded = bool(match_info["empty_empty_transition"])

        row = {
            "transition_key": key_t,
            "prefix": prefix,
            "transition_from": key_t,
            "transition_to": key_t1,
            "transition_from_path": str(txt_t),
            "transition_to_path": str(txt_t1),
            "frame_idx": int(frame_t),
            "next_frame_idx": int(frame_t1),
            "gt_num_t": int(len(gt_t)),
            "gt_num_t1": int(len(gt_t1)),
            "gt_num_matches": int(match_info["matched_count"]),
            "gt_mean_match_quality": float(match_info["mean_match_iou"]),  # legacy alias kept
            "gt_mean_match_iou": float(match_info["mean_match_iou"]),
            "gt_mean_match_cost": float(match_info["mean_match_cost"]),
            "gt_mean_center_dist": float(match_info["mean_center_dist"]),
            "gt_mean_scale_change": float(match_info["mean_scale_change"]),
            "gt_mean_aspect_change": float(match_info["mean_aspect_change"]),
            "gt_match_coverage": float(match_info["match_coverage"]),
            "gt_frame_consistency": float(match_info["gt_frame_consistency"]),
            "gt_match_coverage_main_eval": float(match_info["match_coverage"]),
"gt_mean_match_iou_main_eval": float(match_info["mean_match_iou"]),
            "gt_is_consistent_binary": int(legacy_binary),
            "gt_is_empty_empty": bool(match_info["empty_empty_transition"]),
            "exclude_from_main_eval": bool(eval_excluded),
            "matching_algorithm": match_info["matching_algorithm"],
            "is_terminal": False,
            "valid_transition": bool(valid_transition),
            "original_index": int(i),
        }
        add_binary_threshold_columns(row, float(match_info["gt_frame_consistency"]), binary_thresholds)
        frame_rows.append(row)

        if key_t in object_results:
            duplicate_transition_keys.append(key_t)
        object_results[key_t] = {
            **row,
            "matches": match_info["match_pairs"],
        }

    if include_terminal and sorted_txts:
        last_txt = sorted_txts[-1]
        prefix_last, frame_last = extract_prefix_and_frame(last_txt)
        key_last = normalize_transition_key(last_txt)
        gt_last = parse_yolo_gt_txt(last_txt)

        terminal_row = {
            "transition_key": key_last,
            "prefix": prefix_last,
            "transition_from": key_last,
            "transition_to": None,
            "transition_from_path": str(last_txt),
            "transition_to_path": None,
            "frame_idx": int(frame_last),
            "next_frame_idx": None,
            "gt_num_t": int(len(gt_last)),
            "gt_num_t1": 0,
            "gt_num_matches": 0,
            "gt_mean_match_quality": 0.0,
            "gt_mean_match_iou": 0.0,
            "gt_mean_match_cost": 0.0,
            "gt_mean_center_dist": 0.0,
            "gt_mean_scale_change": 0.0,
            "gt_mean_aspect_change": 0.0,
            "gt_match_coverage": 0.0,
            "gt_frame_consistency": 0.0,
            "gt_is_consistent_binary": 0,
            "gt_is_empty_empty": False,
            "exclude_from_main_eval": True,
            "matching_algorithm": "hungarian_lightweight_tracker" if SCIPY_AVAILABLE else "greedy_lightweight_tracker",
            "is_terminal": True,
            "valid_transition": False,
            "original_index": int(len(sorted_txts) - 1),
        }
        add_binary_threshold_columns(terminal_row, 0.0, binary_thresholds)
        frame_rows.append(terminal_row)
        object_results[key_last] = {**terminal_row, "matches": []}

    df = pd.DataFrame(frame_rows)
    df.to_csv(frame_csv_path, index=False)

    with open(object_json_path, "w", encoding="utf-8") as f:
        json.dump(object_results, f, indent=2, ensure_ascii=False)

    non_terminal = df.loc[~df["is_terminal"]].copy() if not df.empty else pd.DataFrame()
    main_eval_df = non_terminal.loc[~non_terminal["exclude_from_main_eval"]].copy() if not non_terminal.empty else pd.DataFrame()

    summary = {
        "prefix": prefix,
        "gt_consistency_threshold_legacy": float(gt_consistency_threshold),
        "binary_thresholds": [float(x) for x in binary_thresholds],
        "include_terminal": bool(include_terminal),
        "matching_algorithm": "hungarian_lightweight_tracker" if SCIPY_AVAILABLE else "greedy_lightweight_tracker",
        "frame_consistency_mode": frame_consistency_mode,
        "tracker_params": {
            "max_center_dist": float(max_center_dist),
            "max_scale_change": float(max_scale_change),
            "max_aspect_change": float(max_aspect_change),
            "max_total_cost": float(max_total_cost),
            "w_iou": float(w_iou),
            "w_center": float(w_center),
            "w_scale": float(w_scale),
            "w_aspect": float(w_aspect),
        },
        "num_txt_files": int(len(sorted_txts)),
        "num_transitions_total": int(len(frame_rows)),
        "num_non_terminal_transitions": int((~df["is_terminal"]).sum()) if not df.empty else 0,
        "num_valid_transitions": int(df["valid_transition"].sum()) if not df.empty else 0,
        "num_empty_empty": int(non_terminal["gt_is_empty_empty"].sum()) if not non_terminal.empty else 0,
        "num_excluded_from_main_eval": int(non_terminal["exclude_from_main_eval"].sum()) if not non_terminal.empty else 0,
        "num_zero_match": int((non_terminal["gt_num_matches"] == 0).sum()) if not non_terminal.empty else 0,
        "mean_gt_frame_consistency_all_non_terminal": float(non_terminal["gt_frame_consistency"].mean()) if not non_terminal.empty else None,
        "mean_gt_frame_consistency_main_eval": float(main_eval_df["gt_frame_consistency"].mean()) if not main_eval_df.empty else None,
        "mean_gt_match_coverage_main_eval": float(main_eval_df["gt_match_coverage"].mean()) if not main_eval_df.empty else None,
        "mean_gt_mean_match_iou_main_eval": float(main_eval_df["gt_mean_match_iou"].mean()) if not main_eval_df.empty else None,
        "mean_gt_mean_match_cost_main_eval": float(main_eval_df["gt_mean_match_cost"].mean()) if not main_eval_df.empty else None,
        "duplicate_transition_keys": sorted(set(duplicate_transition_keys)),
        "frame_csv": str(frame_csv_path),
        "object_json": str(object_json_path),
    }

    for thr in binary_thresholds:
        thr_tag = str(thr).replace(".", "p")
        col = f"gt_is_consistent_binary_t{thr_tag}"
        summary[f"num_positive_binary_t{thr_tag}"] = int(main_eval_df[col].sum()) if not main_eval_df.empty else 0

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "prefix": prefix,
        "output_dir": str(output_dir),
        "frame_csv": str(frame_csv_path),
        "object_json": str(object_json_path),
        "summary_json": str(summary_json_path),
        **summary,
    }


def group_gt_for_prefix(
    gt_dir: str,
    save_path: str,
    gt_consistency_threshold: float = 0.5,
    include_terminal: bool = False,
    binary_thresholds: List[float] = None,
    frame_consistency_mode: str = "f1_like",
    max_center_dist: float = 0.12,
    max_scale_change: float = 1.2,
    max_aspect_change: float = 1.0,
    max_total_cost: float = 0.85,
    w_iou: float = 0.50,
    w_center: float = 0.30,
    w_scale: float = 0.15,
    w_aspect: float = 0.05,
) -> Dict[str, Any]:
    if binary_thresholds is None:
        binary_thresholds = list(DEFAULT_BINARY_THRESHOLDS)

    gt_dir_path = Path(gt_dir)
    if not gt_dir_path.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    txt_files = sorted(gt_dir_path.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No GT txt files found in: {gt_dir}")

    grouped: Dict[str, List[Path]] = {}
    for txt_path in txt_files:
        prefix, frame_idx = extract_prefix_and_frame(txt_path)
        if frame_idx < 0:
            continue
        grouped.setdefault(prefix, []).append(txt_path)

    save_root = Path(save_path)
    save_root.mkdir(parents=True, exist_ok=True)

    prefixes = sorted(grouped.keys())
    group_summaries = []
    for idx, prefix in enumerate(prefixes):
        subdir = save_root / f"{idx:02d}_{prefix}"
        result = generate_gt_consistency_reference_for_prefix(
            txt_files=grouped[prefix],
            output_dir=subdir,
            prefix=prefix,
            gt_consistency_threshold=gt_consistency_threshold,
            include_terminal=include_terminal,
            binary_thresholds=binary_thresholds,
            frame_consistency_mode=frame_consistency_mode,
            max_center_dist=max_center_dist,
            max_scale_change=max_scale_change,
            max_aspect_change=max_aspect_change,
            max_total_cost=max_total_cost,
            w_iou=w_iou,
            w_center=w_center,
            w_scale=w_scale,
            w_aspect=w_aspect,
        )
        group_summaries.append(result)
        print(f"[OK] {prefix} -> {subdir}")

    index_summary = {
        "gt_dir": str(gt_dir_path),
        "save_path": str(save_root),
        "gt_consistency_threshold_legacy": float(gt_consistency_threshold),
        "binary_thresholds": [float(x) for x in binary_thresholds],
        "include_terminal": bool(include_terminal),
        "matching_algorithm": "hungarian_lightweight_tracker" if SCIPY_AVAILABLE else "greedy_lightweight_tracker",
        "frame_consistency_mode": frame_consistency_mode,
        "tracker_params": {
            "max_center_dist": float(max_center_dist),
            "max_scale_change": float(max_scale_change),
            "max_aspect_change": float(max_aspect_change),
            "max_total_cost": float(max_total_cost),
            "w_iou": float(w_iou),
            "w_center": float(w_center),
            "w_scale": float(w_scale),
            "w_aspect": float(w_aspect),
        },
        "num_prefixes": int(len(prefixes)),
        "prefixes": prefixes,
        "groups": group_summaries,
    }

    index_path = save_root / "index_summary.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_summary, f, indent=2, ensure_ascii=False)

    print(f"Saved root index summary to: {index_path}")
    return index_summary


def parse_binary_thresholds(arg: str) -> List[float]:
    values = []
    for token in str(arg).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("binary thresholds list is empty")
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate GT consistency references grouped by prefix using a lightweight GT tracker."
    )
    parser.add_argument("--gt-dir", required=True, help="Directory containing GT txt files")
    parser.add_argument("--save-path", required=True, help="Root output directory")
    parser.add_argument("--gt-consistency-threshold", type=float, default=0.5)
    parser.add_argument("--binary-thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--include-terminal", action="store_true")

    parser.add_argument("--frame-consistency-mode", type=str, default="f1_like", choices=["f1_like", "max_norm"])
    parser.add_argument("--max-center-dist", type=float, default=0.12)
    parser.add_argument("--max-scale-change", type=float, default=1.2)
    parser.add_argument("--max-aspect-change", type=float, default=1.0)
    parser.add_argument("--max-total-cost", type=float, default=0.85)

    parser.add_argument("--w-iou", type=float, default=0.50)
    parser.add_argument("--w-center", type=float, default=0.30)
    parser.add_argument("--w-scale", type=float, default=0.15)
    parser.add_argument("--w-aspect", type=float, default=0.05)

    args = parser.parse_args()

    group_gt_for_prefix(
        gt_dir=args.gt_dir,
        save_path=args.save_path,
        gt_consistency_threshold=args.gt_consistency_threshold,
        include_terminal=args.include_terminal,
        binary_thresholds=parse_binary_thresholds(args.binary_thresholds),
        frame_consistency_mode=args.frame_consistency_mode,
        max_center_dist=args.max_center_dist,
        max_scale_change=args.max_scale_change,
        max_aspect_change=args.max_aspect_change,
        max_total_cost=args.max_total_cost,
        w_iou=args.w_iou,
        w_center=args.w_center,
        w_scale=args.w_scale,
        w_aspect=args.w_aspect,
    )