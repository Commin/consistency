import numpy as np
import os
import math

def get_conf(box):
    return float(box[5]) if len(box) > 5 else 1.0

def center_xywh(box):
    return float(box[1]), float(box[2]), float(box[3]), float(box[4])

def diag(box):
    _, _, w, h = center_xywh(box)
    return math.sqrt(w * w + h * h)

def center_distance(box_a, box_b):
    xa, ya, _, _ = center_xywh(box_a)
    xb, yb, _, _ = center_xywh(box_b)
    return math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)

def calculate_iou(box1, box2):
    """Calculates IoU for [cx, cy, w, h] format."""
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    inter_x1, inter_y1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
    inter_x2, inter_y2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = (box1[2] * box1[3]) + (box2[2] * box2[3]) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def iou_xywh(box_a_xywh, box_b_xywh):
    """
    IoU for YOLO-format normalized xywh boxes.

    Input:
        box_a_xywh, box_b_xywh:
            [x_center, y_center, width, height]
        or
            [x_center, y_center, width, height, confidence]

    The optional 5th confidence value is ignored.
    """
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

def apply_nms_to_preds(preds, iou_threshold=0.5, conf_threshold=0.0):
    """
    Apply class-wise NMS to predictions.

    Input format for each prediction:
        [class_id, x_center, y_center, width, height]
    or
        [class_id, x_center, y_center, width, height, confidence]

    Returns:
        filtered_preds: list of predictions after NMS
        kept_indices: indices of kept boxes in the original input list
    """
    import math

    if not preds:
        return [], []

    # confidence filtering first
    indexed_preds = [(idx, p) for idx, p in enumerate(preds) if get_conf(p) >= conf_threshold]
    if not indexed_preds:
        return [], []

    # group by class
    class_to_items = {}
    for idx, p in indexed_preds:
        cls_id = int(p[0])
        class_to_items.setdefault(cls_id, []).append((idx, p))

    kept = []

    for cls_id, items in class_to_items.items():
        # sort by confidence descending
        items = sorted(items, key=lambda x: get_conf(x[1]), reverse=True)

        while items:
            best_idx, best_box = items.pop(0)
            kept.append((best_idx, best_box))

            remaining = []
            for idx, box in items:
                iou = iou_xywh(best_box[1:5], box[1:5])
                if iou < iou_threshold:
                    remaining.append((idx, box))
            items = remaining

    # preserve descending confidence order is fine, but often stable original order is nicer
    kept = sorted(kept, key=lambda x: x[0])
    filtered_preds = [p for _, p in kept]
    kept_indices = [idx for idx, _ in kept]
    return filtered_preds, kept_indices


def calculate_object_level_matches(preds_t,
                                   preds_t_plus_1,
                                   apply_nms=True,
                                   nms_iou_threshold=0.5,
                                   nms_conf_threshold=0.0,
                                   min_iou=0.05,
                                   min_conf=0.0,
                                   distance_ratio=1.5,
                                   min_distance_px=100.0):
    """
    Compute object-level matched pairs and mean matched IoU quality.

    Steps:
        1. Optional confidence filtering + NMS on each frame
        2. Same-class candidate generation
        3. Spatial gating
        4. IoU gating
        5. Greedy one-to-one matching

    Returns:
        mean_match_quality: float
            Mean IoU over matched pairs.
            If no matched pairs exist, returns 0.0.

        match_pairs: list of dict
            Each dict contains:
                {
                    "box_t_idx": original index in preds_t,
                    "box_t+1_idx": original index in preds_t_plus_1,
                    "iou": float
                }
    """

    # Step 1: filter + optional NMS
    if apply_nms:
        preds_t_refined, kept_t = apply_nms_to_preds(
            preds_t, iou_threshold=nms_iou_threshold, conf_threshold=nms_conf_threshold
        )
        preds_t1_refined, kept_t1 = apply_nms_to_preds(
            preds_t_plus_1, iou_threshold=nms_iou_threshold, conf_threshold=nms_conf_threshold
        )
    else:
        preds_t_refined = [p for p in preds_t if get_conf(p) >= min_conf]
        preds_t1_refined = [p for p in preds_t_plus_1 if get_conf(p) >= min_conf]
        kept_t = [i for i, p in enumerate(preds_t) if get_conf(p) >= min_conf]
        kept_t1 = [i for i, p in enumerate(preds_t_plus_1) if get_conf(p) >= min_conf]

    if not preds_t_refined or not preds_t1_refined:
        return 0.0, []

    # Step 2-4: candidate generation with gating
    possible_matches = []
    for i, b_t in enumerate(preds_t_refined):
        cls_t = int(b_t[0])
        dist_thresh = max(float(min_distance_px), float(distance_ratio) * diag(b_t))

        for j, b_t1 in enumerate(preds_t1_refined):
            cls_t1 = int(b_t1[0])

            # same class
            if cls_t != cls_t1:
                continue

            # spatial gating
            if center_distance(b_t, b_t1) > dist_thresh:
                continue

            # IoU gating
            iou = iou_xywh(b_t[1:5], b_t1[1:5])
            if iou < float(min_iou):
                continue

            possible_matches.append((iou, i, j))

    if not possible_matches:
        return 0.0, []

    # Step 5: greedy one-to-one matching
    possible_matches.sort(key=lambda x: x[0], reverse=True)

    used_t = set()
    used_t1 = set()
    matched_ious = []
    match_pairs = []

    for iou, i, j in possible_matches:
        if i in used_t or j in used_t1:
            continue

        used_t.add(i)
        used_t1.add(j)
        matched_ious.append(float(iou))

        match_pairs.append({
            "box_t_idx": kept_t[i],          # original index in preds_t
            "box_t+1_idx": kept_t1[j],       # original index in preds_t_plus_1
            "iou": float(iou)
        })

    if not matched_ious:
        return 0.0, []

    mean_match_quality = sum(matched_ious) / len(matched_ious)
    return float(mean_match_quality), match_pairs


def calculate_frame_level_consistency(preds_t,
                                      preds_t_plus_1,
                                      apply_nms=True,
                                      nms_iou_threshold=0.5,
                                      nms_conf_threshold=0.0,
                                      min_iou=0.05,
                                      min_conf=0.0,
                                      distance_ratio=1.5,
                                      min_distance_px=100.0):
    """
    Compute frame-level temporal consistency:

        C_t = (sum matched IoUs + 1[N_t + N_t+1 = 0]) / max(N_t, N_t+1, 1)

    where:
        - N_t, N_t+1 are the numbers of refined detections after confidence filtering / NMS
        - matched IoUs come from object-level one-to-one matching

    Returns:
        C_t: float
            Frame-level consistency

        mean_match_quality: float
            Mean IoU over matched pairs

        match_pairs: list of dict
            Each dict contains:
                {
                    "box_t_idx": original index in preds_t,
                    "box_t+1_idx": original index in preds_t_plus_1,
                    "iou": float
                }
    """

    # refine counts must match the object-level pipeline
    if apply_nms:
        preds_t_refined, _ = apply_nms_to_preds(
            preds_t, iou_threshold=nms_iou_threshold, conf_threshold=nms_conf_threshold
        )
        preds_t1_refined, _ = apply_nms_to_preds(
            preds_t_plus_1, iou_threshold=nms_iou_threshold, conf_threshold=nms_conf_threshold
        )
    else:
        preds_t_refined = [p for p in preds_t if get_conf(p) >= min_conf]
        preds_t1_refined = [p for p in preds_t_plus_1 if get_conf(p) >= min_conf]

    N_t = len(preds_t_refined)
    N_t1 = len(preds_t1_refined)

    mean_match_quality, match_pairs = calculate_object_level_matches(
        preds_t,
        preds_t_plus_1,
        apply_nms=apply_nms,
        nms_iou_threshold=nms_iou_threshold,
        nms_conf_threshold=nms_conf_threshold,
        min_iou=min_iou,
        min_conf=min_conf,
        distance_ratio=distance_ratio,
        min_distance_px=min_distance_px,
    )

    # sum of matched IoUs
    matched_iou_sum = sum(p["iou"] for p in match_pairs)

    # unified frame-level formula
    vacancy_term = 1.0 if (N_t + N_t1 == 0) else 0.0
    C_t = (matched_iou_sum + vacancy_term) / max(N_t, N_t1, 1)

    return float(C_t), float(mean_match_quality), match_pairs


def parse_yolo_txt(filepath):
    """
    Parses a YOLO-format text file and returns a list of predictions.
    Format: [class_id, cx, cy, w, h, confidence]
    """
    preds = []
    if not os.path.exists(filepath):
        return preds
        
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # We only need [class_id, cx, cy, w, h], ignore confidence if present
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                if len(parts) >= 6:
                    confidence = float(parts[5])
                    preds.append([class_id, cx, cy, w, h, confidence])
                else:
                    preds.append([class_id, cx, cy, w, h])
    return preds

def evaluate_consistency_from_files(file_t, file_t_plus_1):
    """
    Reads two YOLO format text files and calculates their temporal consistency.
    """
    preds_t = parse_yolo_txt(file_t)
    preds_t1 = parse_yolo_txt(file_t_plus_1)
    
    return calculate_frame_level_consistency(preds_t, preds_t1)

def get_object_level_matches_from_files(file_t, file_t_plus_1):
    preds_t = parse_yolo_txt(file_t)
    preds_t1 = parse_yolo_txt(file_t_plus_1)
    
    return get_object_level_matches_from_files(preds_t, preds_t1)

def compare_boundary(r_score, ssim_score, alpha, beta):
    """
    r_score: Aggregate agreement score
    ssim_score: SSIM score
    alpha, beta: parameter for boundary
    """

    if r_score >= alpha * ssim_score + beta:
        return 1
    else:
        return 0

# --- Example Usage ---
if __name__ == "__main__":

    # Example usage of the original mock data
    frame_t_preds = [
        [0, 100, 100, 50, 50],  # Person A
        [1, 200, 200, 30, 80]   # Car A
    ]

    frame_t1_preds = [
        [0, 105, 102, 52, 48],  # Person A (moved slightly)
        [1, 250, 200, 30, 80],  # Car A (moved significantly)
        [0, 500, 500, 40, 40]   # New Person B
    ]

    consistency, details = calculate_object_level_matches(frame_t_preds, frame_t1_preds)

    C_t, mean_match_quality, match_pairs = calculate_frame_level_consistency(frame_t_preds, frame_t1_preds)

    print(f"Object-wide Consistency Score: {consistency:.4f}")
    for i, match in enumerate(details):
        print(f"Object {i} (Class {match['box_t_idx']}) Match IoU: {match['iou']:.4f}")
    
    print(f"Frame-wide Consistency Score: {C_t:.4f}")
    print(f"Frame-wide Mean Match Quality: {mean_match_quality:.4f}")
    
