import numpy as np
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import statsmodels.formula.api as smf
import argparse
import os
import glob
import cv2
import json
import consistency
import utils

def get_video_id(filename):
    base = os.path.basename(filename)
    return base.split("_frame")[0] if "_frame" in base else "default_video"

def parse_gt_txt(filepath):
    gts = []
    if not os.path.exists(filepath): return gts
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    gts.append([int(float(parts[0])), int(float(parts[1])), 
                                float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])
                except ValueError: pass
    return gts

def get_ssim(frame1_path, frame2_path, ssim_dict=None, gt_img_dir=None):
    if not frame1_path or not frame2_path: return 0.0
    key1 = utils.get_file_key(frame1_path)
    
    if ssim_dict and key1 in ssim_dict:
        return float(ssim_dict[key1])
    
    p1 = frame1_path if os.path.exists(frame1_path) else os.path.join(gt_img_dir, key1 + ".jpg")
    if not os.path.exists(p1): return 0.0
    
    img1, img2 = cv2.imread(p1), cv2.imread(frame2_path)
    if img1 is not None and img2 is not None:
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        return ssim(img1, img2, channel_axis=-1)
    return 0.0

def calibrate_video_specific(consistency_stem_dict, frames, preds_list, names, gt_img_dir, ssim_dict, quantile=0.05):
    """
    Match correct time-adjacent frame pairs (t -> t+1) within the same video, ensuring that:
    SSIM (t -> t+1)
    IoU matches (t -> t+1) 
    """
    dataset = []
    for t in range(len(names) - 1):
        stem_t = utils.get_file_key(names[t])
        stem_t1 = utils.get_file_key(names[t+1])
        
        # Extract SSIM for T -> T+1 (SSIM csv records itself and the next frame, so look in T)
        s_t = get_ssim(frames[t], frames[t+1], ssim_dict, gt_img_dir)
        
        # Extract IoU matches for T -> T+1 (JSON records itself and the previous frame, so look in T+1)
        if consistency_stem_dict is not None:
            if stem_t1 in consistency_stem_dict:
                # Analyze {"score": ..., "matches": [...]} structure
                frame_data = consistency_stem_dict[stem_t1]
                matches = frame_data.get("matches", [])
                for m in matches:
                    if "iou" in m:
                        dataset.append({'ssim': s_t, 'iou': m['iou']})
        else:
            # Fallback: if no JSON available, use lbl predictions for simple matching
            p_t, p_t1 = preds_list[t], preds_list[t+1]
            for i, box_t in enumerate(p_t):
                for j, box_t1 in enumerate(p_t1):
                    if int(box_t[0]) == int(box_t1[0]): # 同类
                        iou = consistency.calculate_iou(box_t[1:5], box_t1[1:5])
                        if iou > 0.0: dataset.append({'ssim': s_t, 'iou': iou})
    
    if len(dataset) < 5: return 0.0, 0.3
    df = pd.DataFrame(dataset)
    try:
        res = smf.quantreg('iou ~ ssim', df).fit(q=quantile)
        alpha = max(0.0, res.params['ssim'])
        beta = res.params['Intercept']
        return alpha, beta
    except: return 0.0, 0.3

def evaluate_gt_consistency(vid_name, frames, gt_list, gt_img_dir, ssim_dict, alpha, beta, tau_min):
    y_true, y_pred_dyn = [], []
    y_pred_fixed = {0.3: [], 0.5: []}
    
    for t in range(len(frames) - 1):
        s_t = get_ssim(frames[t], frames[t+1], ssim_dict, gt_img_dir)
        tau_dyn = max(tau_min, alpha * s_t + beta)

        for g_t in gt_list[t]:
            id_t = g_t[1]
            if id_t == -1: continue
            matched_g1 = [g for g in gt_list[t+1] if g[1] == id_t]
            if matched_g1 and g_t[0] == matched_g1[0][0]:
                iou = consistency.calculate_iou(g_t[2:6], matched_g1[0][2:6])
                y_true.append(1)
                y_pred_dyn.append(1 if iou >= tau_dyn else 0)
                for thr in y_pred_fixed.keys():
                    y_pred_fixed[thr].append(1 if iou >= thr else 0)
    
    if not y_true: return None
    return {
        "video_id": vid_name, "alpha": alpha, "beta": beta,
        "recall_dyn": sum(y_pred_dyn) / len(y_true),
        "recall_0.3": sum(y_pred_fixed[0.3]) / len(y_true),
        "recall_0.5": sum(y_pred_fixed[0.5]) / len(y_true)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--lbl-dir")
    parser.add_argument("--ssim-csv", required=True)
    parser.add_argument("--csv-output", default="gt_video_metrics.csv")
    parser.add_argument("--consistency-json")
    parser.add_argument("--quantile", type=float, default=0.05)
    parser.add_argument("--tau-min", type=float, default=0.2)
    args = parser.parse_args()

    has_input = args.lbl_dir is not None or args.consistency_json is not None
    if not has_input:
        raise ValueError("Either --lbl-dir or --consistency-json must be provided.")

    gt_image_dir = os.path.join(args.gt_dir, "images")
    gt_label_dir = os.path.join(args.gt_dir, "labels")

    # --- 1. Read and clean SSIM dictionary keys to ensure they are in stem format for consistent matching ---
    with open(args.ssim_csv, 'r') as f:
        raw_ssim = json.load(f)
        ssim_dict = {utils.get_file_key(k): v for k, v in raw_ssim.items()}
    ssim_keys = set(ssim_dict.keys())

    # --- 2. Preprocess inputs to create a consistency stem dictionary (if JSON provided) and determine the valid keys that have both SSIM and either JSON records or lbl predictions, ensuring consistent matching based on stem keys ---
    consistency_stem_dict = None
    valid_keys_set = set()

    if args.consistency_json and os.path.exists(args.consistency_json):
        print("Preprocessing using consistency-json...")
        with open(args.consistency_json, 'r') as f:
            raw_consistency = json.load(f)
            consistency_stem_dict = {utils.get_file_key(k): v for k, v in raw_consistency.items()}
            
        same_keys_list, _, _ = utils.compare_file_key(gt_image_dir, args.consistency_json)
        # If using JSON, we require that the keys must exist in the JSON and have corresponding SSIM records to be considered valid for calibration, ensuring that we are working with frames that have both types of information available for accurate matching and analysis.
        valid_keys_set = set(same_keys_list).intersection(ssim_keys)
    else:
        print("Preprocessing using lbl-dir...")
        lbl_files = glob.glob(os.path.join(args.lbl_dir, "*.txt"))
        lbl_keys = {utils.get_file_key(f) for f in lbl_files}
        valid_keys_set = lbl_keys.intersection(ssim_keys)

    if not valid_keys_set:
        print("Error: No overlapping keys found between inputs and SSIM CSV.")
        exit(1)

    # --- 3. Group frames by video based on valid keys, ensuring that we are organizing and processing frames in the context of their respective videos, which is essential for performing accurate time-adjacent frame matching and consistency analysis within each video sequence. The grouping will be done based on the video ID extracted from the stem keys, and we will maintain the order of frames within each video to ensure correct temporal alignment for the subsequent calibration and evaluation steps. ---
    video_groups = {}
    
    # Sort the valid keys to ensure that frames are processed in temporal order within each video, which is crucial for accurate calibration and consistency evaluation. The sorting will be based on the frame number extracted from the stem keys, ensuring that we are correctly matching adjacent frames (t -> t+1) for each video sequence.
    for stem in sorted(list(valid_keys_set)):
        vid = get_video_id(stem)
        video_groups.setdefault(vid, {"f": [], "g": [], "l": [], "n": []})
        
        txt_name = stem + ".txt"
        img_name = stem + ".jpg"
        
        video_groups[vid]["n"].append(txt_name)
        video_groups[vid]["g"].append(parse_gt_txt(os.path.join(gt_label_dir, txt_name)))
        
        if args.lbl_dir:
            video_groups[vid]["l"].append(consistency.parse_yolo_txt(os.path.join(args.lbl_dir, txt_name)))
        else:
            video_groups[vid]["l"].append([])
            
        video_groups[vid]["f"].append(os.path.join(gt_image_dir, img_name))

    print(f"Total Videos Grouped: {len(video_groups)}")

    # --- 4. Process each video group to perform calibration and evaluation, ensuring that we are analyzing the consistency of frame pairs within the context of their respective videos. For each video, we will perform the following steps:
    results = []
    for vid, data in video_groups.items():
        if len(data["f"]) < 2: continue
        
        a_vid, b_vid = calibrate_video_specific(
            vid, consistency_stem_dict, data["f"], data["l"], data["n"], 
            gt_image_dir, ssim_dict, valid_keys_set, args.quantile
        )
        print(f"Video {vid}: Alpha={a_vid:.4f}, Beta={b_vid:.4f}")
        
        metrics = evaluate_gt_consistency(
            vid, data["f"], data["l"], data["g"], data["n"], 
            gt_image_dir, ssim_dict, a_vid, b_vid, args.tau_min
        )
        if metrics: results.append(metrics)

    if results:
        pd.DataFrame(results).to_csv(args.csv_output, index=False)
        print(f"\nSuccess: Metrics saved to '{args.csv_output}'. Evaluated {len(results)} videos.")