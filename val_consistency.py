import argparse
import os
from pathlib import Path
from ultralytics import YOLO

# Import local utility functions from the existing pipeline
from construct import get_image_paths_from_yaml
from consistency import parse_yolo_txt, calculate_frame_level_consistency, compare_boundary

def run_all_val_consistency(model_path, data_yaml, project_root, alpha, beta):
    """
    Run YOLO validation on all data specified in data_yaml, 
    and then compute & save frame-level and object-level consistency.
    """
    print(f"\n--- 1. Running YOLO Validation ---")
    stage_name = model_path.split("/weights/")[0].split("/")[-1]
    print(f"Model: {model_path}")
    print(f"Stage: {stage_name}")
    print(f"Data:  {data_yaml}")
    
    # 1. set output directory all_val_xxxx 
    # extract identifier from yaml filename, e.g., data0_5.yaml -> data0_5
    yaml_name = Path(data_yaml).stem
    val_dir_name = f"all_val_{yaml_name}"
    
    output_project = str(Path(project_root).resolve())
    
    # validate model
    target_split = 'val' # default for consistency evaluation, can also be set to test/train
    model = YOLO(model_path)
    
    metrics = model.val(
        data=data_yaml,
        split=target_split, 
        save_txt=True,
        save_conf=True,
        project=output_project,
        name=val_dir_name,
        device='0'
    )
    
    save_dir = Path(metrics.save_dir)
    labels_dir = save_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✅ Validation complete. Results saved at: {save_dir}")
    
    print("\n--- 2. Padding Empty Tracking Labels ---")
    img_list = get_image_paths_from_yaml(data_yaml, split=target_split)
    
    padded_count = 0
    for img_path in img_list:
        txt_path = labels_dir / f"{Path(img_path).stem}.txt"
        if not txt_path.exists():
            txt_path.touch()
            padded_count += 1
            
    print(f"Padded {padded_count} empty txt files for frames without detections.")
    
    print("\n--- 3. Measuring Consistency ---")

    labels_path = os.path.join(save_dir, "labels")
    csv_path, json_path = get_all_detection_results(str(labels_path), stage_name, alpha, beta, img_list=img_list)
    print("\n✅ All processes finished!")
    return csv_path, json_path

def get_all_detection_results(path, stage_name, alpha=1.0, beta=0.1, img_list=None):
    """
    Get all detection txt files from path, compute object-level and frame-level
    consistency in natural frame order, and save the results.

    This function assumes the following helper functions already exist:
        - parse_yolo_txt(txt_path)
        - calculate_object_level_matches(preds_t, preds_t_plus_1, ...)
        - calculate_frame_level_consistency(preds_t, preds_t_plus_1, ...)
        - compare_boundary(r_score, ssim_score, alpha, beta)

    Output convention:
        - Frame-level CSV:
            key/path = frame T, representing transition T -> T+1
        - Object-level JSON:
            key/path = frame T, representing transition T -> T+1

    Args:
        path (str): Path to the directory containing detection txt files

    Returns:
        tuple: (frame_level_csv_path, object_level_json_path)
    """
    import json
    from pathlib import Path
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    txt_files = list(path.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No txt files found in: {path}")

    def extract_sequence_and_frame(file_path):
        """
        Parse sequence name and frame number from filenames like:
            MVI_0790_VIS_OB_frame0.txt
            MVI_0790_VIS_OB_frame5_jpg.rf.xxxxx.txt
        """
        name = Path(file_path).stem
        parts = name.split("_frame")
        if len(parts) == 2:
            seq = parts[0]
            tail = parts[1]
            try:
                frame_num = int(tail.split("_")[0].split(".")[0])
                return seq, frame_num
            except ValueError:
                return seq, -1
        return name, -1

    # Sort in natural temporal order
    sorted_txts = sorted(txt_files, key=extract_sequence_and_frame)

    output_dir = path / "consistency_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # eg,. path is ~/all_val_data_all/labels
    # frame_csv_path is  ~/all_val_data_all/all_val_data_all_frame_consistency.csv
    
    if stage_name is None:
        stage_name = path.name
    frame_csv_path = path.parent.parent / f"frame_consistency_{stage_name}.csv"
    object_json_path = path.parent.parent / f"object_consistency_{stage_name}.json"

    import cv2
    from skimage.metrics import structural_similarity as ssim

    # Build image lookup dictionary
    img_dict = {Path(p).stem: p for p in img_list} if img_list else {}

    def get_image_path(stem_name, possible_labels_dir):
        if stem_name in img_dict: 
            return img_dict[stem_name]
        
        # Fallback: deduce from labels directory (e.g., ../../images/stem.jpg)
        possible_img_dir = possible_labels_dir.parent.parent / "images"
        if possible_img_dir.exists():
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                img_path = possible_img_dir / (stem_name + ext)
                if img_path.exists():
                    img_dict[stem_name] = str(img_path)
                    return str(img_path)
        return None

    frame_rows = []
    object_results = {}

    for i in range(len(sorted_txts) - 1):
        txt_t = sorted_txts[i]
        txt_t1 = sorted_txts[i + 1]

        seq_t, frame_t = extract_sequence_and_frame(txt_t)
        seq_t1, frame_t1 = extract_sequence_and_frame(txt_t1)

        preds_t = parse_yolo_txt(str(txt_t))
        preds_t1 = parse_yolo_txt(str(txt_t1))

        # Default for non-consecutive or sequence switch transitions
        C_t = 1.0
        mean_match_coverage_ratio = 1.0
        match_pairs = []
        ssim_score = 1.0

        # Only compute for consecutive frames in the same sequence
        if seq_t == seq_t1:
            stem_t = Path(txt_t).stem
            stem_t1 = Path(txt_t1).stem
            
            img_t_path = get_image_path(stem_t, Path(txt_t).parent)
            img_t1_path = get_image_path(stem_t1, Path(txt_t1).parent)
            
            if img_t_path and img_t1_path:
                img1 = cv2.imread(img_t_path)
                img2 = cv2.imread(img_t1_path)
                if img1 is not None and img2 is not None:
                    if img1.shape != img2.shape:
                        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
                    ssim_score = float(ssim(img1, img2, channel_axis=-1))

            # If you do not require strict adjacency, replace this with only seq_t == seq_t1
            if frame_t1 > frame_t:
                C_t, mean_match_coverage_ratio, match_pairs = calculate_frame_level_consistency(
                    preds_t,
                    preds_t1,
                    apply_nms=True,
                    nms_iou_threshold=0.5,
                    nms_conf_threshold=0.0,
                    min_iou=0.05,
                    min_conf=0.0,
                    distance_ratio=1.5,
                    min_distance_px=0.1,  # normalized YOLO xywh coordinates
                )
                
        boundary_res = compare_boundary(C_t, ssim_score, alpha, beta)

        frame_rows.append({
            "path": Path(txt_t).stem,
            "next_path": Path(txt_t1).stem,
            "sequence": seq_t,
            "frame": frame_t,
            "next_frame": frame_t1,
            "frame_consistency": float(C_t),
            "mean_match_coverage_ratio": float(mean_match_coverage_ratio),
            "ssim_score": float(ssim_score),
            "boundary_result": int(boundary_res),
            "num_matches": int(len(match_pairs)),
            "original_index": i,
        })

        object_results[Path(txt_t).stem] = {
            "transition_from": Path(txt_t).stem,
            "transition_to": Path(txt_t1).stem,
            "frame_consistency": float(C_t),
            "mean_match_coverage_ratio": float(mean_match_coverage_ratio),
            "ssim_score": float(ssim_score),
            "boundary_result": int(boundary_res),
            "num_matches": int(len(match_pairs)),
            "matches": match_pairs,
        }

    # Add the last frame as an empty terminal record
    last_txt = sorted_txts[-1]
    seq_last, frame_last = extract_sequence_and_frame(last_txt)

    frame_rows.append({
        "path": Path(last_txt).stem,
        "next_path": None,
        "sequence": seq_last,
        "frame": frame_last,
        "next_frame": None,
        "frame_consistency": 1.0,
        "mean_match_coverage_ratio": 1.0,
        "ssim_score": 1.0,
        "boundary_result": 1,
        "num_matches": 0,
        "original_index": len(sorted_txts) - 1,
    })

    object_results[Path(last_txt).stem] = {
        "transition_from": Path(last_txt).stem,
        "transition_to": None,
        "frame_consistency": 1.0,
        "mean_match_coverage_ratio": 1.0,
        "ssim_score": 1.0,
        "boundary_result": 1,
        "num_matches": 0,
        "matches": [],
    }

    # Save outputs
    df = pd.DataFrame(frame_rows)
    df.to_csv(frame_csv_path, index=False)

    with open(object_json_path, "w", encoding="utf-8") as f:
        json.dump(object_results, f, indent=4, ensure_ascii=False)

    print(f"Saved frame-level consistency CSV to: {frame_csv_path}")
    print(f"Saved object-level consistency JSON to: {object_json_path}")

    return str(frame_csv_path), str(object_json_path)
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone YOLO Validation and Consistency Evaluation")
    parser.add_argument("--project", type=str, default="runs/consistency_eval", help="Root directory to save the all_val_xxxx folders")
    parser.add_argument("--weights", type=str, help="Path to the YOLO model weights (.pt file)")
    parser.add_argument("--data", type=str, help="Path to the target data configuration (.yaml file)")
    parser.add_argument("--txt-path", type=str, help="Path to the target txt files")
    parser.add_argument("--alpha-beta", type=str, default="1.0:0.1", help="Alpha and beta parameters for consistency boundary from calibration or fixed values alpha:beta, default 1.0:0.1")
 
    args = parser.parse_args()
    
    alpha, beta = args.alpha_beta.split(":")
    alpha = float(alpha)
    beta = float(beta)
    
    # User can specify weights (.pt) and data (.yaml) or only txt-path including labels
    if args.weights and args.data:
        if not os.path.exists(args.weights):
            raise FileNotFoundError(f"Model weights not found: {args.weights}")
        if not os.path.exists(args.data):
            raise FileNotFoundError(f"Data yaml not found: {args.data}")
        run_all_val_consistency(args.weights, args.data, args.project, alpha, beta)
    elif args.txt_path:
        if not os.path.exists(args.txt_path):
            raise FileNotFoundError(f"Txt path not found: {args.txt_path}")
        stage_name = args.txt_path.split("test_val_")[-1].split("/")[0]
        get_all_detection_results(args.txt_path, stage_name, alpha, beta)
    else:
        raise ValueError("Please provide either --weights and --data or --txt-path")
